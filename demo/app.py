import json
import sys
import io
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EON RMSA – Demo", page_icon="📡", layout="wide")

try:
    from demo import sim_core as sc
except Exception as exc:  
    st.error("Không import được Optical Networking Gym. Kiểm tra .venv và lệnh "
             "`pip install -e .` + `python setup.py build_ext --inplace`.")
    st.exception(exc)
    st.stop()

from demo import csv_loader as csvl

ALGO_NAMES = list(sc.ALGOS)
COLORS = sc.COLORS
MOD_COLORS = dict(zip(sc.MOD_ORDER, ["#9ecae1", "#4292c6", "#74c476", "#fdae6b", "#e6550d", "#756bb1"]))
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


@st.cache_resource(show_spinner=False)
def get_topology(name):
    return sc.load_topology_view(name)


def show_fig(fig, target=None):
    (target or st).pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ sidebar
st.title("📡 So sánh thuật toán RMSA trong mạng quang linh hoạt")
st.caption("ONG · mô phỏng dùng chung agent và cấu hình team/ · tab CSV đọc dữ liệu đã lưu. "
           "Các thuật toán được so sánh với cùng seed và cấu hình lưu lượng.")

with st.sidebar:
    st.header("🎛️ Cấu hình chung")
    topology = st.selectbox("Topology", list(sc.TOPOLOGIES), index=0,
                            help="nobel-eu là European Topology (28 nút, 41 link).")
    scenario = st.selectbox("Kịch bản", list(sc.SCENARIOS))
    scn = sc.SCENARIOS[scenario]
    st.info(scn["desc"])
    base_seed = st.number_input("Seed", 0, 10_000, 42, 1)
    use_power = st.checkbox("Đặt công suất phát (dBm)", value=False,
                            help="Bỏ chọn = dùng giá trị mặc định của scenario.")
    power = st.slider("Công suất phát mỗi kênh (dBm)", -8.0, 8.0, 0.0, 1.0, disabled=not use_power)
    launch_power = float(power) if use_power else None
    st.caption(f"Cấu hình: {topology} · {scn['num_slots']} slot · "
               f"bit rate {scn['bit_rates']} Gb/s · k=3 đường")

view = get_topology(topology)
tab_topo, tab_live, tab_sweep, tab_csv, tab_notes = st.tabs(
    ["🗺️ Topology", "🚀 Mô phỏng trực tiếp", "📊 Thí nghiệm & kết quả", "📂 Đọc dữ liệu từ csv", "📝 Ghi chú"])

# ------------------------------------------------------------------ tab topology
with tab_topo:
    st.subheader(f"Topology {topology}")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(9, 6))
        sc.draw_network(ax, view, label_nodes=True)
        show_fig(fig)
    with c2:
        km = np.array([l[3] for l in view.links])
        st.metric("Số nút", len(view.names))
        st.metric("Số link", len(view.links))
        st.metric("Độ dài link trung bình", f"{km.mean():.0f} km")
        st.metric("Link dài nhất", f"{km.max():.0f} km")
        st.caption(f"Đọc từ file `{view.source_file}` của thư viện. "
                   + ("Vị trí nút theo toạ độ địa lý." if view.has_geo
                      else "File không có toạ độ nên nút được xếp theo vòng tròn."))
    with st.expander("Danh sách link"):
        st.dataframe(pd.DataFrame(
            [(view.names[s], view.names[t], round(k, 1)) for _, s, t, k in view.links],
            columns=["Nút 1", "Nút 2", "Độ dài (km)"]), hide_index=True)

# ------------------------------------------------------------------ tab live
with tab_live:
    st.subheader("Mô phỏng từng yêu cầu của hai thuật toán")
    c1, c2, c3, c4 = st.columns(4)
    live_load = c1.select_slider("Tải chào (Erlang)", [50, 100, 150, 200, 250, 300, 400, 500, 600], 300)
    live_n = c2.slider("Số yêu cầu", 50, 1000, 200, 50)
    refresh = c3.slider("Cập nhật hình mỗi N yêu cầu", 1, 20, 5)
    delay = c4.slider("Chờ giữa các lần cập nhật (s)", 0.0, 0.5, 0.05, 0.05)
    st.caption("Ảnh phổ lấy ngay sau hành động, trước khi tiến thời gian. Màu link = mức sử dụng phổ (vàng nhạt → đỏ). Đường xanh = tuyến vừa được cấp phát; "
               "vòng đỏ + nét đứt = yêu cầu bị chặn. Ô đen ở ma trận phổ = slot đang bận.")

    if st.button("▶️ Bắt đầu mô phỏng", type="primary"):
        envs, infos, recs, aux = {}, {}, {a: [] for a in ALGO_NAMES}, {}
        try:
            for a in ALGO_NAMES:
                envs[a] = sc.make_env_for(topology, live_load, live_n, scn["num_slots"],
                                          scn["bit_rates"], base_seed,
                                          launch_power_dbm=launch_power)
                _, infos[a] = envs[a].reset(seed=int(base_seed))

            cols = st.columns(len(ALGO_NAMES))
            ph = {}
            for col, a in zip(cols, ALGO_NAMES):
                with col:
                    st.markdown(f"### {a}")
                    ph[a] = {k: st.empty() for k in ("metrics", "net", "spec", "log")}
            chart_ph = st.empty()
            progress = st.progress(0.0)
            mismatch = False
            done = {a: False for a in ALGO_NAMES}

            def render():
                for a in ALGO_NAMES:
                    if not recs[a]:
                        continue
                    r, x = recs[a][-1], aux[a]
                    df = pd.DataFrame(recs[a])
                    with ph[a]["metrics"].container():
                        m = st.columns(4)
                        m[0].metric("Đã xử lý", len(df))
                        m[1].metric("Bị chặn", int((~df.accepted).sum()))
                        m[2].metric("SBP", f"{100 * r['sbp_cum']:.2f}%")
                        m[3].metric("BBR", f"{100 * r['bbr_cum']:.2f}%")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ttl = (f"#{len(df)}  {r['src']} → {r['dst']}  {r['bit_rate']}G  "
                           + (f"✔ {r['modulation']}, {r['path_km']:.0f} km" if r["accepted"] else "✘ bị chặn"))
                    sc.draw_network(ax, view, x["link_util"], x["path_link_ids"], r["accepted"],
                                    x["endpoints"], title=ttl)
                    show_fig(fig, ph[a]["net"])
                    fig, ax = plt.subplots(figsize=(6, 2.2))
                    sc.draw_spectrum(ax, x["occupied"], "Ma trận phổ (link × slot)")
                    show_fig(fig, ph[a]["spec"])
                    ph[a]["log"].dataframe(
                        df.tail(8)[::-1][["src", "dst", "bit_rate", "accepted", "modulation",
                                          "gsnr_db", "path_km", "slot"]].round(1),
                        hide_index=True)
                chart = pd.DataFrame({a: [100 * r["sbp_cum"] for r in recs[a]] for a in ALGO_NAMES
                                      if recs[a]})
                chart.index = range(1, len(chart) + 1)
                chart_ph.line_chart(chart, y_label="SBP tích lũy (%)", x_label="Yêu cầu")

            n_done = 0
            while not all(done.values()):
                for a in ALGO_NAMES:
                    if done[a]:
                        continue
                    rec, infos[a], done[a], aux[a] = sc.step_request(
                        envs[a], sc.ALGOS[a], infos[a])
                    recs[a].append(rec)
                n_done += 1
                if len(ALGO_NAMES) > 1 and not mismatch:
                    k = [(recs[a][-1]["src"], recs[a][-1]["dst"], recs[a][-1]["bit_rate"])
                         for a in ALGO_NAMES if recs[a]]
                    if len(set(k)) > 1:
                        mismatch = True
                progress.progress(min(n_done / live_n, 1.0))
                if n_done % refresh == 0 or all(done.values()):
                    render()
                    if delay > 0:
                        time.sleep(delay)

            if mismatch:
                st.warning("Hai thuật toán nhận chuỗi yêu cầu khác nhau, so sánh có thể không công bằng.")
            st.success("Hoàn tất.")
            summ = []
            for a in ALGO_NAMES:
                df = pd.DataFrame(recs[a])
                summ.append({"Thuật toán": a, "Yêu cầu": len(df), "Chấp nhận": int(df.accepted.sum()),
                             "Bị chặn": int((~df.accepted).sum()),
                             "SBP (%)": round(100 * df.sbp_cum.iloc[-1], 2),
                             "BBR (%)": round(100 * df.bbr_cum.iloc[-1], 2),
                             "Thời gian TB (ms/yêu cầu)": round(df.ms.mean(), 3),
                             "Sử dụng phổ TB (%)": round(100 * df.util.mean(), 1)})
            st.dataframe(pd.DataFrame(summ), hide_index=True)
            st.caption("Một lần chạy ngắn chỉ mang tính minh hoạ; sự khác biệt giữa hai thuật toán "
                       "thường nằm trong dao động ngẫu nhiên. Dùng tab Thí nghiệm để có kết quả nhiều seed.")
        except Exception as exc:
            st.exception(exc)
            st.error("Mô phỏng dừng do lỗi từ thư viện hoặc cấu hình.")
        finally:
            for e in envs.values():
                try:
                    e.close()
                except Exception:
                    pass

# ------------------------------------------------------------------ tab sweep
with tab_sweep:
    st.subheader("Thí nghiệm nhiều mức tải / công suất phát, nhiều seed")
    c1, c2, c3 = st.columns(3)
    x_choice = c1.radio("Biến cần quét", ["Tải chào (Erlang)", "Công suất phát (dBm)"])
    x_name = "load" if x_choice.startswith("Tải") else "launch_power_dbm"
    if x_name == "load":
        x_values = c1.multiselect("Các mức tải", [50, 100, 150, 200, 250, 300, 400, 500, 600],
                                  [100, 200, 300, 400])
        fixed_load = 300
    else:
        x_values = c1.multiselect("Các mức công suất (dBm)", list(range(-8, 9, 2)), [-6, -2, 0, 2, 6])
        fixed_load = c1.slider("Tải cố định (Erlang)", 50, 600, 300, 50)
    algos_sel = c2.multiselect("Thuật toán", ALGO_NAMES, ALGO_NAMES)
    n_req = c2.slider("Số yêu cầu mỗi lần chạy", 200, 5000, 1000, 100)
    n_seeds = c3.slider("Số seed", 1, 10, 3)
    warm = c3.slider("Bỏ phần đầu làm warm-up (%)", 0, 30, 0, 5,
                     help="Mạng bắt đầu rỗng nên các yêu cầu đầu ít bị chặn; bỏ bớt để gần trạng thái ổn định.")
    n_runs = len(algos_sel) * len(x_values) * n_seeds
    st.caption(f"{n_runs} lần chạy · ước tính khoảng {n_runs * n_req * 0.002:.0f} giây")

    cfg = dict(topology=topology, scenario=scenario, x_name=x_name, x_values=sorted(x_values),
               fixed_load=fixed_load, power=launch_power, n_req=n_req, algos=algos_sel,
               seeds=[int(base_seed) + i for i in range(n_seeds)])
    if st.button("▶️ Chạy thí nghiệm", type="primary", disabled=not (x_values and algos_sel)):
        bar = st.progress(0.0, text="Đang chạy...")
        try:
            log = sc.run_sweep(algos_sel, topology, scenario, x_name, cfg["x_values"], fixed_load,
                               launch_power, n_req, cfg["seeds"],
                               progress=lambda f, t: bar.progress(f, text=t))
            st.session_state["sweep"] = dict(cfg=cfg, log=log)
            bar.empty()
        except Exception as exc:
            bar.empty()
            st.exception(exc)

    sw = st.session_state.get("sweep")
    if sw is None:
        st.info("Chọn cấu hình rồi bấm **Chạy thí nghiệm**. Kết quả (biểu đồ, bảng, file CSV) sẽ hiện ở đây.")
    else:
        log, scfg = sw["log"], sw["cfg"]
        if scfg != cfg:
            st.warning("Kết quả bên dưới ứng với cấu hình đã chạy trước đó (khác cấu hình hiện tại).")
        runs = sc.summarize_runs(log, warm / 100)
        agg = sc.aggregate_runs(runs)
        xl = "Tải chào (Erlang)" if scfg["x_name"] == "load" else "Công suất phát (dBm)"
        with st.expander("Cấu hình của lần chạy đang hiển thị", expanded=False):
            st.json({k: v for k, v in scfg.items()})
        use_log = st.checkbox("Thang log cho SBP/BBR", value=False)

        def line_chart(ycol, ylabel, log_scale=False):
            fig, ax = plt.subplots(figsize=(6, 3.8))
            for a in scfg["algos"]:
                d = agg[agg.algo == a]
                if log_scale:
                    d = d[d[ycol] > 0]
                ax.errorbar(d.x, d[ycol], yerr=d[ycol + "_std"].fillna(0), marker="o", capsize=3,
                            label=a, color=COLORS[a], lw=2)
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel(ylabel)
            ax.legend()
            return fig

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Tỷ lệ chặn yêu cầu (SBP)** — trung bình ± độ lệch chuẩn giữa các seed")
            show_fig(line_chart("sbp", "SBP", use_log))
        with r2:
            st.markdown("**Tỷ lệ chặn bit rate (BBR)**")
            show_fig(line_chart("bbr", "BBR", use_log))
        r3, r4 = st.columns(2)
        with r3:
            st.markdown("**Thời gian xử lý trung bình**")
            show_fig(line_chart("avg_ms", "ms / yêu cầu"))
        with r4:
            st.markdown("**Mức sử dụng phổ trung bình**")
            show_fig(line_chart("util_pct", "Sử dụng phổ (%)"))

        st.markdown("---")
        pick = st.select_slider(f"Xem GSNR và điều chế tại {xl.lower()} =", scfg["x_values"],
                                value=scfg["x_values"][len(scfg["x_values"]) // 2])
        acc = log[(log.x == pick) & log.accepted]
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**GSNR theo độ dài tuyến** (mỗi chấm là một kết nối được chấp nhận)")
            thr = sc.modulation_table()
            fig, axes = plt.subplots(1, len(scfg["algos"]), figsize=(6, 3.8), sharey=True, squeeze=False)
            for ax, a in zip(axes[0], scfg["algos"]):
                d = acc[acc.algo == a]
                for m in sc.MOD_ORDER:
                    dm = d[d.modulation == m]
                    if len(dm):
                        ax.scatter(dm.path_km, dm.gsnr_db, s=10, alpha=0.6, label=m, color=MOD_COLORS[m])
                for m, (t, _) in thr.items():
                    if m in set(d.modulation):
                        ax.axhline(t, color=MOD_COLORS[m], ls=":", lw=1)
                ax.set_title(a, fontsize=10)
                ax.set_xlabel("Độ dài tuyến (km)")
            axes[0][0].set_ylabel("GSNR (dB)")
            axes[0][-1].legend(fontsize=7, title="Điều chế", title_fontsize=7)
            fig.tight_layout()
            show_fig(fig)
            st.caption("Đường chấm = ngưỡng GSNR tối thiểu của từng điều chế (lấy từ thư viện).")
        with g2:
            st.markdown("**Phân bố định dạng điều chế** (% kết nối được chấp nhận)")
            fig, ax = plt.subplots(figsize=(6, 3.8))
            w = 0.8 / max(len(scfg["algos"]), 1)
            for i, a in enumerate(scfg["algos"]):
                d = acc[acc.algo == a]
                pct = [100 * (d.modulation == m).mean() if len(d) else 0 for m in sc.MOD_ORDER]
                ax.bar(np.arange(len(sc.MOD_ORDER)) + i * w, pct, w, label=a, color=COLORS[a])
            ax.set_xticks(np.arange(len(sc.MOD_ORDER)) + w * (len(scfg["algos"]) - 1) / 2)
            ax.set_xticklabels(sc.MOD_ORDER)
            ax.set_ylabel("%")
            ax.legend()
            show_fig(fig)

        st.markdown("---")
        st.markdown("**Bảng kết quả (trung bình các seed)**")
        st.dataframe(agg.rename(columns={"algo": "Thuật toán", "x": xl}).round(4), hide_index=True)
        metadata = dict(log.attrs.get("metadata", {}), ui_config=scfg, warmup_frac=warm / 100)
        st.download_button("⬇️ Tải cấu hình và phiên bản (JSON)",
                           json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
                           "experiment_config.json", "application/json")
        st.download_button("⬇️ Tải bảng tổng hợp (CSV)",
                           agg.to_csv(index=False).encode("utf-8-sig"), "summary.csv", "text/csv")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Tải bảng theo từng seed (CSV)",
                           runs.to_csv(index=False).encode("utf-8-sig"), "ket_qua_theo_seed.csv", "text/csv")
        d2.download_button("⬇️ Tải log từng yêu cầu (CSV)",
                           log.to_csv(index=False).encode("utf-8-sig"), "log_tung_yeu_cau.csv", "text/csv")



# ------------------------------------------------------------------ tab csv của nhóm
_EXTRA_COLORS = ["#7f7f7f", "#8c564b", "#17becf", "#bcbd22", "#9467bd", "#e377c2"]
_ALGO_COLORS = {"KSP-FF": COLORS["KSP-FF"], "LS-BM-KSP": COLORS["LS-BM-KSP"], "BM-LS-KSP": "#2e86ab"}


def algo_color(name, order):
    return _ALGO_COLORS.get(name, _EXTRA_COLORS[order % len(_EXTRA_COLORS)])


@st.cache_data(show_spinner=False)
def parse_csv_cached(raw, name):
    return csvl.load_csv(raw, name)


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


def show_with_png(fig, fname, key):
    data = png_bytes(fig)
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("⬇️ Tải PNG", data, fname, "image/png", key=key)


def xy_figure(agg, ycol, ylabel, xlabel, algos, title, log_scale=False, multiplier=1.0):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, a in enumerate(algos):
        d = agg[agg.algo == a]
        if ycol not in d.columns or d[ycol].isna().all():
            continue
        if log_scale:
            d = d[d[ycol] > 0]
        err = d[ycol + "_std"].fillna(0) * multiplier if ycol + "_std" in d.columns else None
        ax.errorbar(d.x, d[ycol] * multiplier, yerr=err, marker="o", capsize=3, lw=2, label=a,
                    color=algo_color(a, i))
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig


with tab_csv:
    st.subheader("Biểu đồ từ dữ liệu CSV của nhóm")
    st.caption("Tab này chỉ đọc và vẽ file CSV (không chạy mô phỏng)")
    st.caption("Dữ liệu cũ: data/member4/kb1, kb2, kb3. Sử dụng phổ cũ lấy sau khi tiến thời gian; "
               "kết quả chạy mới lấy ngay sau hành động. Không so sánh trực tiếp hai cách lấy mẫu.")
    uploads = st.file_uploader("Chọn một hoặc nhiều file CSV", type=["csv"], accept_multiple_files=True)
    folder = st.text_input("Hoặc đọc mọi file .csv trong một thư mục (tuỳ chọn, đường dẫn tính từ nơi chạy Streamlit)",
                           value="", placeholder="ví dụ: team_csv")
    sources = [(u.name, u.getvalue()) for u in (uploads or [])]
    if folder.strip():
        fp = Path(folder.strip())
        if fp.is_dir():
            sources += [(q.name, q.read_bytes()) for q in sorted(fp.glob("*.csv"))]
        else:
            st.warning(f"Không thấy thư mục `{folder}`.")

    if not sources:
        st.info("Chưa có file. Hãy tải lên hoặc nhập thư mục chứa CSV.")
    else:
        xmode = st.radio("Trục ngang của dữ liệu", ["Tải chào (Erlang)", "Công suất phát (dBm)"], horizontal=True,)
        xlabel = xmode
        warm_csv = st.slider("Warm-up khi tính từ log từng yêu cầu (%)", 0, 30, 0, 5) / 100

        datasets, labels, errors = {}, {}, []
        for name, raw in sources:
            try:
                d = parse_csv_cached(raw, name)
            except Exception as exc:
                errors.append(f"`{name}`: không đọc được ({exc})")
                continue
            label = st.text_input(f"Tên hiển thị cho `{name}`  (nhận dạng: {d['kind']})",
                                  value=Path(name).stem, key=f"lbl_{name}")
            pts = d["pts"]
            if d["kind"] == "unknown":
                cols = ["(không có)"] + list(d["raw"].columns)
                with st.expander(f"Ghép cột thủ công cho `{name}`", expanded=True):
                    c = st.columns(3)
                    lc = c[0].selectbox("Cột tải/x", cols, key=f"mx_{name}")
                    sb = c[1].selectbox("Cột SBP", cols, key=f"ms_{name}")
                    bb = c[2].selectbox("Cột BBR", cols, key=f"mb_{name}")
                    c = st.columns(3)
                    ac = c[0].selectbox("Cột thuật toán", cols, key=f"ma_{name}")
                    uc = c[1].selectbox("Cột sử dụng phổ", cols, key=f"mu_{name}")
                    tc = c[2].selectbox("Cột thời gian", cols, key=f"mt_{name}")
                    none = lambda v: None if v == "(không có)" else v
                    if lc != "(không có)" and sb != "(không có)":
                        pts = csvl.pts_from_mapping(d["raw"], lc, sb, none(bb), none(ac), label,
                                                    none(uc), none(tc))
                    else:
                        st.info("Chọn ít nhất cột tải và cột SBP.")
            elif d["kind"] == "log":
                pts = csvl.pts_from_log(d["log"], warm_csv)
            if pts is not None and not pts.empty:
                datasets[label] = dict(pts=pts, log=d["log"], file=name, kind=d["kind"])
        for e in errors:
            st.error(e)

        if not datasets:
            st.stop()

        # ---- kiểm tra dữ liệu
        st.markdown("### 🔎 Kiểm tra nhanh dữ liệu")
        st.caption("Chỉ nêu dấu hiệu để bạn hỏi lại nguồn dữ liệu; không phải kết luận dữ liệu đúng hay sai.")
        any_flag = False
        for lab, ds in datasets.items():
            notes = csvl.check_dataset(ds["pts"], lab)
            if notes:
                any_flag = True
                with st.expander(f"{lab}: {len(notes)} dấu hiệu cần lưu ý", expanded=False):
                    for n in notes:
                        st.warning(n)
        cross = csvl.cross_checks({lab: ds["pts"] for lab, ds in datasets.items()})
        if cross:
            any_flag = True
            with st.expander(f"Giữa các file: {len(cross)} dấu hiệu cần lưu ý", expanded=False):
                for n in cross:
                    st.warning(n)
        if not any_flag:
            st.success("Các kiểm tra tự động không thấy dấu hiệu bất thường (điều này không đảm bảo dữ liệu đúng).")

        # ---- biểu đồ của một file
        st.markdown("### 📈 Biểu đồ")
        pick_lab = st.selectbox("Chọn file / kịch bản để vẽ", list(datasets))
        ds = datasets[pick_lab]
        pts = ds["pts"]
        all_algos = list(dict.fromkeys(pts["algo"]))
        algos_c = st.multiselect("Thuật toán", all_algos, all_algos, key=f"csv_algos_{pick_lab}")
        log_c = st.checkbox("Thang log cho SBP/BBR", value=False, key=f"csv_log_{pick_lab}")
        as_pct = st.checkbox("Hiển thị tỷ lệ chặn theo %", value=False, key=f"csv_pct_{pick_lab}")
        agg = csvl.aggregate_points(pts, algos_c) if algos_c else None
        src_note = f"Nguồn: {ds['file']} ({pick_lab})"
        if agg is not None and len(agg):
            mult = 100.0 if as_pct else 1.0
            unit = " (%)" if as_pct else ""
            charts = [("sbp", "SBP" + unit, True), ("bbr", "BBR" + unit, True),
                      ("util_pct", "Sử dụng phổ (%)", False), ("avg_ms", "ms / yêu cầu", False)]
            avail = [c for c in charts if c[0] in agg.columns and agg[c[0]].notna().any()]
            cols = st.columns(2)
            for i, (col, yl, is_rate) in enumerate(avail):
                with cols[i % 2]:
                    fig = xy_figure(agg, col, yl, xlabel, algos_c, f"{csvl.METRICS[col]}\n{src_note}",
                                    log_scale=log_c and is_rate, multiplier=mult if is_rate else 1.0)
                    show_with_png(fig, f"{Path(ds['file']).stem}_{col}.png", f"dl_{pick_lab}_{col}")
            missing = [csvl.METRICS[c] for c in ("util_pct", "avg_ms") if c not in agg.columns]
            if missing:
                st.caption("File này không có cột: " + ", ".join(missing) + ".")

            # mức cải thiện giữa hai thuật toán
            if len(algos_c) >= 2 and "sbp" in agg.columns:
                c1, c2 = st.columns(2)
                ref = c1.selectbox("Thuật toán gốc", algos_c, 0, key=f"csv_ref_{pick_lab}")
                cmp_opts = [a for a in algos_c if a != ref]
                other = c2.selectbox("Thuật toán so sánh", cmp_opts, 0, key=f"csv_cmp_{pick_lab}")
                m = agg[agg.algo == ref][["x", "sbp"]].merge(agg[agg.algo == other][["x", "sbp"]], on="x",
                                                              suffixes=("_ref", "_cmp"))
                m = m[m.sbp_ref > 0]
                if len(m):
                    fig, ax = plt.subplots(figsize=(6.5, 3.8))
                    ax.plot(m.x, 100 * (m.sbp_ref - m.sbp_cmp) / m.sbp_ref, "o-", lw=2, color="#2ca02c")
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(f"Giảm SBP của {other} so với {ref} (%)")
                    ax.set_title(f"Mức cải thiện SBP\n{src_note}", fontsize=10)
                    fig.tight_layout()
                    show_with_png(fig, f"{Path(ds['file']).stem}_cai_thien.png", f"dl_{pick_lab}_imp")
                    st.caption("Giá trị dương = thuật toán so sánh chặn ít hơn. Nếu file chỉ có một seed thì "
                               "chênh lệch nhỏ có thể chỉ là ngẫu nhiên.")

        # ---- GSNR và điều chế 
        if ds["log"] is not None:
            st.markdown("### 📡 GSNR và điều chế (từ log từng yêu cầu)")
            lg = ds["log"].copy()
            lg["accepted"] = lg["accepted"].astype(bool)
            xs = sorted(lg["x"].unique())
            xp = st.select_slider("Mức x để xem", xs, value=xs[len(xs) // 2], key=f"csv_xpick_{pick_lab}")
            acc = lg[(lg.x == xp) & lg.accepted & lg.algo.isin(algos_c)]
            thr = {}
            try:
                thr = sc.modulation_table()
            except Exception:
                pass
            g1, g2 = st.columns(2)
            with g1:
                fig, axes = plt.subplots(1, max(len(algos_c), 1), figsize=(6.5, 4), sharey=True, squeeze=False)
                for ax, a in zip(axes[0], algos_c):
                    d = acc[acc.algo == a]
                    for m_ in sc.MOD_ORDER:
                        dm = d[d.modulation == m_]
                        if len(dm):
                            ax.scatter(dm.path_km, dm.gsnr_db, s=10, alpha=0.6, label=m_, color=MOD_COLORS[m_])
                    for m_, (t, _) in thr.items():
                        if m_ in set(d.modulation):
                            ax.axhline(t, color=MOD_COLORS[m_], ls=":", lw=1)
                    ax.set_title(a, fontsize=10)
                    ax.set_xlabel("Độ dài tuyến (km)")
                axes[0][0].set_ylabel("GSNR (dB)")
                axes[0][-1].legend(fontsize=7, title="Điều chế", title_fontsize=7)
                fig.suptitle(f"GSNR theo độ dài tuyến, x = {xp}\n{src_note}", fontsize=9)
                fig.tight_layout()
                show_with_png(fig, f"{Path(ds['file']).stem}_gsnr.png", f"dl_{pick_lab}_gsnr")
            with g2:
                fig, ax = plt.subplots(figsize=(6.5, 4))
                w = 0.8 / max(len(algos_c), 1)
                for i, a in enumerate(algos_c):
                    d = acc[acc.algo == a]
                    pct = [100 * (d.modulation == m_).mean() if len(d) else 0 for m_ in sc.MOD_ORDER]
                    ax.bar(np.arange(len(sc.MOD_ORDER)) + i * w, pct, w, label=a, color=algo_color(a, i))
                ax.set_xticks(np.arange(len(sc.MOD_ORDER)) + w * (len(algos_c) - 1) / 2)
                ax.set_xticklabels(sc.MOD_ORDER)
                ax.set_ylabel("% kết nối được chấp nhận")
                ax.set_title(f"Phân bố điều chế, x = {xp}\n{src_note}", fontsize=10)
                ax.legend()
                fig.tight_layout()
                show_with_png(fig, f"{Path(ds['file']).stem}_dieu_che.png", f"dl_{pick_lab}_mod")
        else:
            st.caption("File này không có log từng yêu cầu nên không vẽ được GSNR theo khoảng cách và phân bố điều chế "
                       "(cần file `log_tung_yeu_cau.csv`).")

        # ---- so sánh các kịch bản
        if len(datasets) >= 2:
            st.markdown("### 🔀 So sánh các kịch bản")
            all_a = list(dict.fromkeys(a for d_ in datasets.values() for a in d_["pts"]["algo"]))
            c1, c2 = st.columns(2)
            a_sel = c1.selectbox("Thuật toán", all_a, key="cmp_algo")
            metric = c2.selectbox("Chỉ số", ["sbp", "bbr", "util_pct", "avg_ms"],
                                  format_func=lambda k: csvl.METRICS[k], key="cmp_metric")
            fig, ax = plt.subplots(figsize=(7, 4))
            drawn = 0
            for i, (lab, d_) in enumerate(datasets.items()):
                if metric not in d_["pts"].columns:
                    continue
                ag = csvl.aggregate_points(d_["pts"], [a_sel])
                if metric not in ag.columns or ag.empty:
                    continue
                ax.errorbar(ag.x, ag[metric], yerr=ag[metric + "_std"].fillna(0), marker="o", capsize=3, lw=2,
                            label=lab, color=_EXTRA_COLORS[i % len(_EXTRA_COLORS)] if i > 1 else ["#d1495b", "#2e86ab"][i])
                drawn += 1
            if drawn:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(csvl.METRICS[metric])
                ax.set_title(f"{csvl.METRICS[metric]} của {a_sel} theo kịch bản", fontsize=10)
                ax.legend(fontsize=8)
                fig.tight_layout()
                show_with_png(fig, f"so_sanh_kich_ban_{metric}.png", "dl_cmp")
            else:
                plt.close(fig)
                st.info("Các file đã chọn không có chỉ số hoặc thuật toán này.")
