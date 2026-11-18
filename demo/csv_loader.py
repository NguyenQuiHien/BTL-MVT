from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"sbp": "Tỷ lệ chặn yêu cầu (SBP)", "bbr": "Tỷ lệ chặn bit rate (BBR)",
           "util_pct": "Sử dụng phổ (%)", "avg_ms": "Thời gian xử lý (ms/yêu cầu)"}


# ----------------------------------------------------------------------------- đọc file
def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_table(raw: bytes):
    """Tự nhận dấu phân cách (',' hoặc ';') và dấu thập phân (',' khi file dùng ';')."""
    text = _decode(raw)
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        num = pd.to_numeric(s.str.replace(",", ".", regex=False) if sep == ";" else s, errors="coerce")
        df[c] = num if num.notna().mean() >= 0.9 else df[c].astype(str).str.strip()
    return df, sep


def detect_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"gsnr_db", "accepted", "modulation", "algo", "idx"} <= cols:
        return "log"
    if {"algo", "x", "sbp", "bbr"} <= cols:
        return "runs"
    if "Load_Erlangs" in cols and any(c.startswith("SBP_") for c in cols):
        return "wide"
    if {"load", "episode_service_blocking_rate"} <= cols:
        return "sweep"
    return "unknown"


def _algo_label(s: str) -> str:
    return s.replace("_", "-")


def _first(df, prefix):
    hit = [c for c in df.columns if c.startswith(prefix)]
    return hit[0] if hit else None


def normalize(df: pd.DataFrame, kind: str, filename: str):
    """-> (pts, log). pts: bảng điểm [algo, x, sbp, bbr, util_pct, avg_ms, seed, requests]."""
    log = None
    if kind == "wide":
        parts = []
        for a in [c[4:] for c in df.columns if c.startswith("SBP_")]:
            d = pd.DataFrame({"algo": _algo_label(a), "x": df["Load_Erlangs"], "sbp": df[f"SBP_{a}"]})
            if f"BBR_{a}" in df.columns:
                d["bbr"] = df[f"BBR_{a}"]
            u, r = _first(df, f"SpecUtil_{a}"), _first(df, f"Runtime_{a}")
            if u:
                d["util_pct"] = df[u]
            if r:
                d["avg_ms"] = df[r]
            parts.append(d)
        pts = pd.concat(parts, ignore_index=True)
    elif kind == "sweep":
        m = re.search(r"__(.+?)__", Path(filename).stem)
        algo = _algo_label(m.group(1)) if m else Path(filename).stem
        pts = pd.DataFrame({"algo": algo, "x": df["load"], "sbp": df["episode_service_blocking_rate"]})
        if "episode_bit_rate_blocking_rate" in df.columns:
            pts["bbr"] = df["episode_bit_rate_blocking_rate"]
        if "episode_services_processed" in df.columns:
            pts["requests"] = df["episode_services_processed"]
    elif kind == "runs":
        keep = [c for c in ["algo", "x", "seed", "requests", "sbp", "bbr", "avg_ms", "util_pct", "sbp_std", "bbr_std", "avg_ms_std", "util_pct_std", "seeds"] if c in df.columns]
        pts = df[keep].copy()
    elif kind == "log":
        log = df.copy()
        log["accepted"] = log["accepted"].astype(str).str.lower().isin(["true", "1", "1.0"])
        pts = pts_from_log(log, 0.0)
    else:
        pts = None
    return pts, log


def pts_from_log(log: pd.DataFrame, warmup_frac: float = 0.0) -> pd.DataFrame:
    out = []
    for (algo, x, seed), g in log.groupby(["algo", "x", "seed"], sort=False):
        g = g[g["idx"] >= int(len(g) * warmup_frac)]
        blocked = ~g["accepted"]
        out.append(dict(algo=algo, x=x, seed=seed, requests=len(g), sbp=float(blocked.mean()),
                        bbr=float(g.loc[blocked, "bit_rate"].sum() / g["bit_rate"].sum()),
                        avg_ms=float(g["ms"].mean()), util_pct=100.0 * float(g["util"].mean())))
    return pd.DataFrame(out)


def pts_from_mapping(df, load_col, sbp_col, bbr_col=None, algo_col=None, algo_name="thuật toán",
                     util_col=None, ms_col=None):
    pts = pd.DataFrame({"x": df[load_col], "sbp": df[sbp_col]})
    pts["algo"] = df[algo_col].astype(str) if algo_col else algo_name
    for name, col in (("bbr", bbr_col), ("util_pct", util_col), ("avg_ms", ms_col)):
        if col:
            pts[name] = df[col]
    return pts


def aggregate_points(pts: pd.DataFrame, algos=None) -> pd.DataFrame:
    d = pts if algos is None else pts[pts["algo"].isin(algos)]
    if "seed" not in d.columns and not d.duplicated(["algo", "x"]).any() and any(c.endswith("_std") for c in d.columns):
        return d.sort_values(["algo", "x"]).copy()
    cols = [c for c in METRICS if c in d.columns and d[c].notna().any()]
    if d.empty or not cols:
        return pd.DataFrame(columns=["algo", "x"])
    g = d.groupby(["algo", "x"])[cols].agg(["mean", "std"])
    g.columns = [a if b == "mean" else f"{a}_std" for a, b in g.columns]
    return g.reset_index().sort_values(["algo", "x"])


# ----------------------------------------------------------------------------- kiểm tra
def check_dataset(pts: pd.DataFrame, label: str) -> list[str]:
    """Nêu các dấu hiệu cần hỏi lại nguồn dữ liệu (không kết luận đúng/sai thay người gửi)."""
    notes = []
    if pts is None or pts.empty:
        return notes
    for col in ("sbp", "bbr"):
        if col in pts.columns and pts[col].max() > 1.0:
            notes.append(f"**{col.upper()} có giá trị lớn hơn 1** (lớn nhất {pts[col].max():.4g}). Tỷ lệ chặn phải nằm "
                         "trong khoảng 0–1; thường do Excel hoặc định dạng vùng đọc dấu chấm thập phân thành dấu "
                         "phân cách hàng nghìn (0.13 thành 13). Nên lấy lại file gốc do script xuất ra.")
    for algo, g in pts.groupby("algo"):
        g = g.groupby("x", as_index=False).mean(numeric_only=True).sort_values("x")
        if len(g) < 3:
            continue
        if "sbp" in g and g["sbp"].nunique() == 1:
            notes.append(f"**{algo}: SBP giống hệt nhau ở mọi mức tải.** Có thể mỗi điểm quá ngắn để thấy ảnh "
                         "hưởng của tải, hoặc tải không được áp dụng.")
        if {"sbp", "bbr"} <= set(g.columns) and g["sbp"].max() <= 1 and g["bbr"].max() <= 1:
            ok = g["sbp"] > 0
            ratio = (g.loc[ok, "bbr"] / g.loc[ok, "sbp"])
            if len(ratio) >= 3 and ratio.std() / ratio.mean() < 1e-3:
                notes.append(f"**{algo}: tỷ lệ BBR/SBP không đổi ({ratio.mean():.3f}) ở mọi mức tải.** Số đo từ "
                             "mô phỏng thật hiếm khi đều như vậy; nên hỏi cách số liệu được tạo ra.")
        for col, name in (("util_pct", "mức sử dụng phổ"), ("avg_ms", "thời gian xử lý")):
            if col in g and len(g) >= 4 and g[col].nunique() > 1:
                r = np.corrcoef(g["x"], g[col])[0, 1]
                if r * r > 0.9999:
                    notes.append(f"**{algo}: {name} tăng tuyến tính gần như hoàn hảo theo tải** (R² = {r*r:.5f}). "
                                 "Đường cong từ mô phỏng thật thường cong hoặc nhiễu hơn.")
    has_multiple = ("seed" in pts and pts["seed"].nunique() > 1) or ("seeds" in pts and (pts["seeds"] > 1).all())
    if not has_multiple:
        notes.append("**Chỉ có một seed (hoặc file không ghi seed)**, nên không có thanh lỗi và kết quả có thể là ngẫu nhiên.")
    if "requests" in pts.columns and pts["requests"].notna().any() and pts["requests"].max() < 500:
        notes.append(f"**Mỗi điểm chỉ có {int(pts['requests'].max())} yêu cầu**, quá ít cho tỷ lệ chặn ổn định "
                     "(mạng còn gần như rỗng).")
    return notes


def cross_checks(datasets: dict) -> list[str]:
    """So sánh giữa các file: cột giống hệt nhau ở hai kịch bản khác nhau."""
    notes, names = [], list(datasets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = datasets[names[i]], datasets[names[j]]
            if a is None or b is None:
                continue
            for algo in set(a["algo"]) & set(b["algo"]):
                m = a[a["algo"] == algo].merge(b[b["algo"] == algo], on="x", suffixes=("_a", "_b"))
                if len(m) < 3:
                    continue
                for col, name in (("util_pct", "mức sử dụng phổ"), ("avg_ms", "thời gian xử lý")):
                    ca, cb = f"{col}_a", f"{col}_b"
                    if ca in m and cb in m and m[ca].notna().all() and np.allclose(m[ca], m[cb]):
                        notes.append(f"**{algo}: cột {name} của '{names[i]}' và '{names[j]}' giống hệt nhau** "
                                     "ở các mức tải trùng nhau. Hai kịch bản khác nhau thường cho số khác nhau.")
    return notes


def load_csv(raw: bytes, filename: str) -> dict:
    df, sep = read_table(raw)
    df = df.rename(columns={"Thuật toán": "algo", "Tải chào (Erlang)": "x", "Công suất phát (dBm)": "x"})
    kind = detect_kind(df)
    pts, log = normalize(df, kind, filename)
    return dict(name=filename, kind=kind, sep=sep, raw=df, pts=pts, log=log)
