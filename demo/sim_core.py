from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from team.env_factory import create_env
from team.agents.rmsa_agents import make_agent
from team.config import MODULATIONS, TOPOLOGIES, SCENARIOS
from team.metadata import json_value, runtime_metadata
from optical_networking_gym.defaults import MODULATION_CATALOG, resolve_topology

MOD_ORDER = ["BPSK", "QPSK", "8QAM", "16QAM", "32QAM", "64QAM"]


ALGOS = {"KSP-FF": "jocn_ksp_ff_bm", "LS-BM-KSP": "jocn_ls_bm_ksp"}
COLORS = {"KSP-FF": "#d1495b", "LS-BM-KSP": "#2e86ab"}

def make_env_for(topology, load, n_requests, num_slots, bit_rates, seed,
                 k_paths=3, launch_power_dbm=None):
    return create_env(dict(topology=topology, load=load, episode_length=n_requests,
                           num_slots=num_slots, bit_rates=bit_rates, seed=seed,
                           k_paths=k_paths, launch_power_dbm=launch_power_dbm))


def modulation_table():
    """{tên: (ngưỡng OSNR/GSNR tối thiểu dB, hiệu suất phổ)} lấy từ thư viện."""
    return {n: (float(MODULATION_CATALOG[n].minimum_osnr),
                int(MODULATION_CATALOG[n].spectral_efficiency)) for n in MOD_ORDER}


# --------------------------------------------------------------------------- topology
@dataclass
class TopologyView:
    name: str
    names: list
    pos: np.ndarray            
    links: list                
    has_geo: bool
    source_file: str


def _read_geo_coords(topology, names):
    path = resolve_topology(topology)
    if path.suffix.lower() != ".xml":
        return None, path.name
    root = ET.parse(path).getroot()
    coords = {}
    for node in root.iter():
        if node.tag.split("}")[-1] != "node" or node.get("id") is None:
            continue
        x = y = None
        for ch in node.iter():
            tag = ch.tag.split("}")[-1]
            if tag == "x":
                x = float(ch.text)
            elif tag == "y":
                y = float(ch.text)
        if x is not None and y is not None:
            coords[node.get("id")] = (x, y)
    if not all(n in coords for n in names):
        return None, path.name
    arr = np.array([coords[n] for n in names], dtype=float)
    arr[:, 0] *= np.cos(np.radians(arr[:, 1].mean()))   
    return arr, path.name


def load_topology_view(topology) -> TopologyView:
    env = make_env_for(topology, 100, 1, 320, (10, 40, 100, 400), 0)
    try:
        topo = env.simulator.topology
        names = list(topo.node_names)
        links = [(int(l.id), int(l.source_index), int(l.target_index), float(l.length_km))
                 for l in topo.links]
    finally:
        env.close()
    pos, fname = _read_geo_coords(topology, names)
    has_geo = pos is not None
    if pos is None:
        ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
        pos = np.column_stack([np.cos(ang), np.sin(ang)])
    return TopologyView(topology, names, pos, links, has_geo, fname)


def draw_network(ax, view, link_util=None, path_link_ids=(), accepted=True,
                 endpoints=None, label_nodes=False, title=""):
    """Vẽ topology; màu link = mức sử dụng phổ; đường được chọn tô xanh, bị chặn đánh dấu đỏ."""
    import matplotlib.pyplot as plt
    idx = {lid: i for i, (lid, *_rest) in enumerate(view.links)}
    segs = [[view.pos[s], view.pos[t]] for _, s, t, _ in view.links]
    if link_util is None:
        colors = "#b8b8b8"
    else:
        colors = plt.cm.YlOrRd(np.clip(np.asarray(link_util, dtype=float), 0, 1) * 0.9)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=2.2, zorder=1))
    if accepted and path_link_ids:
        ps = [segs[idx[l]] for l in path_link_ids if l in idx]
        ax.add_collection(LineCollection(ps, colors="#1a9850", linewidths=5, zorder=2))
    ax.scatter(view.pos[:, 0], view.pos[:, 1], s=22, c="#1f4e79", zorder=3)
    if label_nodes:
        for n, (x, y) in zip(view.names, view.pos):
            ax.annotate(n, (x, y), fontsize=6, xytext=(2, 3), textcoords="offset points")
    if endpoints is not None:
        a, b = endpoints
        col = "#1a9850" if accepted else "#d7191c"
        for i in (a, b):
            ax.scatter(*view.pos[i], s=110, facecolors="none", edgecolors=col, linewidths=2.5, zorder=4)
            ax.annotate(view.names[i], view.pos[i], fontsize=8, fontweight="bold",
                        xytext=(4, 6), textcoords="offset points", zorder=5)
        if not accepted:
            ax.plot(*zip(view.pos[a], view.pos[b]), "--", color=col, lw=2, zorder=2)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


def draw_spectrum(ax, occupied, title=""):
    ax.imshow(np.asarray(occupied, dtype=float), aspect="auto", cmap="Greys",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Link")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


# --------------------------------------------------------------------------- simulation
def step_request(env, policy, info):
    """Xử lý 1 yêu cầu. Trả về (record, info mới, kết thúc?, aux cho hiển thị)."""
    sim = env.simulator
    req = sim.current_request
    paths = sim.current_analysis.paths         
    names = sim.topology.node_names
    mods = sim.config.modulations
    src, dst, bit_rate = int(req.source_id), int(req.destination_id), int(req.bit_rate)

    snapshot = {}
    previous_callback = sim.post_action_callback

    def capture(transition):
        # Capture before advancing time and releasing expired connections.
        snapshot["occupied"] = (np.asarray(sim.state.slot_allocation) >= 0).copy()
        if previous_callback is not None:
            previous_callback(transition)

    sim.post_action_callback = capture
    try:
        agent = make_agent(policy)
        t0 = time.perf_counter()
        action = agent.solution(env.observation(sim.current_observation), info, env)
        _, _, terminated, truncated, info = env.step(action)
        ms = (time.perf_counter() - t0) * 1000.0
    finally:
        sim.post_action_callback = previous_callback

    accepted = bool(info["accepted"])
    pi, mi = info.get("chosen_path_index"), info.get("chosen_modulation_index")
    path = paths[pi] if (accepted and pi is not None) else None
    occupied = snapshot["occupied"]

    rec = dict(
        src=names[src], dst=names[dst], bit_rate=bit_rate, accepted=accepted,
        status=str(info.get("status")),
        modulation=mods[mi].name if (accepted and mi is not None) else None,
        gsnr_db=float(info["osnr"]) if accepted else np.nan,
        gsnr_req_db=float(info["osnr_req"]) if accepted else np.nan,
        path_km=float(path.length_km) if path is not None else np.nan,
        hops=int(path.hops) if path is not None else None,
        slot=info.get("chosen_slot"),
        ms=ms, util=float(occupied.mean()),
        sbp_cum=float(info["episode_service_blocking_rate"]),
        bbr_cum=float(info["episode_bit_rate_blocking_rate"]),
    )
    aux = dict(path_link_ids=tuple(path.link_ids) if path is not None else (),
               endpoints=(src, dst), link_util=occupied.mean(axis=1), occupied=occupied)
    return rec, info, bool(terminated or truncated), aux


def run_sweep(algos, topology, scenario, x_name, x_values, fixed_load, fixed_power,
              n_requests, seeds, progress=None):
    """Chạy lưới (thuật toán × giá trị x × seed). Trả về DataFrame log từng request."""
    sc = SCENARIOS[scenario]
    runs = [(a, x, s) for a in algos for x in x_values for s in seeds]
    rows = []
    effective_configs = []
    for k, (algo, x, seed) in enumerate(runs):
        load = x if x_name == "load" else fixed_load
        power = x if x_name == "launch_power_dbm" else fixed_power
        env = make_env_for(topology, load, n_requests, sc["num_slots"], sc["bit_rates"],
                           seed, launch_power_dbm=power)
        try:
            effective_configs.append(dict(algo=algo, x=x, seed=int(seed), config=json_value(env.simulator.config)))
            _, info = env.reset(seed=int(seed))
            i = 0
            while True:
                rec, info, done, _ = step_request(env, ALGOS[algo], info)
                rec.update(algo=algo, x=x, seed=int(seed), idx=i)
                rec.update(topology=topology, scenario=scenario, x_name=x_name,
                           load=float(load), launch_power_dbm=float(env.simulator.config.launch_power_dbm),
                           num_slots=sc["num_slots"], util_sample="post_action_before_advance")
                rows.append(rec)
                i += 1
                if done:
                    break
        finally:
            env.close()
        if progress:
            progress((k + 1) / len(runs), f"Đã chạy {k + 1}/{len(runs)}: {algo}, {x_name}={x}, seed={seed}")
    result = pd.DataFrame(rows)
    result.attrs["metadata"] = dict(runtime=runtime_metadata(), effective_configs=effective_configs,
        timing="agent selection + env.step including spectrum snapshot",
        util_sample="post_action_before_advance")
    return result


def summarize_runs(log, warmup_frac=0.0):
    """Mỗi (thuật toán, x, seed) -> SBP, BBR, thời gian/yêu cầu, mức sử dụng phổ."""
    out = []
    for (algo, x, seed), g in log.groupby(["algo", "x", "seed"], sort=False):
        g = g[g["idx"] >= int(len(g) * warmup_frac)]
        blocked = ~g["accepted"]
        out.append(dict(
            algo=algo, x=x, seed=seed, requests=len(g),
            sbp=float(blocked.mean()),
            bbr=float(g.loc[blocked, "bit_rate"].sum() / g["bit_rate"].sum()),
            avg_ms=float(g["ms"].mean()),
            util_pct=100.0 * float(g["util"].mean()),
        ))
    return pd.DataFrame(out)


def aggregate_runs(runs):
    agg = runs.groupby(["algo", "x"]).agg(
        sbp=("sbp", "mean"), sbp_std=("sbp", "std"),
        bbr=("bbr", "mean"), bbr_std=("bbr", "std"),
        avg_ms=("avg_ms", "mean"), avg_ms_std=("avg_ms", "std"),
        util_pct=("util_pct", "mean"), util_pct_std=("util_pct", "std"),
        seeds=("seed", "nunique"),
    ).reset_index()
    return agg.sort_values(["algo", "x"])
