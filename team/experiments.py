"""Export load/power sweeps without opening Streamlit (install the demo extra)."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from team.config import SCENARIOS, TOPOLOGIES
from demo.sim_core import run_sweep, summarize_runs, aggregate_runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--topology', choices=list(TOPOLOGIES), default='nobel-eu')
    p.add_argument('--axis', choices=['load', 'power'], default='load')
    p.add_argument('--values', nargs='+', type=float, default=None)
    p.add_argument('--load', type=float, default=300)
    p.add_argument('--power', type=float, default=None)
    p.add_argument('--requests', type=int, default=1000)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--algorithms', nargs='+', choices=['KSP-FF', 'LS-BM-KSP'], default=['KSP-FF', 'LS-BM-KSP'])
    p.add_argument('--warmup', type=float, default=0.0, help='Fraction of initial requests omitted, 0 <= value < 1')
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args()
    if not 0 <= a.warmup < 1 or a.requests < 1 or a.load <= 0:
        p.error('warmup phải trong [0,1), requests và load phải dương')
    values = a.values or ([50, 100, 150, 200, 250, 300] if a.axis == 'load' else [-8, -6, -4, -2, 0, 2, 4, 6, 8])
    if a.axis == 'load' and any(v <= 0 for v in values):
        p.error('Các mức tải phải dương')
    if len(set(values)) != len(values) or len(set(a.seeds)) != len(a.seeds) or len(set(a.algorithms)) != len(a.algorithms):
        p.error('Không lặp giá trị, seed hoặc thuật toán trong cùng một sweep')
    scenario = list(SCENARIOS)[a.scenario - 1]
    log = run_sweep(a.algorithms, a.topology, scenario,
                    'load' if a.axis == 'load' else 'launch_power_dbm', values,
                    a.load, a.power, a.requests, a.seeds,
                    progress=lambda _, message: print(message, flush=True))
    runs = summarize_runs(log, a.warmup)
    summary = aggregate_runs(runs)
    # A unique run directory prevents accidental overwrite of earlier results.
    base = a.output or Path(__file__).resolve().parent / 'results' / 'experiments'
    folder = base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    log.to_csv(folder / 'requests.csv', index=False, encoding='utf-8-sig')
    runs.to_csv(folder / 'runs.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(folder / 'summary.csv', index=False, encoding='utf-8-sig')
    metadata = dict(log.attrs['metadata'], scenario=scenario, axis=a.axis, values=values,
                    seeds=a.seeds, requests=a.requests, warmup_frac=a.warmup)
    (folder / 'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Đã lưu: {folder}')


if __name__ == '__main__':
    main()
