"""Run one episode using the same factory and agents as the Streamlit demo."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from team.agents.first_fit import FirstFitAgent
from team.agents.rmsa_agents import make_agent
from team.config import SCENARIOS, TOPOLOGIES
from team.env_factory import create_env
from team.metadata import json_value, runtime_metadata
from team.runner import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['first_fit', 'ksp_ff', 'ls_bm_ksp'], default='first_fit')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--topology', choices=list(TOPOLOGIES), default='nobel-eu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', type=float, default=300.0)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--power', type=float, default=None, help='Launch power in dBm; omitted uses ONG scenario default')
    args = parser.parse_args()
    if args.requests < 1 or args.load <= 0:
        parser.error('requests và load phải dương')
    scenario_name = list(SCENARIOS)[args.scenario - 1]
    sc = SCENARIOS[scenario_name]
    config = dict(seed=args.seed, load=args.load, episode_length=args.requests,
                  topology=args.topology, num_slots=sc['num_slots'], bit_rates=sc['bit_rates'],
                  k_paths=3, modulations_to_consider=4, launch_power_dbm=args.power)
    agent = FirstFitAgent() if args.algorithm == 'first_fit' else make_agent(args.algorithm)
    env = create_env(config)
    try:
        effective = json_value(env.simulator.config)
        metrics = run_episode(env, agent, seed=args.seed)
    finally:
        env.close()
    result = dict(algorithm=agent.name, scenario=scenario_name, config=config,
                  effective_config=effective, metrics=metrics, provenance=runtime_metadata())
    folder = Path(__file__).resolve().parent / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = folder / f'{agent.name}_{stamp}.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Đã lưu: {output}')


if __name__ == '__main__':
    main()
