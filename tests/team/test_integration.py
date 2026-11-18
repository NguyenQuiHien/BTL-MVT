import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from team.agents.rmsa_agents import make_agent
from team.env_factory import create_env
from team.metadata import json_value
from demo.sim_core import make_env_for, step_request, run_sweep, summarize_runs, aggregate_runs, SCENARIOS
from demo.csv_loader import load_csv, aggregate_points


@pytest.mark.parametrize('algorithm', ['KSP-FF', 'LS-BM-KSP'])
def test_demo_matches_shared_agent_and_snapshot(algorithm):
    env = make_env_for('ring_4', 10, 4, 80, (100,), 42)
    reference = create_env(dict(topology='ring_4', load=10, episode_length=4,
                                num_slots=80, bit_rates=(100,), seed=42))
    try:
        _, info = env.reset(seed=42)
        obs, ri = reference.reset(seed=42)
        captured = []
        original = env.simulator.post_action_callback

        def observer(transition):
            captured.append((env.simulator.state.slot_allocation >= 0).copy())
            original(transition)

        env.simulator.post_action_callback = observer
        agent = make_agent(algorithm)
        for _ in range(4):
            rec, info, done, aux = step_request(env, algorithm, info)
            action = agent.solution(obs, ri, reference)
            obs, _, terminated, truncated, ri = reference.step(action)
            assert rec['accepted'] == ri['accepted']
            assert rec['sbp_cum'] == ri['episode_service_blocking_rate']
            np.testing.assert_array_equal(aux['occupied'], captured[-1])
            assert rec['util'] == captured[-1].mean()
            assert done == (terminated or truncated)
            assert env.simulator.post_action_callback is observer
        assert done and len(captured) == 4
        json.dumps(json_value(env.simulator.config), allow_nan=False)
    finally:
        env.close()
        reference.close()


def test_power_sweep_and_metadata():
    log = run_sweep(['KSP-FF', 'LS-BM-KSP'], 'ring_4', list(SCENARIOS)[1],
                    'launch_power_dbm', [-8, 8], 100, None, 3, [42])
    assert len(log) == 12
    assert set(log.launch_power_dbm) == {-8, 8}
    assert (log.load == 100).all()
    assert set(log.util_sample) == {'post_action_before_advance'}
    configs = log.attrs['metadata']['effective_configs']
    assert {v['config']['launch_power_dbm'] for v in configs} == {-8, 8}
    json.dumps(log.attrs['metadata'], allow_nan=False)
    runs = summarize_runs(log)
    assert len(runs) == 4
    assert runs.sbp.between(0, 1).all()
    assert runs.bbr.between(0, 1).all()


def test_original_member4_summaries_match_logs():
    root = Path(__file__).resolve().parents[2] / 'data/member4'
    for k in (1, 2, 3):
        folder = root / f'kb{k}'
        log = pd.read_csv(folder / f'kb{k}_log_tung_yeu_cau.csv')
        actual = aggregate_runs(summarize_runs(log))
        file = folder / f'kb{k}_ketqua.csv'
        loaded = load_csv(file.read_bytes(), file.name)
        assert loaded['kind'] == 'runs'
        expected = aggregate_points(loaded['pts'])
        merged = actual.merge(expected, on=['algo', 'x'], suffixes=('_calc', '_file'))
        assert len(merged) == 12
        for col in ['sbp', 'bbr', 'avg_ms', 'util_pct', 'sbp_std', 'bbr_std']:
            np.testing.assert_allclose(merged[col + '_calc'].round(4), merged[col + '_file'], atol=1e-8)
