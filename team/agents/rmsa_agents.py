"""Team adapters for the ONG heuristics, shared with the demo."""
import numpy as np
from optical_networking_gym import select_heuristic_action


class RMSAAgent:
    def __init__(self, name, policy):
        self.name, self.policy = name, policy

    def solution(self, obs, info, env):
        # These policies use the environment context/mask, not obs alone.
        action = int(select_heuristic_action(self.policy, env, info))
        if not env.action_space.contains(action):
            raise ValueError(f"Action ngoài action_space: {action}")
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()
        if mask is None or np.asarray(mask).shape != (env.action_space.n,):
            raise ValueError("Action mask không tồn tại hoặc sai kích thước")
        if not bool(mask[action]):
            raise ValueError(f"Action bị mask loại: {action}")
        return action


class KSPFFAgent(RMSAAgent):
    def __init__(self):
        super().__init__("ksp_ff", "jocn_ksp_ff_bm")


class LSBMKSPAgent(RMSAAgent):
    def __init__(self):
        super().__init__("ls_bm_ksp", "jocn_ls_bm_ksp")


def make_agent(name):
    if name in ("ksp_ff", "KSP-FF", "jocn_ksp_ff_bm"):
        return KSPFFAgent()
    if name in ("ls_bm_ksp", "LS-BM-KSP", "jocn_ls_bm_ksp"):
        return LSBMKSPAgent()
    raise ValueError(f"Thuật toán không được hỗ trợ: {name}")
