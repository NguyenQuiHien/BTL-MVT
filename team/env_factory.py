from optical_networking_gym import make_env
from team.config import MODULATIONS, TOPOLOGIES


def create_env(config):
    topology = config.get("topology", "nobel-eu")
    extra = {}
    if config.get("launch_power_dbm") is not None:
        extra["launch_power_dbm"] = float(config["launch_power_dbm"])
    return make_env(
        scenario=TOPOLOGIES[topology],
        seed=int(config["seed"]), load=float(config["load"]),
        episode_length=int(config["episode_length"]),
        num_spectrum_resources=int(config.get("num_slots", 320)),
        k_paths=int(config.get("k_paths", 3)),
        modulations_to_consider=int(config.get("modulations_to_consider", 4)),
        bit_rates=tuple(config.get("bit_rates", (10, 40, 100, 400))),
        modulations=MODULATIONS,
        overrides={"topology_id": topology}, **extra,
    )
