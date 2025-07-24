import sys
import json
import numpy as np
import simpy
import gc

from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module
from src.utils.support import initialize_network


def run_single_step(trial_number, params, strategy, rl_mode, scenario, step):
    cfg = cfg_module()
    sparams = sparams_module()

    cfg.SIMULATION_TIME_us = scenario["sim_time_us"]
    cfg.SEED = scenario["seed"]

    cfg.ENABLE_RL = True
    cfg.RL_MODE = rl_mode
    cfg.FIRST_AS_PRIMARY = True

    cfg.USE_WANDB = False
    cfg.ENABLE_CONSOLE_LOGGING = False
    cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
    cfg.ENABLE_REWARD_DECOMPOSITION = False

    cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True
    cfg.BSSs_Advanced = scenario["bsss_advanced"]

    cfg.ENABLE_STATS_COMPUTATION = False

    cfg.AGENTS_SETTINGS = params

    try:
        env = simpy.Environment()
        network = initialize_network(cfg, sparams, env)
        env.run(until=cfg.SIMULATION_TIME_us)
        for ap in network.get_aps():
            if ap.mac_layer.rl_driven is not None:
                results = ap.mac_layer.rl_controller.results
                if not results:
                    raise ValueError("No results")
                result = np.mean(results)
                print(
                    f"Trial {trial_number} step {step} completed: {result} [{strategy}]",
                    file=sys.stderr,
                )  # sys.stderr
                return result

    except Exception as e:
        print(f"{e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    trial_number = int(sys.argv[1])
    params = json.loads(sys.argv[2])
    strategy = sys.argv[3]
    rl_mode = int(sys.argv[4])
    scenario = json.loads(sys.argv[5])
    step = int(sys.argv[6])

    try:
        result = run_single_step(
            trial_number, params, strategy, rl_mode, scenario, step
        )
        print(json.dumps({"result": result}))
    except Exception:
        sys.exit(1)
