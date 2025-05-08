from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    PRESS_TO_EXIT_MSG,
    RESULTS_MSG,
)


from functools import partial

import optuna
import optuna.visualization as vis
import random
import os


def generate_training_scenarios(num_scenarios: int = 25, sim_time_us_list=None, seed=None):
    from src.utils.scenario_gen import generate_random_scenario

    training_scenarios = []

    rng = random.Random(seed)

    for i in range(num_scenarios):
        if sim_time_us_list is not None:
            if i >= len(sim_time_us_list):
                raise ValueError("sim_time_us_list must be as long as num_scenarios")
            sim_time_us = sim_time_us_list[i]
        else:
            sim_time_us = None
        training_scenarios.append(
            generate_random_scenario(seed=i, num_bss=rng.randint(2, 6), num_rl_bss=1, sim_time_us=sim_time_us, disable_start_end=True)
        )

    return training_scenarios


def objective(
    trial: optuna.Trial, training_scenarios: dict, strategy: str, rl_mode: int
):
    from tests._user_config_tests import UserConfig as cfg_module
    from tests._sim_params_tests import SimParams as sparams_module
    from src.utils.support import initialize_network

    import simpy
    import numpy as np

    runs = []

    for step, scenario in enumerate(training_scenarios):
        cfg = cfg_module()
        sparams = sparams_module()

        cfg.SIMULATION_TIME_us = scenario["sim_time_us"]

        cfg.SEED = scenario["seed"]

        cfg.ENABLE_RL = True
        cfg.RL_MODE = rl_mode

        cfg.USE_WANDB = False
        cfg.ENABLE_CONSOLE_LOGGING = False
        cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
        cfg.ENABLE_REWARD_DECOMPOSITION = False

        cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True
        cfg.BSSs_Advanced = scenario["bsss_advanced"]

        cfg.ENABLE_STATS_COMPUTATION = False

        agents_settings = {"strategy": strategy}

        if strategy in ["sw_linucb", "linucb"]:
            agents_settings["alpha"] = trial.suggest_float("alpha", 0.1, 2.0)
            if strategy == "sw_linucb":
                agents_settings["window_size"] = trial.suggest_int("window_size", 0, 20)

        elif strategy in ["epsilon_greedy", "decay_epsilon_greedy"]:
            agents_settings["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.5)
            agents_settings["eta"] = trial.suggest_float("eta", 1e-5, 1e-1, log=True)
            agents_settings["gamma"] = trial.suggest_float("gamma", 0.7, 0.99)
            agents_settings["alpha_ema"] = trial.suggest_float("alpha_ema", 0.01, 0.3)
            if strategy == "decay_epsilon_greedy":
                agents_settings["decay_rate"] = trial.suggest_float(
                    "decay_rate", 0.8, 1.0
                )

        cfg.AGENTS_SETTINGS = agents_settings

        try:
            env = simpy.Environment()
            network = initialize_network(cfg, sparams, env)
            env.run(until=cfg.SIMULATION_TIME_us)

            for ap in network.get_aps():
                if ap.id == 1:
                    results = ap.mac_layer.rl_controller.results
                    if len(results) == 0:
                        result = np.inf
                        break
                    result = np.mean(results)
                    break

            runs.append(result)

            trial.report(result, step)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")

    return np.mean(runs)


N_TRIALS = 1
N_SCENARIOS = 1
SEED = 1
RL_MODE = 1
STRATEGY = "sw_linucb"
CLEANUP_STUDY = True
DISPLAY_STUDY_FIGS = False

if CLEANUP_STUDY:
    if os.path.exists("tuning_study.db"):
        os.remove("tuning_study.db")

if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    training_scenarios = generate_training_scenarios(
        num_scenarios=N_SCENARIOS, seed=SEED
    )

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///tuning_study.db",
        study_name=f"{RL_MODE}_{STRATEGY}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )  # minimize delay

    study.optimize(
        partial(
            objective,
            training_scenarios=training_scenarios,
            strategy=STRATEGY,
            rl_mode=RL_MODE,
        ),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )
    print(TEST_COMPLETED_MSG)

    print(RESULTS_MSG)
    print(f"Best trial value (reward): {study.best_trial.value:.4f}")
    print(f"Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    if len(study.trials) >= 3:
        print("\nTop 3 Trials:")
        top_trials = sorted(
            [t for t in study.trials if t.value is not None], key=lambda t: t.value
        )[:3]
        for t in top_trials:
            print(f"  Trial {t.number} - Value: {t.value:.4f}, Params: {t.params}")

    print("\nStudy Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(
        f"  Completed: {sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)}"
    )
    print(
        f"  Pruned: {sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}"
    )
    print(
        f"  Failed: {sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)}"
    )

    if DISPLAY_STUDY_FIGS:
        vis.plot_optimization_history(study).show()
        if len(study.trials) > 1:
            vis.plot_param_importances(study).show()
        vis.plot_parallel_coordinate(study).show()
        vis.plot_edf(study)

        input(PRESS_TO_EXIT_MSG)
    
