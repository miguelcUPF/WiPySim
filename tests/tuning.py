from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    RESULTS_MSG,
)


from functools import partial

import optuna
import optuna.visualization as vis
import random
from optuna.importance import get_param_importances


def style_plotly_as_matplotlib(fig, small_fonts=False):
    fig.update_layout(
        font=dict(family="DejaVu Serif", size=10 if small_fonts else 16, color="black"),
        title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        showgrid=False,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        showgrid=False,
    )
    return fig


def generate_training_scenarios(
    sim_time_choices: list, num_bss_choices: list, seed=None
):
    from src.utils.scenario_gen import generate_random_scenario

    rng = random.Random(seed)

    training_scenarios = []

    # Create all combinations
    scenario_pool = [(t, b) for t in sim_time_choices for b in num_bss_choices]
    rng.shuffle(scenario_pool)

    for i, (sim_time_us, num_bss) in enumerate(scenario_pool):
        training_scenarios.append(
            generate_random_scenario(
                seed=i,
                num_bss=num_bss,
                num_rl_bss=1,
                sim_time_us=sim_time_us,
                disable_start_end=True,
            )
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

    agents_settings = {"strategy": strategy}

    if strategy in ["sw_linucb", "linucb"]:
        agents_settings["alpha"] = trial.suggest_float("alpha", 0.1, 2.0)
        if strategy == "sw_linucb":
            agents_settings["window_size"] = trial.suggest_int("window_size", 0, 84)

            if agents_settings["window_size"] == -1:
                agents_settings["window_size"] = None

    elif strategy in ["epsilon_greedy", "decay_epsilon_greedy"]:
        agents_settings["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.3)
        agents_settings["eta"] = trial.suggest_float("eta", 1e-4, 1e-1, log=True)
        agents_settings["gamma"] = trial.suggest_float("gamma", 0.7, 0.99)
        agents_settings["alpha_ema"] = trial.suggest_float("alpha_ema", 0.01, 0.3)
        if strategy == "decay_epsilon_greedy":
            agents_settings["decay_rate"] = trial.suggest_float("decay_rate", 0.8, 1.0)

        agents_settings["channel_frequency"] = 1
        agents_settings["primary_frequency"] = 1
        agents_settings["cw_frequency"] = 1

    for step, scenario in enumerate(training_scenarios):
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

        cfg.AGENTS_SETTINGS = agents_settings

        try:
            env = simpy.Environment()
            network = initialize_network(cfg, sparams, env)
            env.run(until=cfg.SIMULATION_TIME_us)

            for ap in network.get_aps():
                if ap.mac_layer.rl_driven is not None:
                    results = ap.mac_layer.rl_controller.results
                    if len(results) == 0:
                        raise ValueError("No results")
                    result = np.mean(results)
                    break

            runs.append(result)

            trial.report(result, step)

            if trial.should_prune():
                print(f"Trial {trial.number} step {step} pruned: {result}")
                raise optuna.TrialPruned()

            print(f"Trial {trial.number} step {step} completed: {result}")
        except Exception as e:
            print(f"Trial {trial.number} step {step} failed: {e}")

    return np.mean(runs)


N_TRIALS = 100
SEED = 1  # or None
RL_MODE = 1
STRATEGY = "sw_linucb"
DISPLAY_STUDY_FIGS = True

SIM_TIME_CHOICES = [1_000_000, 2_000_000, 4_000_000, 8_000_000]
NUM_BSS_CHOICES = [2, 3, 4]

if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    training_scenarios = generate_training_scenarios(
        SIM_TIME_CHOICES, NUM_BSS_CHOICES, seed=SEED
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{RL_MODE}_{STRATEGY}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )  # minimize delay

    study.optimize(
        partial(
            objective,
            training_scenarios=training_scenarios,
            strategy=STRATEGY,
            rl_mode=RL_MODE,
        ),
        n_trials=N_TRIALS,
        n_jobs=-1,
        show_progress_bar=True,
    )
    print(TEST_COMPLETED_MSG)

    print(RESULTS_MSG)
    print(f"Best trial value (reward): {study.best_trial.value:.4f}")
    print(f"Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nParameter Importances:")
    param_importances = get_param_importances(study)

    for param, importance in param_importances.items():
        print(f"{param}: {importance}")

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
        fig1 = style_plotly_as_matplotlib(vis.plot_optimization_history(study))
        fig1.show()

        if len(study.trials) > 1:
            fig2 = style_plotly_as_matplotlib(vis.plot_param_importances(study))
            fig2.show()

        fig3 = style_plotly_as_matplotlib(
            vis.plot_parallel_coordinate(study), small_fonts=True
        )
        fig3.show()

        fig4 = style_plotly_as_matplotlib(vis.plot_edf(study))
        fig4.show()
