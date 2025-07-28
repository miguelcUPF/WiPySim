import subprocess
import sys
import os

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_trial(env_vars, script_path, python_executable, run_name):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    env = os.environ.copy()
    env["WANDB_START_METHOD"] = "thread"
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_vars)

    try:
        print(f"[START] {run_name}")
        subprocess.run(
            [python_executable, script_path],
            check=True,
            env=env,
            capture_output=False,
            text=True,
        )
        print(f"[OK] {run_name}")
    except Exception as e:
        print(f"[FAILED] {run_name}")
        print("=== STDOUT ===")
        print(e.stdout)
        print("=== STDERR ===")
        print(e.stderr)


def generate_tasks():
    strategies = ["ucb", "osub", "sw_osub", "sw_linucb", "epsilon_greedy"]
    strategy_rl_mode_map = {
        "ucb": [0, 1],
        "osub": [1],
        "sw_osub": [1],
        "sw_linucb": [0, 1],
        "epsilon_greedy": [0, 1],
    }
    scenarios = ["A", "B", "C"]
    rl_driven_values = [0, 1]  # False, True
    seeds = range(1, 21)

    enable_channel_options = False
    channel_options = [
        [1],
        [2],
        [3],
        [4],
        [1, 2],
        [3, 4],
        [1, 2, 3, 4],
    ]

    script_path = "tests/journal/trial_runner.py"
    python_executable = sys.executable

    tasks = []

    for scenario in scenarios:
        for all_rl_driven in rl_driven_values:
            for strategy in strategies:
                for rl_mode in strategy_rl_mode_map[strategy]:
                    for seed in seeds:
                        run_name = f"{scenario}_all{all_rl_driven}_{strategy}_rl{rl_mode}_seed{seed}"
                        env_vars = {
                            "STRATEGY": strategy,
                            "RL_MODE": str(rl_mode),
                            "SEED": str(seed),
                            "SCENARIO": scenario,
                            "ALL_RL_DRIVEN": str(all_rl_driven),
                            "WANDB_RUN_NAME": run_name,
                        }

                        tasks.append(
                            (env_vars, script_path, python_executable, run_name)
                        )

    if enable_channel_options:
        for scenario in scenarios:
            for ap1_channels in channel_options:
                channels_str = "-".join(map(str, ap1_channels))
                for seed in seeds:
                    run_name = f"{scenario}_channels{channels_str}_seed{seed}"
                    env_vars = {
                        "SEED": str(seed),
                        "SCENARIO": scenario,
                        "AP1_CHANNELS": ",".join(map(str, ap1_channels)),
                        "WANDB_RUN_NAME": run_name,
                    }

                    tasks.append((env_vars, script_path, python_executable, run_name))

    return tasks


if __name__ == "__main__":
    max_workers = 20
    tasks = generate_tasks()

    print(f"Launching {len(tasks)} parallel runs with up to {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_trial, env_vars, script_path, python_executable, run_name
            )
            for env_vars, script_path, python_executable, run_name in tasks
        ]

        for future in as_completed(futures):
            future.result()
