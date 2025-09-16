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


if __name__ == "__main__":
    max_workers = 20
    strategies = ["ucb", "osub", "sw_osub", "sw_linucb", "epsilon_greedy"]
    strategy_rl_mode_map = {
        "ucb": [0, 1],
        "osub": [1],
        "sw_osub": [1],
        "sw_linucb": [0, 1],
        "epsilon_greedy": [0, 1],
    }
    scenario = "3"  # "A", "B", "1", "2", or "3"
    seeds = range(1, 21)

    enable_channel_options = True
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

    for strategy in strategies:
        for rl_mode in strategy_rl_mode_map[strategy]:
            tasks = []
            for seed in seeds:
                run_name = f"{strategy}_rl{rl_mode}_seed{seed}"
                env_vars = {
                    "STRATEGY": strategy,
                    "RL_MODE": str(rl_mode),
                    "SEED": str(seed),
                    "SCENARIO": scenario,
                    "WANDB_RUN_NAME": run_name,
                }
                tasks.append((env_vars, script_path, python_executable, run_name))

            print(f"\nRunning {len(tasks)} seeds for {strategy} RL{rl_mode}...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        run_trial, env_vars, script_path, python_executable, run_name
                    )
                    for env_vars, script_path, python_executable, run_name in tasks
                ]
                for future in as_completed(futures):
                    future.result()  # wait for all seeds to finish

            print(f"Completed all seeds for {strategy} RL{rl_mode}")

    if enable_channel_options:
        # Override channel options for scenarios 1,2,3
        scenario_channels = (
            [[1, 2, 3, 4]] if scenario in ["1", "2", "3"] else channel_options
        )
        for baseline_channels in scenario_channels:
            tasks = []
            channels_str = "-".join(map(str, baseline_channels))
            for seed in seeds:
                run_name = f"channels{channels_str}_seed{seed}"
                env_vars = {
                    "SEED": str(seed),
                    "SCENARIO": scenario,
                    "BASELINE_CHANNELS": ",".join(map(str, baseline_channels)),
                    "WANDB_RUN_NAME": run_name,
                }
                tasks.append((env_vars, script_path, python_executable, run_name))

            print(
                f"\nRunning {len(tasks)} seeds for channels {channels_str} scenario {scenario}..."
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        run_trial, env_vars, script_path, python_executable, run_name
                    )
                    for env_vars, script_path, python_executable, run_name in tasks
                ]
                for future in as_completed(futures):
                    future.result()

            print(
                f"Completed all seeds for channels {channels_str} scenario {scenario}"
            )
