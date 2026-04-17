# WiPySim: An IEEE 802.11 DCF Wireless Network Simulator with CMAB-Based Optimization

This Python event-driven simulator, built using SimPy, models **IEEE 802.11 Distributed Coordination Function (DCF) wireless networks** with support for **multi-armed bandit (MAB)** and **contextual multi-armed bandit (CMAB)** algorithms to optimize parameters such as **channel group allocation**, **primary channel selection**, and the **Contention Window (CW) size**.

The simulator supports both **single-agent** and **multi-agent** architectures, and implements several bandit-based learning strategies, including:

- **Upper Confidence Bound (UCB)** ([Auer et al., 2002](https://doi.org/10.1023/A:1013689704352))
- **Linear UCB (LinUCB)** ([Li et al., 2010](https://dl.acm.org/doi/abs/10.1145/1772690.1772758))
- **Sliding-Window LinUCB (SW-LinUCB)** ([Cheung et al., 2019](https://dl.acm.org/doi/10.1145/3297280.3297440))
- **Optimal Sampling for Unimodal Bandits (OSUB)** ([Combes & Proutière, 2014](https://proceedings.mlr.press/v32/combes14.html))
- **Explore-Explore-Then-Commit (E2TC)** ([Zhang et al., 2025](https://doi.org/10.48550/arXiv.2502.17175))
- **Epsilon-RMSProp Linear Bandit (E-RLB)** algorithm ([Casasnovas et al., 2025](https://doi.org/10.48550/arXiv.2511.23352))

 
It has been used in the following research works:
- [***"Learning-Based Channel Access in Wi-Fi: A Multi-Armed Bandit Approach"***](https://doi.org/10.48550/arXiv.2511.10143)
- [***"Performance Evaluation of Multi-Armed Bandit Algorithms for Wi-Fi Channel Access"***](https://doi.org/10.48550/arXiv.2511.23352)


and has been developed as part of the ongoing research of [**Miguel Casasnovas Bielsa**](https://scholar.google.com/citations?user=jRxBfaMAAAAJ&hl=ca) within the [**Wireless Networking Research Group**](https://www.upf.edu/web/wnrg) at **Universitat Pompeu Fabra (UPF)**, and as part of his Master’s Thesis in Data Science at **Universitat Oberta de Catalunya (UOC)**.

## Features
- Detailed IEEE 802.11 DCF simulation
- Support for single and multi-agent CMAB architectures
- Configurable wireless and traffic parameters
- Event logging and statistics collection
- Traffic trace recording and loading (CSV format with fields such as `"frame.time_relative"` and `"frame.len"`, compatible with both simulator-generated traces and external tools like Wireshark)
- Visualization and figure generation

   and much more...

## Installation

Python 3 is required.

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulator from the root project directory:

```bash
python -m src.main
```
or, alternatively, run toy tests:
```bash
python -m tests.<test_name>
````
Replace `<test_name>` with any Python test file in the `tests/` folder, for instance:
```bash
python -m tests.test_1bss
````
## Configuration

The simulator can be customized through two main configuration files:

- **`src/sim_params.py`** controls low-level simulation parameters, including inter-frame space durations, contention window limits, header sizes, spatial stream count, bonding mode, number of channels, and other PHY and MAC layer settings.

- **`src/user_config.py`** defines high-level behavior such as simulation duration, random seed, agent architecture (single or multi-agent), learning algorithm (UCB, OSUB, SW-LinUCB, E-RLB, etc.), as well as options for enabling statistics, logging, and traffic trace recording or using `wandb`. It also allows to configure the network layout, including the number and placement of APs, traffic characteristics, and more.

## Output and Logging
The simulator supports:

* Event logging (console and `data/events/`)

* Statistics export (`data/statistics/`)

* Figure generation (`figs/`)

## Project Structure

```bash
.
├── src/
│   ├── components/          # Core network modules
│   │   ├── network.py       # Network creation (APs and STAs)
│   │   ├── app.py           # Application layer abstraction
│   │   ├── mac.py           # MAC layer abstraction
│   │   ├── phy.py           # PHY layer abstraction
│   │   ├── medium.py        # Medium/channel behavior
│   │   └── rl_agents/       # CMAB agents, contexts, and actions
│   ├── traffic/             # Traffic generation, loading, and recording
│   ├── utils/               # Helpers: config validation, units, logger, plotting, stats
│   ├── sim_params.py        # Low-level simulation settings
│   └── user_config.py       # High-level simulation settings
├── tests/  # Toy and validation tests, including configurations and runs for published paper scenarios (see `journal/` subfolder)
├── data/
│   ├── events/              # Logged simulation events
│   └── statistics/          # Output metrics
├── figs/                    # Generated figures
├── docs/                    # Author's thesis
├── requirements.txt
└── README.md
````
