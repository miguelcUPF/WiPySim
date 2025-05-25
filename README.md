# WiPySim: An IEEE 802.11 DCF Wireless Network Simulator with CMAB-Based Optimization

This Python event-driven simulator, built using SimPy, models **IEEE 802.11 Distributed Coordination Function (DCF) wireless networks** with support for **contextual multi-armed bandit (CMAB)** algorithms to optimize parameters such as the **channel group allocation**, **primary channel selection**, and the **Contention Window (CW) size**. 

It supports both **single-agent** and **multi-agent** architectures, including the [**Sliding Window Linear Upper Confidence Bound (SW-LinUCB)**](https://dl.acm.org/doi/abs/10.1145/3297280.3297440?casa_token=ZlbqTUxIPakAAAAA:Mm-uaROK5Qy1tzSB0Tr1j19Z883qIK4rbTHL7tjPRs9XnaasbhX9hWUyz22a-YwEF-Ecwx9x4YE) algorithm and an **epsilon-greedy RMSProp-based** algorithm (described in the thesis located in the `docs/` directory)

It has been developed as part of the continuous research of the author, [**Miguel Casasnovas Bielsa**](https://scholar.google.com/citations?user=jRxBfaMAAAAJ&hl=ca), within the [**Wireless Networking Research Group**](https://www.upf.edu/web/wnrg) at **Universitat Pompeu Fabra (UPF)**, and as part of its Master’s Thesis in Data Science at **Universitat Oberta de Catalunya (UOC)**.

## Features
- Detailed IEEE 802.11 DCF simulation
- Support for single and multi-agent CMAB architectures
- Configurable wireless and traffic parameters
- Event logging and statistics collection
- Traffic trace recording and loading (from CSV with `"frame.time_relative"` and `"frame.len"` columns)
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

- **`src/user_config.py`** defines high-level behavior such as simulation duration, random seed, agent architecture (single or multi-agent), learning algorithm (SW-LinUCB or epsilon-greedy RMSProp), as well as options for enabling statistics, logging, and traffic trace recording or using `wandb`. It also allows to configure the network layout, including the number and placement of APs, traffic characteristics, and more.

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
├── tests/                   # Toy and validation tests
├── data/
│   ├── events/              # Logged simulation events
│   └── statistics/          # Output metrics
├── figs/                    # Generated figures
├── docs/                    # Author's thesis
├── requirements.txt
└── README.md
````
