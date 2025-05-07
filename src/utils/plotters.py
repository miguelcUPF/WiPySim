from src.sim_params import SimParams as sparams_module
from src.user_config import UserConfig as cfg_module

from src.utils.file_manager import get_project_root
from src.utils.event_logger import get_logger
from src.components.network import AP, STA, Network, Node
from src.utils.data_units import DataUnit

from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from collections import deque

import os
import math
import simpy
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif"]
rcParams["mathtext.fontset"] = "dejavuserif"

"""
Reference: https://github.com/matplotlib/matplotlib/issues/21688
"""


class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow class."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *args, **kwargs):
        """
        Initialize an Arrow3D object.

        Args:
            xs (np.ndarray): X-coordinates of the arrow in 3D space.
            ys (np.ndarray): Y-coordinates of the arrow in 3D space.
            zs (np.ndarray): Z-coordinates of the arrow in 3D space.
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d: np.ndarray = xs, ys, zs

    def do_3d_projection(self) -> float:
        """
        Perform 3D projection using the Axes' projection matrix.

        Returns:
            float: The minimum z-coordinate of the projected points.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class BasePlotter:
    """Base class for all plotters, handling saving and displaying plots."""

    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment = None):
        """
        Initialize a BasePlotter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """
        self.cfg = cfg
        self.sparams = sparams
        self.env: simpy.Environment = env

        self.name: str = "PLOTTER"
        self.logger: logging.Logger = get_logger(self.name, cfg, sparams, env)

    def save_plot(self, figure: plt.Figure, save_name: str, save_format: str) -> None:
        """
        Saves the plot with a unique filename.

        Args:
            figure (plt.Figure): The figure to save.
            save_name (str): The base name of the saved file.
            save_format (str): The format of the saved file (e.g. pdf, png).
        """
        if not save_name or not save_format:
            return

        save_folder = os.path.join(get_project_root(), self.cfg.FIGS_SAVE_PATH)
        os.makedirs(save_folder, exist_ok=True)

        file_path = os.path.join(save_folder, f"{save_name}.{save_format}")
        figure.savefig(file_path)


class NetworkPlotter(BasePlotter):
    """Plotter for network visualization."""

    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment = None):
        """
        Initialize a NetworkPlotter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """
        super().__init__(cfg, sparams, env)

    @staticmethod
    def _get_bss_colors(nodes: list[Node]) -> dict:
        """
        Returns a dictionary mapping BSS IDs to colors."""
        unique_bss = set(node.bss_id for node in nodes)
        cmap = plt.get_cmap("tab20", len(unique_bss))
        return {bss_id: cmap(i) for i, bss_id in enumerate(unique_bss)}

    def plot_network(
        self,
        network: Network,
        node_size: int = 80,
        label_nodes: bool = True,
        show_distances: bool = True,
        save_name: str = "network_3d",
        save_format: str = "pdf",
    ):
        """
        Plots a network graph using matplotlib.

        Args:
            network (Network): The network to plot.
            node_size (int, optional): The size of the nodes in the plot. Defaults to 80.
            label_nodes (bool, optional): Whether to label the nodes with their IDs. Defaults to True.
            show_distances (bool, optional): Whether to show the distances between nodes. Defaults to True.
            save_name (str, optional): The base name of the saved file. Defaults to "network_3d".
            save_format (str, optional): The format of the saved file (e.g. pdf, png). Defaults to "pdf".
        """
        if network is None or nx.is_empty(network.graph):
            self.logger.error("Network is empty. Nothing to plot.")
            return

        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        self.logger.header(f"Generating Network 3D plot...")

        # Create a figure and a 3D Axes
        plt.ion()
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")

        nodes = network.get_nodes()
        positions = nx.get_node_attributes(network.graph, "pos")

        bss_colors = self._get_bss_colors(nodes)

        # Plot nodes
        for node in nodes:
            x, y, z = positions[node.id]
            node_size_factor = 1
            if isinstance(node, AP):
                marker = "o"  # Circle for APs
                node_size_factor = 1.5
            elif isinstance(node, STA):
                marker = "s"  # Square for STAs
            else:
                marker = "D"

            color = bss_colors[node.bss_id]

            ax.scatter(
                x,
                y,
                z,
                c=[color],
                edgecolors=["black"],
                marker=marker,
                s=node_size * node_size_factor,
            )

            if label_nodes:
                ax.text(
                    x,
                    y,
                    z,
                    f"{node.id}",
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="center",
                    zorder=50,
                    fontweight="bold",
                )

        # Plot edges
        for edge in network.graph.edges:
            node_src = network.get_node(edge[0])
            node_dst = network.get_node(edge[1])

            color = bss_colors[node_src.bss_id]

            x1, y1, z1 = positions[node_src.id]
            x2, y2, z2 = positions[node_dst.id]

            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, lw=1)

            # Label the edges with distance if requested
            if show_distances:
                distance = network.get_distance_between_nodes(
                    node_src.id, node_dst.id, 2
                )
                ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    (z1 + z2) / 2,
                    f"{distance:.2f}",
                    color="#272727",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )

        # Plot traffic arrows
        for node_src in nodes:
            for traffic_source in node_src.traffic_flows:
                x1, y1, z1 = positions[node_src.id]
                x2, y2, z2 = positions[traffic_source.dst_id]

                color = bss_colors[node_src.bss_id]

                # Create a 3D arrow by plotting the line in 3D space
                arrow = Arrow3D(
                    (x1, x2),
                    (y1, y2),
                    (z1, z2),
                    mutation_scale=20,
                    color=color,
                    lw=1,
                    arrowstyle="->",
                )
                ax.add_patch(arrow)

        # Create legend
        legend_elements_bss = [
            mlines.Line2D([0], [0], color=bss_colors[bss], lw=4, label=f"BSS: {bss}")
            for bss in set(node.bss_id for node in nodes)
        ]
        legend_elements_nodes = [
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="none",
                linestyle="None",
                markeredgewidth=1,
                markersize=5,
                label="AP",
            ),
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="s",
                markerfacecolor="none",
                linestyle="None",
                markeredgewidth=1,
                markersize=5,
                label="STA",
            ),
        ]

        ax.legend(
            handles=legend_elements_bss + legend_elements_nodes,
            loc="best",
            fontsize="small",
            frameon=False,
        )

        # Set the labels, title and background color
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Network Graph (t={network.env.now / 1000} ms)")
        ax.set_facecolor("white")

        plt.tight_layout()

        if self.cfg.ENABLE_FIGS_SAVING:
            self.save_plot(fig, save_name, save_format)
        if self.cfg.ENABLE_FIGS_DISPLAY:
            plt.show()
        else:
            plt.close(fig)


class TrafficPlotter(BasePlotter):
    """Plotter for traffic data."""

    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment = None,
        max_data_units: int = 500,
    ):
        """
        Initialize a TrafficPlotter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
            max_data_units (int, optional): The maximum number of data units to store. Defaults to 500.
        """
        super().__init__(cfg, sparams, env)
        self.traffic_data = deque(maxlen=max_data_units)

    def add_data_unit(self, data_unit: DataUnit):
        """Adds a new data unit to the queue for plotting."""
        self.traffic_data.append(data_unit)

    def _plot_traffic(
        self,
        time_attr: str,
        title: str = "Traffic",
        save_name: str = "traffic",
        save_format: str = "pdf",
        show_xticks: bool = True,
        show_xlabel: bool = True,
        show_legend: bool = True,
        show_yticks: bool = True,
        show_ylabel: bool = True,
        start_x_from_zero: bool = False,
        fig_size: tuple[float, float] = (6.4, 2.4),
    ) -> None:
        """
        Generalized function for plotting traffic data based on a time attribute.

        Args:
            time_attr (str): The time attribute to use for the x-axis. Must be 'creation_time_us' or 'reception_time_us'.
            title (str, optional): The title of the plot. Defaults to "Traffic".
            save_name (str, optional): The name to use for saving the plot. Defaults to "traffic".
            save_format (str, optional): The format to use for saving the plot. Defaults to "pdf".
        """

        # If there is no data to plot, log an error and return
        if not self.traffic_data:
            self.logger.error(f"No data to plot: {title}")
            return

        # If both saving and displaying are disabled, return
        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        # Create a new figure and axis
        plt.ion()
        fig, ax = plt.subplots(figsize=fig_size)

        # Get first time value
        first_data_unit = min(self.traffic_data, key=lambda p: getattr(p, time_attr))
        print(getattr(first_data_unit, time_attr))

        # Extract x and y values from the data
        x_values_ms = [
            (
                getattr(data_unit, time_attr) / 1e3
                - (
                    0
                    if not start_x_from_zero
                    else getattr(first_data_unit, time_attr) / 1e3
                )
            )
            for data_unit in self.traffic_data
        ]
        y_values_bytes = [data_unit.size_bytes for data_unit in self.traffic_data]

        # Extract data unit types from the data
        data_unit_types = {data_unit.type for data_unit in self.traffic_data}

        # Create a color map for the data unit types
        color_map = {
            "RTS": mcolors.TABLEAU_COLORS["tab:orange"],
            "CTS": mcolors.TABLEAU_COLORS["tab:green"],
            "DATA": mcolors.TABLEAU_COLORS["tab:blue"],
            "MPDU": mcolors.TABLEAU_COLORS["tab:cyan"],
            "BACK": mcolors.TABLEAU_COLORS["tab:purple"],
            "DEFAULT": mcolors.TABLEAU_COLORS["tab:brown"],
        }

        # Create a list of colors for the scatter plot
        colors = [
            color_map.get(data_unit.type, color_map["DEFAULT"])
            for data_unit in self.traffic_data
        ]

        # Create the scatter plot
        ax.scatter(
            x_values_ms,
            y_values_bytes,
            s=50,
            color=colors,
            marker="|",
        )

        # Set the y-axis limits
        ax.set_ylim((0, None))

        # Set the title and labels
        ax.set_title(title, loc="left")

        if not show_xticks:
            plt.xticks([])
        if not show_yticks:
            plt.yticks([])

        # Set the x-axis limits
        if show_xlabel:
            ax.set_xlabel("Time (ms)")
        if show_ylabel:
            ax.set_ylabel("Size (bytes)")

        # Create a legend for the data unit types
        if show_legend:
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, label=label, color=color_map[label])
                for label in data_unit_types
                if label in color_map
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                fontsize="small",
                frameon=False,
                bbox_to_anchor=(1, 0.5),
            )

        # Adjust the layout
        plt.tight_layout()

        # Save the plot
        if self.cfg.ENABLE_FIGS_SAVING and save_name is not None:
            self.save_plot(fig, save_name, save_format)

        # Display the plot
        if self.cfg.ENABLE_FIGS_DISPLAY:
            plt.show()
        else:
            plt.close(fig)

    def plot_generation(self, **kwargs):
        """Plots packet creation times."""
        self._plot_traffic(time_attr="creation_time_us", **kwargs)

    def plot_reception(self, **kwargs):
        """Plots packet reception times."""
        self._plot_traffic(time_attr="reception_time_us", **kwargs)


class CollisionProbPlotter(BasePlotter):
    """Plotter for collision probabilities."""

    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment = None):
        """
        Initializes a CollisionProbPlotter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """
        super().__init__(cfg, sparams, env)

    def validate_data(self, data: dict, m_values: list[int], cw_min_values: list[int]):
        """Validates the data passed to the plot method."""
        for cw_min in cw_min_values:
            cw_min_data = data.get(cw_min)
            if cw_min_data is None:
                self.logger.error(f"Missing CW_MIN={cw_min} in data.")
                continue

            for m in m_values:
                m_data = cw_min_data.get(m)
                if m_data is None:
                    self.logger.error(f"Missing m={m} for CW_MIN={cw_min}")
                    continue

                for n, collision_probabilities in m_data.items():
                    if not isinstance(n, int):
                        self.logger.error(
                            f"Invalid n={n} (should be int) for CW_MIN={cw_min}, m={m}"
                        )
                    if not isinstance(collision_probabilities, dict):
                        self.logger.error(
                            f"Invalid collision probabilities in data[{cw_min}][{m}][{n}] (should be dict)"
                        )

                    simulated_prob = collision_probabilities.get("simulated")
                    if not (0 <= simulated_prob <= 1):
                        self.logger.error(
                            f"Invalid simulated collision probability in data[{cw_min}][{m}][{n}] (should be between 0 and 1)"
                        )

                    theoretical_prob = collision_probabilities.get("theoretical")
                    if not (0 <= theoretical_prob <= 1):
                        self.logger.error(
                            f"Invalid theoretical collision probability in data[{cw_min}][{m}][{n}] (should be between 0 and 1)"
                        )

    def plot_prob(
        self,
        data: dict,
        m_values: list[int],
        cw_mins: list[int],
        save_name: str = "collision_prob",
        save_format: str = "pdf",
    ):
        """
        Plots collision probabilities for different CWmin and m values.
        The data is expected to be structured as follows:
            {
                cw_min: {
                    m: {
                        n: {
                            "simulated": float,
                            "theoretical": float
                        }
                    }
                }
            }

        Args:
            data (dict): The data to plot.
            m_values (list[int]): List of m values. (cw_max = 2^m * cw_min)
            cw_min_values (list[int]): List of cw_min values.
            save_name (str, optional): The name of the plot file. Defaults to "collision_prob".
            save_format (str, optional): The format of the plot file. Defaults to "pdf".
        """
        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        plt.ion()

        # Validate the input data
        self.validate_data(data, m_values, cw_mins)

        num_subplots = len(cw_mins)
        num_cols = len(cw_mins) if len(cw_mins) <= 3 else 3
        num_rows = math.ceil(num_subplots / num_cols)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(6.4 * num_cols, 4.8 * num_rows)
        )

        axes = np.array([axes]) if num_subplots == 1 else axes
        axes = axes.flatten()

        # Use Tableau colors
        tableau_colors = list(mcolors.TABLEAU_COLORS.values())

        # Line styles with markers
        markers = {"simulated": "o--", "theoretical": "s-"}

        # Custom legend elements
        legend_elements = []

        # Add custom legend elements for each m value
        for i, m in enumerate(m_values):
            color = tableau_colors[i % len(tableau_colors)]
            # Add rectangle legend entry for m value
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f"m={m}")
            )

        # Add custom legend elements for simulated and theoretical data
        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                marker="o",
                markerfacecolor="none",
                label="Simulated",
            )
        )
        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                marker="s",
                markerfacecolor="none",
                label="Theoretical",
            )
        )

        # Plot the data
        for idx, (cw_min, ax) in enumerate(zip(cw_mins, axes)):
            ax.set_title(
                r"$\text{CW}_{\text{min}} = $" + f"{cw_min}", fontsize=10, loc="left"
            )
            ax.set_ylabel("Collision Probability")
            ax.set_xlabel("Number of BSSs")
            ax.set_ylim(0, 1)

            for i, m in enumerate(m_values):
                color = tableau_colors[i % len(tableau_colors)]  # Cycle through colors

                n_values = sorted(data[cw_min][m].keys())
                sim_values = [data[cw_min][m][n]["simulated"] for n in n_values]
                theo_values = [data[cw_min][m][n]["theoretical"] for n in n_values]

                ax.plot(
                    n_values,
                    sim_values,
                    markers["simulated"],
                    color=color,
                    markerfacecolor="none",
                    markersize=3,
                )
                ax.plot(
                    n_values,
                    theo_values,
                    markers["theoretical"],
                    color=color,
                    markerfacecolor="none",
                    markersize=3,
                )
            if idx == 0:
                ax.legend(
                    handles=legend_elements,
                    loc="best",
                    fontsize=8,
                    frameon=False,
                )

        # Hide unused subplots if cw_mins < num_rows * num_cols
        for idx in range(len(cw_mins), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust the layout
        plt.tight_layout()

        # Save the plot
        if self.cfg.ENABLE_FIGS_SAVING and save_name is not None:
            self.save_plot(fig, save_name, save_format)

        # Display the plot
        if self.cfg.ENABLE_FIGS_DISPLAY:
            plt.show()
        else:
            plt.close(fig)


class DelayPerLoadPlotter(BasePlotter):
    """Plotter for mean delay vs load."""

    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment = None):
        """
        Initialize a DelayPerLoadPlotter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """
        super().__init__(cfg, sparams, env)

    def validate_data(self, data: dict, load_values: list[float]):
        """Validates the data passed to the plot method."""
        for load in load_values:
            if load not in data:
                self.logger.error(f"Missing load={load} in data.")

    def plot_mean_delay(
        self,
        data: dict[int, pd.DataFrame],
        loads_kbps: list[float],
        save_name: str = "mean_delay_vs_load",
        save_format: str = "pdf",
    ):
        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        plt.ion()

        self.validate_data(data, loads_kbps)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlabel("Traffic load (Mbps)")
        ax.set_ylabel("Delay (μs)")

        loads = np.array(sorted(loads_kbps))
        mean_delays = []
        perc_99_delays = []

        for load in loads:
            df = data[load]
            delays = df["delay_us"]
            if len(delays) == 0:
                mean_delays.append(np.nan)
                perc_99_delays.append(np.nan)
                continue
            mean_delays.append(delays.mean())
            perc_99_delays.append(np.percentile(delays, 99))

        ax.plot(
            loads / 1e3,
            mean_delays,
            "o-",
            label="mean",
            markerfacecolor="none",
            markersize=3,
        )
        ax.plot(
            loads / 1e3,
            perc_99_delays,
            "s--",
            label="99th",
            markerfacecolor="none",
            markersize=3,
        )
        ax.legend(fontsize=8, frameon=False)

        plt.tight_layout()

        if self.cfg.ENABLE_FIGS_SAVING:
            self.save_plot(fig, save_name, save_format)

        if self.cfg.ENABLE_FIGS_DISPLAY:
            plt.show()
        else:
            plt.close(fig)
