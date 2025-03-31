from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.utils.file_manager import get_project_root
from src.utils.event_logger import get_logger
from src.components.network import AP, STA, Network

from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from collections import deque

import os
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif"]
rcParams["mathtext.fontset"] = "dejavuserif"


# https://github.com/matplotlib/matplotlib/issues/21688
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class BasePlotter:
    """Base class for all plotters, handling saving and displaying plots."""

    def __init__(self, cfg: cfg, sparams: sparams, env=None):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.name = "PLOTTER"
        self.logger = get_logger(self.name, cfg, sparams, env)

    def save_plot(self, fig, save_name="plot", save_format="pdf"):
        """Saves the plot with a unique filename."""
        if not save_name or not save_format:
            return

        save_folder = os.path.join(get_project_root(), self.cfg.FIGS_SAVE_PATH)
        os.makedirs(save_folder, exist_ok=True)

        filepath = os.path.join(save_folder, f"{save_name}.{save_format}")
        fig.savefig(filepath)
        self.logger.info(f"Plot saved to {filepath}")


class NetworkPlotter(BasePlotter):
    """Handles plotting a network graph."""

    def __init__(self, cfg: cfg, sparams: sparams, env=None):
        super().__init__(cfg, sparams, env)

    def plot_network(
        self,
        network: Network,
        node_size=80,
        label_nodes=True,
        show_distances=True,
        save_name="network_3d",
        save_format="pdf",
    ):
        """
        Plots a network graph using matplotlib.

        Parameters
        ----------
        network : Network
            The network to plot.
        node_size : int, optional
            Size of the node markers. Defaults to 80.
        label_nodes : bool, optional
            Whether to label each node with its ID. Defaults to True.
        show_distances : bool, optional
            Whether to display the distances of each edge between nodes. Defaults to True.
        save_name : str, optional
            Name of the saved image file. Defaults to "network_graph".
        save_format : str, optional
            Format of the saved image file. Defaults to "pdf".

        Returns
        -------
        None
        """
        if network is None or nx.is_empty(network.graph):
            self.logger.error("Network is empty. Nothing to plot.")
            return

        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        plt.ion()

        self.logger.header(f"Generating Network 3D plot...")

        # Create a figure and a 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        nodes = network.get_nodes()
        pos = nx.get_node_attributes(network.graph, "pos")

        unique_bss = set(node.bss_id for node in nodes if isinstance(node, (AP, STA)))
        cmap = plt.get_cmap("tab20", len(unique_bss))
        bss_colors = {bss_id: cmap(i) for i, bss_id in enumerate(unique_bss)}

        # Plot nodes
        for id, (x, y, z) in pos.items():
            node = network.get_node(id)
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
                    f"{id}",
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

            x1, y1, z1 = pos[node_src.id]
            x2, y2, z2 = pos[node_dst.id]

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
                    f"{distance}",
                    color="#272727",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )

        # Plot traffic arrows
        for node_src in nodes:
            for traffic_source in node_src.traffic_flows:
                x1, y1, z1 = pos[node_src.id]
                x2, y2, z2 = pos[traffic_source.dst_id]

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

        legend_elements_bss = [
            mlines.Line2D([0], [0], color=bss_colors[bss], lw=4, label=f"BSS: {bss}")
            for bss in unique_bss
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
    """Handles plotting generated traffic data."""

    def __init__(self, cfg: cfg, sparams: sparams, env=None, max_data_units=500):
        super().__init__(cfg, sparams, env)
        self.traffic_data = deque(maxlen=max_data_units)

    def add_data_unit(self, data_unit):
        """Adds a new data unit for plotting."""
        self.traffic_data.append(data_unit)

    def _plot_traffic(
        self, time_attr, title="Traffic", save_name="traffic", save_format="pdf"
    ):
        """
        Generalized function for plotting traffic data based on a time attribute.
        `time_attr` can be 'creation_time_us' or 'reception_time_us'.
        """
        if not self.traffic_data:
            self.logger.error(f"No data to plot: {title}")
            return

        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        plt.ion()
        fig, ax = plt.subplots(figsize=(6.4, 2.4))

        x_values_ms = [
            getattr(data_unit, time_attr) / 1e3 for data_unit in self.traffic_data
        ]
        y_values_bytes = [data_unit.size_bytes for data_unit in self.traffic_data]
        data_unit_types = {data_unit.type for data_unit in self.traffic_data}

        color_map = {
            "RTS": mcolors.TABLEAU_COLORS["tab:orange"],
            "CTS": mcolors.TABLEAU_COLORS["tab:green"],
            "DATA": mcolors.TABLEAU_COLORS["tab:cyan"],
            "MPDU": mcolors.TABLEAU_COLORS["tab:blue"],
            "BACK": mcolors.TABLEAU_COLORS["tab:purple"],
            "DEFAULT": mcolors.TABLEAU_COLORS["tab:brown"],
        }

        colors = [
            color_map.get(data_unit.type, color_map["DEFAULT"])
            for data_unit in self.traffic_data
        ]

        ax.scatter(x_values_ms, y_values_bytes, s=50, color=colors, marker="|")

        ax.set_ylim((0, None))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Simulation Time (ms)")
        ax.set_ylabel("Data Unit Size (bytes)")

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

        plt.tight_layout()

        if self.cfg.ENABLE_FIGS_SAVING and save_name is not None:
            self.save_plot(fig, save_name, save_format)

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
    def __init__(self, cfg: cfg, sparams: sparams, env=None):
        super().__init__(cfg, sparams, env)

    def validate_data(self, data, m_list, cw_mins):
        """Logs errors for missing or incorrect data but continues plotting."""
        for cw_min in cw_mins:
            if cw_min not in data:
                self.logger.error(f"Missing CW_MIN={cw_min} in data.")
                continue

            for m in m_list:
                if m not in data[cw_min]:
                    self.logger.error(f"Missing m={m} for CW_MIN={cw_min}")
                    continue

                for n, values in data[cw_min][m].items():
                    if not isinstance(n, int):
                        self.logger.error(
                            f"Invalid n={n} (should be int) for CW_MIN={cw_min}, m={m}"
                        )
                    if (
                        not isinstance(values, dict)
                        or "simulated" not in values
                        or "theoretical" not in values
                    ):
                        self.logger.error(
                            f"Missing required keys in data[{cw_min}][{m}][{n}]"
                        )
                    if not (0 <= values.get("simulated", 0) <= 1) or not (
                        0 <= values.get("theoretical", 0) <= 1
                    ):
                        self.logger.error(
                            f"Invalid collision probability in data[{cw_min}][{m}][{n}] (should be between 0 and 1)"
                        )

    def plot_prob(
        self,
        data: dict,
        m_list: list,
        cw_mins: list,
        save_name="collision_prob",
        save_format="pdf",
    ):
        plt.ion()

        if not self.cfg.ENABLE_FIGS_SAVING and not self.cfg.ENABLE_FIGS_DISPLAY:
            return

        self.validate_data(data, m_list, cw_mins)

        num_subplots = len(cw_mins)
        
        if len(cw_mins) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))

        elif len(cw_mins) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(6.4*2, 4.8))
        else:
            num_cols = 3
            num_rows = math.ceil(num_subplots / num_cols)  # Auto-adjust row count
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.4*3, 4.8*num_rows))
        if len(cw_mins) == 1:
            axes = np.array([axes])
        axes = axes.flatten() 

        tableau_colors = list(mcolors.TABLEAU_COLORS.values())  # Use Tableau colors
        markers = {"simulated": "o--", "theoretical": "s-"}  # Line styles with markers
        
        legend_elements = []  # Custom legend elements

        for i, m in enumerate(m_list):
            color = tableau_colors[i % len(tableau_colors)]
            # Add rectangle legend entry for m value
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f"m={m}")
            )

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
        for idx, (cw_min, ax) in enumerate(zip(cw_mins, axes)):
            ax.set_title(r"$\text{CW}_{\text{min}} = $" + f"{cw_min}", fontsize=10, loc='left')
            ax.set_ylabel("Collision Probability")
            ax.set_xlabel("Number of BSSs")
            ax.set_ylim(0, 1)

            for i, m in enumerate(m_list):
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

        plt.tight_layout()

        if self.cfg.ENABLE_FIGS_SAVING and save_name is not None:
            self.save_plot(fig, save_name, save_format)

        if self.cfg.ENABLE_FIGS_DISPLAY:
            plt.show()
        else:
            plt.close(fig)
