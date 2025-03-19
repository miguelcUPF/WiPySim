import matplotlib.pyplot as plt
import os
import src.components.network as Network
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from matplotlib import rcParams
from collections import deque

from src.utils.file_manager import get_project_root
from src.utils.event_logger import get_logger
from src.user_config import ENABLE_FIGS_DISPLAY, ENABLE_FIGS_SAVING, FIGS_SAVE_PATH
from src.components.network import AP, STA

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
rcParams["mathtext.fontset"] = 'dejavuserif'


class BasePlotter:
    """Base class for all plotters, handling saving and displaying plots."""

    def __init__(self, env=None):
        self.name = "PLOTTER"

        self.logger = get_logger(self.name, env)

    def save_plot(self, fig, save_name="plot", save_format="pdf"):
        """Saves the plot with a unique filename."""
        if not save_name or not save_format:
            return

        save_folder = os.path.join(get_project_root(), FIGS_SAVE_PATH)
        os.makedirs(save_folder, exist_ok=True)

        filepath = os.path.join(
            save_folder, f"{save_name}.{save_format}")
        fig.savefig(filepath)
        self.logger.debug(f"Saved plot: {filepath}")


class NetworkPlotter(BasePlotter):
    """Handles plotting a network graph."""

    def __init__(self, env=None):
        super().__init__(env)

    def plot_network(self, network: Network, node_size=80, label_nodes=True, show_distances=True, save_name="network_3d", save_format="pdf"):
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
            self.logger.warning("Network is empty. Nothing to plot.")
            return

        plt.ion()

        self.logger.header(f"Generating Network 3D plot...")

        # Create a figure and a 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        nodes = network.get_nodes()
        pos = nx.get_node_attributes(network.graph, "pos")

        unique_bss = set(
            node.bss_id for node in nodes if isinstance(node, (AP, STA)))
        cmap = plt.get_cmap("tab20", len(unique_bss))
        bss_colors = {bss_id: cmap(i) for i, bss_id in enumerate(unique_bss)}

        # Plot nodes
        for id, (x, y, z) in pos.items():
            node = network.get_node(id)
            if isinstance(node, AP):
                marker = "o"  # Circle for APs
            elif isinstance(node, STA):
                marker = "s"  # Square for STAs
            else:
                marker = "D"

            color = bss_colors[node.bss_id]

            ax.scatter(x, y, z, c=[color], edgecolors=[
                       "black"], marker=marker, s=node_size)

            if label_nodes:
                ax.text(x, y, z, f"{id}", fontsize=7, color="black",
                        ha='center', va='center', zorder=50, fontweight='bold')

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
                    node_src.id, node_dst.id, 2)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
                        f"{distance}", color="#272727",  fontsize=8, ha='center', va='center', zorder=10)

        # Plot traffic arrows
        for node_src in nodes:
            for traffic_source in node_src.traffic_sources:
                x1, y1, z1 = pos[node_src.id]
                x2, y2, z2 = pos[traffic_source.dst_id]

                color = bss_colors[node_src.bss_id]

                ax.quiver(x1, y1, z1, x2 - x1, y2 - y1, z2 - z1,
                          color=color, arrow_length_ratio=0.1, lw=1)

        legend_elements_bss = [
            mlines.Line2D([0], [0], color=bss_colors[bss],
                          lw=4, label=f"BSS: {bss}")
            for bss in unique_bss
        ]
        legend_elements = [
            mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None',
                          markeredgewidth=1, markersize=5, label="STA"),
            mlines.Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='None', markeredgewidth=1,
                          markersize=5, label="AP")
        ]

        ax.legend(handles=legend_elements_bss + legend_elements, loc="best",
                  fontsize="small", frameon=False)

        # Set the labels, title and background color
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Network Graph (t={network.env.now} ms)")
        ax.set_facecolor('white')

        plt.tight_layout()

        if ENABLE_FIGS_SAVING:
            self.save_plot(fig, save_name, save_format)
        if ENABLE_FIGS_DISPLAY:
            plt.show()


class TrafficPlotter(BasePlotter):
    """Handles plotting generated traffic data."""

    def __init__(self, env=None, max_data_units=500):
        super().__init__(env)
        self.traffic_data = deque(maxlen=max_data_units)

    def add_data_unit(self, data_unit):
        """Adds a new data unit for plotting."""
        self.traffic_data.append(data_unit)

    def _plot_traffic(self, time_attr, title="Traffic", save_name="traffic", save_format="pdf"):
        """
        Generalized function for plotting traffic data based on a time attribute.
        `time_attr` can be 'creation_time_us' or 'reception_time_us'.
        """
        if not self.traffic_data:
            self.logger.warning(f"No data to plot: {title}")
            return

        plt.ion()
        fig, ax = plt.subplots(figsize=(6.4, 2.4))

        x_values_ms = [getattr(data_unit, time_attr) /
                       1e3 for data_unit in self.traffic_data]
        y_values_bytes = [
            data_unit.size_bytes for data_unit in self.traffic_data]
        data_unit_types = {data_unit.type for data_unit in self.traffic_data}

        color_map = {
            "RTS": mcolors.TABLEAU_COLORS["tab:orange"],
            "CTS": mcolors.TABLEAU_COLORS["tab:green"],
            "DATA": mcolors.TABLEAU_COLORS["tab:cyan"],
            "MPDU": mcolors.TABLEAU_COLORS["tab:blue"],
            "BACK": mcolors.TABLEAU_COLORS["tab:purple"],
            "DEFAULT": mcolors.TABLEAU_COLORS["tab:brown"]
        }

        colors = [color_map.get(data_unit.type, color_map["DEFAULT"])
                  for data_unit in self.traffic_data]

        ax.scatter(x_values_ms, y_values_bytes, s=50, color=colors, marker="|")

        ax.set_ylim((0, None))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Simulation Time (ms)")
        ax.set_ylabel("Data Unit Size (bytes)")

        legend_elements = [plt.Rectangle((0, 0), 1, 1, label=label, color=color_map[label])
                           for label in data_unit_types if label in color_map]
        ax.legend(handles=legend_elements, loc="center left",
                  fontsize="small", frameon=False, bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if ENABLE_FIGS_SAVING and save_name is not None:
            self.save_plot(fig, save_name, save_format)

        if ENABLE_FIGS_DISPLAY:
            plt.show()

    def show_generation(self, **kwargs):
        """Plots packet creation times."""
        self._plot_traffic(time_attr="creation_time_us", **kwargs)

    def show_reception(self, **kwargs):
        """Plots packet reception times."""
        self._plot_traffic(time_attr="reception_time_us", **kwargs)
