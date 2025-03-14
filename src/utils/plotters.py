import matplotlib.pyplot as plt
import os
import src.components.network as Network
import networkx as nx
import matplotlib.colors as mcolors

from matplotlib import rcParams
from collections import deque

from src.utils.support import get_project_root, get_unique_filename
from src.utils.event_logger import get_logger
from src.sim_config import ENABLE_FIGS_DISPLAY, ENABLE_FIGS_SAVING, FIGS_SAVE_PATH, ENABLE_FIGS_OVERWRITE


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
        os.makedirs(save_folder, exist_ok=True)  # Ensure folder exists

        if not ENABLE_FIGS_OVERWRITE:
            filepath = get_unique_filename(
                save_folder, save_name, save_format)
        else:
            filepath = os.path.join(
                save_folder, f"{save_name}.{save_format}")
        fig.savefig(filepath)
        self.logger.debug(f"Saved plot: {filepath}")


class NetworkPlotter(BasePlotter):
    """Handles plotting a network graph."""

    def __init__(self, env=None):
        super().__init__(env)

    def plot_network(self, network: Network, node_size=30, edge_alpha=0.6, label_nodes=True, show_distances=True, save_name="network_3d", save_format="pdf"):
        """
        Plots the network graph using matplotlib.

        Parameters
        ----------
        network : Network
            The network to plot.
        node_size : int, optional
            Size of the node markers. Defaults to 30.
        edge_alpha : float, optional
            Transparency of the edges. Defaults to 0.6.
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
        if network is None:
            return

        plt.ion()

        self.logger.header(f"Generating network graph plot...")

        # Create a figure and a 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        pos = nx.get_node_attributes(network.graph, "pos")
        channels = nx.get_edge_attributes(network.graph, "channel")
        unique_channels = sorted(set(channels.values()))
        num_channels = len(unique_channels)
        cmap = plt.get_cmap("viridis", num_channels)
        channel_colors = {ch: cmap(i) for i, ch in enumerate(unique_channels)}

        # Plot edges
        for edge in network.graph.edges():
            x1, y1, z1 = pos[edge[0]]
            x2, y2, z2 = pos[edge[1]]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=channel_colors[channels[edge]],
                    alpha=edge_alpha)

            # Label the edges with distance if requested
            if show_distances:
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
                        f"{network.get_distance_between_nodes(network.get_node_by_id(edge[0]), network.get_node_by_id(edge[1])):.2f}", color=channel_colors[channels[edge]],  fontsize=8, ha='center', va='center', zorder=10)

        # Plot nodes
        for id, (x, y, z) in pos.items():
            ax.scatter(x, y, z, c="white", edgecolors='black',
                       marker="o", s=node_size)

            # Label the nodes if requested
            if label_nodes:
                z_min, z_max = ax.get_zlim()
                z_range = z_max - z_min
                relative_offset = z_range * 0.08
                ax.text(x, y, z + relative_offset, f"{id}", fontsize=8,
                        ha='center', va='center', zorder=10)

        # Set the labels, title and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Network Graph (t={network.env.now} ms)")
        ax.set_facecolor('white')

        legend_elements = [plt.Line2D(
            [0], [0], color=channel_colors[ch], lw=4, label=f"Ch: {ch}") for ch in unique_channels]
        ax.legend(handles=legend_elements, loc="best",
                  fontsize="small", frameon=False)

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
            self.logger.warning("No data to plot")
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

        if ENABLE_FIGS_SAVING:
            self.save_plot(fig, save_name, save_format)

        if ENABLE_FIGS_DISPLAY:
            plt.show()

    def show_generation(self, **kwargs):
        """Plots packet creation times."""
        self._plot_traffic(time_attr="creation_time_us", **kwargs)

    def show_reception(self, **kwargs):
        """Plots packet reception times."""
        self._plot_traffic(time_attr="reception_time_us", **kwargs)
