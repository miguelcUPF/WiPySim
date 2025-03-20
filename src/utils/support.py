from src.components.network import Network
from src.traffic.generator import TrafficGenerator
from src.traffic.loader import TrafficLoader

import simpy
import random


def get_unique_position(bounds: tuple, used_positions: set) -> tuple:
    """Generate a unique random position within bounds."""
    while True:
        x_lim, y_lim, z_lim = bounds
        pos = (
            round(random.uniform(0, x_lim), 2),
            round(random.uniform(0, y_lim), 2),
            round(random.uniform(0, z_lim), 2),
        )
        if pos not in used_positions:
            used_positions.add(pos)
            return pos


def initialize_network(
    env: simpy.Environment, bsss_config: dict, bounds: tuple, network: Network = None
) -> Network:
    if not network:
        network = Network(env)

    used_positions = set()

    for bss in bsss_config:
        bss_id = bss["id"]

        # Create the AP
        ap_id = bss["ap"]["id"]
        ap_pos = bss["ap"].get("pos", get_unique_position(bounds, used_positions))
        ap = network.add_ap(ap_id, ap_pos, bss_id)

        # Create associated STAs
        for sta in bss.get("stas", []):
            sta_id = sta["id"]
            sta_pos = sta.get("pos", get_unique_position(bounds, used_positions))
            network.add_sta(sta_id, sta_pos, bss_id, ap)

        for flow in bss.get("traffic_flows", []):
            dst_id = flow["destination"]

            src_node = network.get_node(ap_id)

            if "file" in flow:
                traffic_loader = TrafficLoader(env, src_node, dst_id, **flow["file"])
                src_node.add_traffic_flow(traffic_loader)
            if "model" in flow:
                traffic_generator = TrafficGenerator(
                    env, src_node, dst_id, **flow["model"]
                )
                src_node.add_traffic_flow(traffic_generator)

    return network


def add_bss_automatically(BSSs, num_bss: int = 0, last_node_id: int = 0):
    bss_id = num_bss + 1
    ap_id = last_node_id + 1
    sta_id = last_node_id + 2

    new_bss = {
        "id": bss_id,
        "ap": {"id": ap_id},
        "stas": [{"id": sta_id}],
        "traffic_flows": [
            {
                "destination": sta_id,
                "model": {"name": "Poisson"},
            }
        ]
    }
    BSSs.append(new_bss)
    return BSSs
