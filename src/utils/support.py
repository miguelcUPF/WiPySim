from src.components.network import Network
from src.traffic.generator import TrafficGenerator
from src.traffic.loader import TrafficLoader

import simpy


def initialize_network(env: simpy.Environment, bsss_config: dict, network: Network = None):
    if not network:
        network = Network(env)

    for bss in bsss_config:
        bss_id = bss["id"]

        # Create the AP
        ap_id = bss["ap"]["id"]
        ap_pos = bss["ap"]["pos"]
        ap = network.add_ap(ap_id, ap_pos, bss_id)

        # Create associated STAs
        for sta in bss["stas"]:
            sta_id = sta["id"]
            sta_pos = sta["pos"]
            network.add_sta(sta_id, sta_pos, bss_id, ap)
        for flow in bss["traffic_flows"]:
            dst_id = flow["destination"]

            src_node = network.get_node(ap_id)

            if "file" in flow:
                traffic_loader = TrafficLoader(env, src_node, dst_id, **flow["file"])
                src_node.add_traffic_flow(traffic_loader)
            if "model" in flow:
                traffic_generator = TrafficGenerator(env, src_node, dst_id, **flow["model"])
                src_node.add_traffic_flow(traffic_generator)
