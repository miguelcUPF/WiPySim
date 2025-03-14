
import simpy
from src.components.network import Network, Node
from src.utils.event_logger import EventLogger
from src.utils.data_units import Packet
from src.sim_config import MAX_PACKET_SIZE_bytes

if __name__ == "__main__":

    env = simpy.Environment()
    env.event_logger = EventLogger(logging=False, debug=True)
    env.network = Network(env)

    env.event_logger.log_event(env.now, "Main", "Starting MAC test...", "header")

    # Create nodes
    nodes = [
        Node(env, 1, (0, 0, 0)),
        Node(env, 2, (3, 4, 0))
    ]

    # Create bidirectional links
    bidirectional_links = [
        (nodes[0], nodes[1]),  # 1 <-> 2, ch None
    ]

    node1 = nodes[0]
    node1.mac_layer.set_selected_channels([1])

    # Create some Packets and send them to the MAC layer
    # packet = Packet(id=1, size=MAX_PACKET_SIZE_bytes, source=node1.id, destination=node1.id, creation_time=env.now)
    # node1.mac_layer.tx_enqueue(packet)

    # packet = Packet(id=2, size=MAX_PACKET_SIZE_bytes, source=node1.id, destination=node1.id, creation_time=env.now)
    # node1.mac_layer.tx_enqueue(packet)
    node1.app_layer.start_traffic

    env.event_logger.log_event(env.now, "Main", f"Node 1 MAC tx queue of length: {len(node1.mac_layer.tx_queue.items)}", "info")
                               
    node1.mac_layer.run()
    
    env.event_logger.log_event(env.now, "Main", "Running the simulation...", "header")
    env.run(until=3000) 