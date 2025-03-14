import simpy
import random
from src.utils.data_units import BACK

class PHY:
    def __init__(self, env: simpy.Environment, node_id: int):
        self.env = env
        self.name = "PHY"
        self.mac_layer = None

        self.node_id = node_id

        self.ampdu = None # TODO: remove just testing

    def transmit_ampdu(self, ampdu):
        # TODO phy header etc
        self.ampdu = ampdu
        yield self.env.timeout(20)  # TODO

    def receive_back(self):
        # TODO
        #back_packet = yield self.medium.get_back_packet(self.node_id)
        timeouts = [10, 40, 90, 100]
        yield self.env.timeout(random.choice(timeouts))
        back = BACK(self.ampdu, self.node_id, self.node_id, self.env.now)
        back.add_lost_mpdus(random.sample(self.ampdu.mpdus, 1)) # TODO: remove just testing
        return back
