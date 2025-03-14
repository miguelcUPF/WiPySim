import simpy


class Channel():
    def __init__(self, env: simpy.Environment, channel_id: int):
        self.env = env
        self.channel_id = channel_id
        self.busy = False  # True if the channel is busy
        self.node_id = None  # node_id that is using the channel to transmit

    def use(self, node_id: int, duration: int):
        """User occupies the channel for the given duration."""
        self.set_busy(node_id)
        yield self.env.timeout(duration)
        self.set_idle()

    def set_busy(self, node_id: int):
        self.busy = True
        self.node_id = node_id

    def set_idle(self):
        self.busy = False
        self.node_id = None


class Medium():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.channels = {1: Channel(env, 1)} # TODO: channel id as key matrix of 20 MHz channels value 0 indicates idle, and otherwise the node_id using the channel depends on the configured band

    def is_channel_busy(self, channel_id: int) -> bool:
        return self.channels[channel_id].busy

    def is_channel_idle(self, channel_id: int) -> bool:
        return not self.channels[channel_id].busy

    def is_channel_idle_for(self, channel_id: int, duration: float) -> bool:
        return self.are_channels_idle_for([channel_id], duration)
    
    def are_all_channels_idle(self, channel_ids: list[int]):
        return all(self.is_channel_idle(channel_id) for channel_id in channel_ids)

    def are_channels_idle_for(self, channel_ids: list[int], duration: float):
        """Check if all the specified channels have been idle for the given duration."""
        start_time = self.env.now
        while True:
            if any(self.is_channel_busy(channel_id) for channel_id in channel_ids):
                start_time = self.env.now
            elif self.env.now - start_time >= duration:
                self.env.event_logger.log_event(self.env.now, "Medium",f"All channels {channel_ids} have been idle for {duration} time units", type="debug")
                return
            yield self.env.timeout(1)

    def get_idle_channels(self):
        return
        # TODO: 20MHz, 40MHz, 80MHz, 160MHz, 320MHz
