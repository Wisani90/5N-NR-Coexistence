class Channel:
    def __init__(self, frequency):
        self.frequency = frequency  # Frequency in MHz
        self.transmit_power = 25.0  # Example transmit power in dBm
        
    def get_frequency(self):
        return self.frequency

    def get_transmit_power(self):
        return self.transmit_power

    @staticmethod
    def get_closest_channel(channel_list, target_frequency):
        if not channel_list:
            raise ValueError("Channel list cannot be empty.")

        closest_channel = None
        min_frequency_difference = float('inf')  # Initialize with a large number

        for channel in channel_list:
            if channel != Channel:
                frequency_difference = abs(target_frequency - channel.frequency)
                if frequency_difference < min_frequency_difference:
                    min_frequency_difference = frequency_difference
                    closest_channel = channel

        return closest_channel