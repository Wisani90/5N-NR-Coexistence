import math

class FreeSpacePathLoss:
    def __init__(self):
        self.c = 3e8  # Speed of light in meters/second

    def compute_path_loss(self, distance, frequency):
        """
        Compute the free space path loss for a given distance and frequency.

        Args:
        - distance: The distance between the transmitter and receiver in meters.
        - frequency: The frequency of the signal in hertz.

        Returns:
        - path_loss: The path loss in dB.
        """
        if distance == 0:
            # raise ValueError("Distance must be greater than zero")
            return 0
        
        # Calculate the path loss using the FSPL formula
        path_loss = 20 * math.log10(distance) + 20 * math.log10(frequency) + 20 * math.log10(4 * math.pi / self.c)
        
        return path_loss
