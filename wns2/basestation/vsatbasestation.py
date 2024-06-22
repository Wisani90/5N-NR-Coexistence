from wns2.basestation.generic import BaseStation
from wns2.environment.channel import Channel
from wns2.pathloss import costhata
from wns2.pathloss.freespace import FreeSpacePathLoss
from scipy import constants
import logging
import math

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class VSATBaseStation(BaseStation):
    def __init__(self, env, bs_id, position, max_data_rate=None, pathloss=None, channels=None):
        # super().__init__(env, bs_id, position, max_data_rate, pathloss)
        self.env = env
        self.bs_id = bs_id
        self.position = position
        self.bs_type = "vsat"
        # self.carrier_bandwidth = 400  # carrier bandwidth [MHz]
        self.subcarrier_bandwidth = 400  # carrier bandwidth [MHz]
        self.carrier_frequency = 3.9e9  # frequency [GHz]
        self.transmission_power = 30  # VSAT transmission power [dBm]
        self.antenna_power = 30
        self.antenna_gain = 35  # VSAT antenna gain [dBi]
        if pathloss == None:
            self.pathloss = costhata.CostHataPathLoss(costhata.EnvType.URBAN)
            
        # allocation structures
        self.ue_pb_allocation = {}
        self.ue_data_rate_allocation = {}
        self.allocated_prb = 0
        self.allocated_data_rate = 0
        #Commercial communication satellite uses a frequency band of 500 M Hz 
        # bandwidth near 6 G Hz for uplink transmission and another 500 M Hz 
        # bandwidth near 4 G Hz for downlink transmission 
        # https://arxiv.org/pdf/1206.1722#:~:text=Commercial%20communication%20satellite%20uses%20a,4.8%20G%20Hz%20is%20used.
        self.total_prb = 500
        self.T = 10 #Length of moving average for array utilization
        self.resource_utilization_array = [0] * self.T
        self.resource_utilization_counter = 0
        self.load_history = []
        self.data_rate_history = []
        self.RBG_size = 2
        self.channel = Channel.get_closest_channel(channels, self.carrier_frequency)
        
        return

    def get_position(self):
        return self.position
    def get_carrier_frequency(self):
        return self.carrier_frequency
    def get_bs_type(self):
        return self.bs_type
    def get_id(self):
        return self.bs_id
    def get_usage_ratio(self):
        """
        Calculate the ratio of allocated Physical Resource Blocks (PRBs) to total PRBs available for the base station.

        Returns:
        - Usage ratio: Allocated PRBs / Total PRBs
        """
        return self.allocated_prb / self.total_prb
    
    def compute_rsrp(self, ue):
        return self.transmission_power + self.antenna_gain - self.pathloss.compute_path_loss(ue, self)

    def compute_sinr(self, rsrp):
        interference = 0
        for elem in rsrp:
            bs_i = self.env.bs_by_id(elem)
            if elem != self.bs_id and bs_i.get_carrier_frequency() == self.carrier_frequency:
                rbur_i = bs_i.get_rbur()
                interference += (10 ** (rsrp[elem]/10)) * rbur_i
        thermal_noise = constants.Boltzmann * 293.15 * self.carrier_bandwidth * 1e6
        sinr = (10**(rsrp[self.bs_id]/10)) / (thermal_noise + interference)
        return 10 * math.log10(sinr)
    