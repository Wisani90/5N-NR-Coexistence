from wns2.basestation.generic import BaseStation
import math
from scipy import constants
from wns2.environment import environment
from wns2.environment.channel import Channel
from wns2.pathloss import costhata
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

MAX_PRB = 200

#Table 5.3.3-1: Minimum guardband [kHz] (FR1) and Table: 5.3.3-2: Minimum guardband [kHz] (FR2), 3GPPP 38.104
#number of prb depending on the numerology (0,1,2,3), on the frequency range (FR1, FR2) and on the base station bandwidth
NRbandwidth_prb_lookup = {
    0:[{
        5:25,
        10:52,
        15:79,
        20:106,
        25:133,
        30:160,
        40:216,
        50:270
    }, None],
    1:[{
        5:11,
        10:24,
        15:38,
        20:51,
        25:65,
        30:78,
        40:106,
        50:133,
        60:162,
        70:189,
        80:217,
        90:245,
        100:273
    }, None],
    2:[{
        10:11,
        15:18,
        20:24,
        25:31,
        30:38,
        40:51,
        50:65,
        60:79,
        70:93,
        80:107,
        90:121,
        100:135
    },
    {
        50:66,
        100:132,
        200:264
    }],
    3:[None, 
    {
        50:32,
        56:100,
        100:66,
        200:132,
        400:264
    }]
}

class NRBaseStation(BaseStation):
    """
    Class representing a 5G New Radio (NR) base station in a wireless network.

    Attributes:
    - env: The environment in which the base station operates.
    - bs_id: Unique identifier of the base station.
    - position: Position of the base station.
    - carrier_frequency: Carrier frequency of the base station.
    - total_bandwidth: Total bandwidth allocated to the base station.
    - numerology: Numerology parameter specifying the time-frequency grid spacing.
    - max_data_rate: Maximum data rate supported by the base station.
    - antenna_power: Power transmitted by the base station antenna (in dBm).
    - antenna_gain: Gain of the base station antenna (in dBi).
    - feeder_loss: Loss in the feeder cables connecting the antenna to the transmitter (in dB).
    - pathloss: Path loss model used for computing signal attenuation.
    """
    
    def __init__(self, env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None, channels=None):
        """
        Initialize the NRBaseStation object.

        Args:
        - env: The environment in which the base station operates.
        - bs_id: Unique identifier of the base station.
        - position: Position of the base station.
        - carrier_frequency: Carrier frequency of the base station.
        - total_bandwidth: Total bandwidth allocated to the base station.
        - numerology: Numerology parameter specifying the time-frequency grid spacing.
        - max_data_rate: Maximum data rate supported by the base station.
        - antenna_power: Power transmitted by the base station antenna (in dBm).
        - antenna_gain: Gain of the base station antenna (in dBi).
        - feeder_loss: Loss in the feeder cables connecting the antenna to the transmitter (in dB).
        - pathloss: Path loss model used for computing signal attenuation.
        """
        if numerology not in NRbandwidth_prb_lookup:
            raise Exception("Invalid numerology for Base Station "+str(bs_id))
        if carrier_frequency >= 410 and carrier_frequency <=7125: # MHz
            self.fr = 0 # Frequency Range 1 (sub 6GHz)
        elif carrier_frequency >= 24250 and carrier_frequency <= 52600: #MHz
            self.fr = 1 # Frequency Range 2 (24.25-52.6GHz)
        else:
            raise Exception("Invalid carirer frequency for Base Station "+str(bs_id))
        if total_bandwidth not in NRbandwidth_prb_lookup[numerology][self.fr]:
            raise Exception("Invalid total bandwith for Base Station "+str(bs_id))
        self.env = env
        self.bs_id = bs_id
        self.bs_type = "nr"
        self.position = position
        self.carrier_frequency = carrier_frequency
        self.total_bandwidth = total_bandwidth
        self.numerology = numerology
        self.antenna_power = antenna_power
        self.antenna_gain = antenna_gain
        self.feeder_loss = feeder_loss
        if pathloss == None:
            self.pathloss = costhata.CostHataPathLoss(costhata.EnvType.URBAN)
        else:
            self.pathloss = pathloss
        self.max_data_rate = max_data_rate

        self.total_prb = NRbandwidth_prb_lookup[self.numerology][self.fr][self.total_bandwidth] * (10 * 2**self.numerology) # 10*2^mu time slots in a time frame
        self.subcarrier_bandwidth = 15*(2**self.numerology) #KHz

        # allocation structures
        self.ue_pb_allocation = {}
        self.ue_data_rate_allocation = {}
        self.allocated_prb = 0
        self.allocated_data_rate = 0
        self.T = 10 #Length of moving average for array utilization
        self.resource_utilization_array = [0] * self.T
        self.resource_utilization_counter = 0
        self.load_history = []
        self.data_rate_history = []
        self.RBG_size = 2
        self.transmission_power = 40  # Transmission power [dBm]
        self.antenna_gain = 35  # Antenna gain [dBi]
        self.channel = Channel.get_closest_channel(channels, carrier_frequency)
        return
    
    def compute_rsrp(self, ue):
        return self.transmission_power + self.antenna_gain - self.pathloss.compute_path_loss(ue, self)

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
        """
        Compute the Reference Signal Received Power (RSRP) for a user equipment (UE) connected to the base station.

        Args:
        - ue: User equipment object for which RSRP is calculated.

        Returns:
        - RSRP: Reference Signal Received Power (in dBm).
        """
        subcarrier_power = 10*math.log10(self.antenna_power*1000 / (12*(self.total_prb/(10*2**self.numerology))))
        return subcarrier_power + self.antenna_gain -self.feeder_loss - self.pathloss.compute_path_loss(ue, self)

    def get_rbur(self):
        """
        Calculate the Resource Block Utilization Ratio (RBUR) for the base station.

        Returns:
        - RBUR: Average RB utilization ratio over a moving window of time.
        """
        return sum(self.resource_utilization_array)/(self.T*self.total_prb)

    def compute_sinr(self, rsrp):
        """
        Compute the Signal-to-Interference-plus-Noise Ratio (SINR) for the base station.

        Args:
        - rsrp: Dictionary containing RSRP values for neighboring base stations.

        Returns:
        - SINR: Signal-to-Interference-plus-Noise Ratio.
        """
        interference = 0
        for elem in rsrp:
            bs_i = self.env.bs_by_id(elem)
            if elem != self.bs_id and bs_i.get_carrier_frequency() == self.carrier_frequency:
                rbur_i = bs_i.get_rbur()
                interference += (10 ** (rsrp[elem]/10))*rbur_i
        thermal_noise = constants.Boltzmann*293.15*15*(2**self.numerology)*1000 # delta_F = 15*2^mu KHz each subcarrier since we are considering measurements at subcarrirer level (like RSRP)
        sinr = (10**(rsrp[self.bs_id]/10))/(thermal_noise + interference)
        logging.debug("BS %s -> SINR: %s", self.bs_id, str(10*math.log10(sinr)))
        return sinr
    
    def compute_prb_NR(self, data_rate, rsrp):
        """
        Compute the number of Physical Resource Blocks (PRBs) required to achieve a given data rate.

        Args:
        - data_rate: Target data rate (in Mbps).
        - rsrp: Dictionary containing RSRP values for neighboring base stations.

        Returns:
        - n_prb: Number of PRBs required to achieve the target data rate.
        - r: Data rate achieved per PRB (in Mbps).
        """
        sinr = self.compute_sinr(rsrp)
        r = 12*self.subcarrier_bandwidth*1e3*math.log2(1+sinr)*(1/(10*(2**self.numerology))) # if a single RB is allocated we transmit for 1/(10*2^mu) seconds each second in 12*15*2^mu KHz bandwidth
        n_prb= math.ceil(data_rate*1e6/r) # the data-rate is in Mbps, so we had to convert it
        return n_prb, r/1e6

    def connect(self, ue_id, desired_data_rate, rsrp):
        # compute the number of PRBs needed for the requested data-rate,
        # then allocate them as much as possible
        # print("CONNECTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        n_prb, r = self.compute_prb_NR(desired_data_rate, rsrp)

        if self.max_data_rate != None:
            if self.max_data_rate - self.allocated_data_rate < r*n_prb:
                data_rate = self.max_data_rate - self.allocated_data_rate
                if data_rate < 0:
                    data_rate = 0 # due to computational errors
                n_prb, r = self.compute_prb_NR(data_rate, rsrp)

        if self.total_prb - self.allocated_prb < n_prb:
            n_prb = self.total_prb - self.allocated_prb
        
        if MAX_PRB != -1 and n_prb > MAX_PRB and self.get_usage_ratio() > 0.8:
            n_prb = MAX_PRB
        
        if ue_id in self.ue_pb_allocation:
            try:
                self.allocated_prb -= self.ue_pb_allocation[ue_id]
                self.ue_pb_allocation[ue_id] = n_prb
                self.allocated_prb += n_prb 
            except Exception as e:
                print(f"{ue_id}: \n{self.ue_pb_allocation}")
                print(e)

        if ue_id in self.ue_data_rate_allocation:
            self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
        self.ue_data_rate_allocation[ue_id] = n_prb*r
        self.allocated_data_rate += n_prb*r 
        return r*n_prb


    def disconnect(self, ue_id):
        print(f"disconnect {self.ue_pb_allocation}")
        try:
            self.allocated_prb -= self.ue_pb_allocation[ue_id]
            self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
            self.allocated_prb = 0
            self.allocated_data_rate = 0
            del self.ue_data_rate_allocation[ue_id]
            del self.ue_pb_allocation[ue_id]
        except Exception as e:
            print(f"{ue_id}: {self.ue_pb_allocation}")
            print(e)
        return
    
    def update_connection(self, ue_id, desired_data_rate, rsrp):
        # this can be called if desired_data_rate is changed or if the rsrp is changed
        # compute the number of PRBs needed for the requested data-rate,
        # then allocate them as much as possible
        #self.allocated_prb -= self.ue_pb_allocation[ue_id]
        #self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
        self.disconnect(ue_id)
        return self.connect(ue_id, desired_data_rate, rsrp)

    def step(self):
        self.resource_utilization_array[self.resource_utilization_counter] = self.allocated_prb
        self.resource_utilization_counter += 1
        if self.resource_utilization_counter % self.T == 0:
            self.resource_utilization_counter = 0

        self.load_history.append(self.get_usage_ratio())
        self.data_rate_history.append(self.allocated_data_rate)

        # print("RESETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        self.reset_condition()
        if (self.reset_condition()):
            # print("RESETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            self.disconnect_all()

    def get_allocated_data_rate(self):
        return self.allocated_data_rate

    def get_data_rate(self, ue_id):
        """
        Calculate the data rate for a specific user equipment (UE) connected to the base station.

        Args:
        - ue_id: ID of the UE for which data rate is calculated.

        Returns:
        - Data rate achieved by the UE (in Mbps).
        """
        interference = 0
        # current_bs_ue_id = list(self.ue_pb_allocation.keys())
        if ue_id in self.ue_pb_allocation:
            ue = self.env.ue_by_id(ue_id)
            rsrp = self.compute_rsrp(ue)
            bs_i = self.env.bs_by_id(self.bs_id)

            rbur_i = bs_i.get_rbur()
            print("RBURRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print(rbur_i)
            interference += (10 ** (rsrp / 10)) * rbur_i
            thermal_noise = constants.Boltzmann * 293.15 * 15 * (
                        2 ** self.numerology) * 1000  # delta_F = 15*2^mu KHz each subcarrier since we are considering measurements at subcarrirer level (like RSRP)
            sinr = (10 ** (rsrp / 10)) / (thermal_noise + interference)
            # sinr = self.compute_sinr(rsrp)
            r = 12 * self.subcarrier_bandwidth * math.log2(1 + sinr) * (1 / ((2 ** self.numerology)))
            alloc_prb = self.ue_pb_allocation[ue_id]
            return r

        return

    def get_buffer_reduction(self, ue_id, nPRG):
        """
        Calculate the reduction in buffer size for a UE due to the allocation of additional Physical Resource Blocks (PRBs).

        Args:
        - ue_id: ID of the UE for which buffer reduction is calculated.
        - nPRG: Number of additional PRGs allocated to the UE.

        Returns:
        - Buffer reduction: Reduction in buffer size (in bits).
        """
        if ue_id in self.ue_pb_allocation:
            ue = self.env.ue_by_id(ue_id)
            rsrp = ue.measure_rsrp()
            print(rsrp)
            sinr = self.compute_sinr(rsrp)
            print(sinr)
            # buffer_reduction = nPRG * 12 * self.subcarrier_bandwidth * 1e3 * math.log2(1 + sinr) * (
            #             1 / (10 * (2 ** self.numerology)))
            buffer_reduction = nPRG * 12 * self.subcarrier_bandwidth * math.log2(1 + sinr) * (
                    1 / ((2 ** self.numerology)))

        else:
            buffer_reduction = 0
        return buffer_reduction

    # Implementazione nuova funzione di schedule per il nostro algoritmo. La funzione è necessaria per implementare il nostro algoritmo. L'algoritmo,
    # per ogni TTI decide quale user schedulare. Gli step della funzione sono:
    #
    # 1 controllo che l'utente sia connesso alla base station
    # 2 Verifica che il resource block sia allocabile nel frame attuale
    # 3 Alloca la risorsa effettivamente
    # 4 Se la risorsa non è allocabile ritorna 0

    def schedule(self, ue_id, nRBG):

        current_bs_ue_id = list(self.ue_pb_allocation.keys())

        if (ue_id in current_bs_ue_id):
            # print("SCHEDULEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print(self.total_prb)
            print(self.allocated_prb)
            # print("SCHEDULEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            if self.total_prb - self.allocated_prb >= nRBG:

                self.allocated_prb = nRBG + self.allocated_prb
                ris = 1

                # if ue_id in self.ue_pb_allocation:
                # self.allocated_prb -= self.ue_pb_allocation[ue_id]
                self.ue_pb_allocation[ue_id] += nRBG
                self.allocated_prb += nRBG

                if ue_id in self.ue_data_rate_allocation:
                    self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
                r = self.get_data_rate(ue_id)
                # self.ue_data_rate_allocation[ue_id] += nRBG * r
                self.allocated_data_rate += nRBG * r
                self.step()
            else:
                ris = 0
                self.step()
        else:
            ris = 0
            self.step()

        return ris

    def reset_condition(self):
        if self.total_prb - self.allocated_prb >= self.RBG_size:
            ris = 0
        else:
            ris = 1

        return ris

    def disconnect_all(self):
        current_bs_ue_id = list(self.ue_pb_allocation.keys())
        for ue_id in current_bs_ue_id:
            self.disconnect(ue_id)
