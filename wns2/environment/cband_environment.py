import numpy as np

from wns2.environment.environment import MIN_RSRP

class Environment:
    def __init__(self, sampling_time=100):
        self.base_stations = []
        self.vsats = []
        # self.h = h
        # self.l = l
        self.ue_list = {}
        self.users = [] 
        self.connection_advertisement = []
        self.sampling_time = sampling_time # in seconds
        self.current_step = 0  # Initialize current_step attribute

    def add_base_station(self, bs):
        self.base_stations.append(bs)

    def add_vsat(self, vsat):
        self.vsats.append(vsat)
        
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
            if self.ue_list[ue_id].get_current_bs() != None:
                bs = self.ue_list[ue_id].get_current_bs()
                self.ue_list[ue_id].disconnect(bs)
            del self.ue_list[ue_id]

    def get_bounds(self):
        all_positions = [bs.position for bs in self.base_stations] + [vsat.position for vsat in self.vsats]
        x_positions, y_positions, z_positions = zip(*all_positions)
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        return x_min, x_max, y_min, y_max

    def get_base_stations(self):
        return self.base_stations

    def get_vsats(self):
        return self.vsats

    def bs_by_id(self, bs_id):
        for bs in self.base_stations:
            if bs.get_id() == bs_id:
                return bs
        for vsat in self.vsats:
            if vsat.get_id() == bs_id:
                return vsat
        return None

    def ue_by_id(self, id):
        return self.ue_list[id]

    def step(self):
        for ue in self.users:
            #self.ue_list[ue].step(substep)
            ue.step()
        for bs in self.base_stations:
            bs.step()
        for vsat in self.vsats:
            vsat.step()
        self.current_step += 1  # Increment current_step attribute

    def collect_data(self):
        data = {
            'base_stations': [],
            'vsats': [],
        }
        for bs in self.base_stations:
            data['base_stations'].append({
                'id': bs.get_id(),
                'position': bs.get_position(),
                'allocated_data_rate': bs.get_allocated_data_rate(),
                'usage_ratio': bs.get_usage_ratio(),
            })
        for vsat in self.vsats:
            data['vsats'].append({
                'id': vsat.get_id(),
                'position': vsat.get_position(),
                'allocated_data_rate': vsat.get_allocated_data_rate(),
                'usage_ratio': vsat.get_usage_ratio(),
            })
        return data
    
    def compute_rsrp(self, ue):
        rsrp = {}
        bs_count = 0
        for bs in self.base_stations:
            rsrp_i = bs.compute_rsrp(ue)
            if rsrp_i > MIN_RSRP or bs.get_bs_type() == "sat":
                rsrp[bs_count] = rsrp_i
            bs_count += 1
        return rsrp
    
    def advertise_connection(self, ue_id):
        self.connection_advertisement.append(ue_id)
        return
    
    def get_sampling_time(self):
        return self.sampling_time
