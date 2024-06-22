import random
import math
import logging
import numpy.random


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class UserEquipment:
    """
    Class representing a User Equipment (UE) in a wireless network.

    Attributes:
    - env: The environment in which the UE operates.
    - ue_id: Unique identifier of the UE.
    - data_rate: Initial data rate of the UE.
    - current_position: Current position of the UE.
    - speed: Speed of the UE.
    - direction: Direction of movement of the UE.
    - random: Flag indicating whether UE movement is random or linear.
    - _lambda_c: Poisson arrival rate for connection events.
    - _lambda_d: Poisson arrival rate for disconnection events.
    """
    
    def __init__(self, env, ue_id, initial_data_rate, starting_position, speed = 0, direction = 0, random = False, _lambda_c = None, _lambda_d = None):
        """
        Initialize the User Equipment object.

        Args:
        - env: The environment in which the UE operates.
        - ue_id: Unique identifier of the UE.
        - initial_data_rate: Initial data rate of the UE.
        - starting_position: Starting position of the UE.
        - speed: Speed of the UE.
        - direction: Direction of movement of the UE.
        - random: Flag indicating whether UE movement is random or linear.
        - _lambda_c: Poisson arrival rate for connection events.
        - _lambda_d: Poisson arrival rate for disconnection events.
        """
        self.ue_id = ue_id
        self.data_rate = initial_data_rate
        self.current_position = starting_position
        self.env = env
        self.speed = speed * self.env.get_sampling_time()
        self.direction = direction
        self._lambda_c = _lambda_c
        if self._lambda_c != None:
            self.connection_time_to_wait = numpy.random.poisson(self._lambda_c)
        self._lambda_d = _lambda_d
        self.last_time = 0
        self.random = random

        self.sampling_time = self.env.get_sampling_time()

        self.bs_data_rate_allocation = {}
        self.buffer = 0
        self.fairness = 0
        self.connected_bs = None



    def get_position(self):
        """
        Get the current position of the UE.

        Returns:
        Tuple representing the current position (x, y, z) of the UE.
        """
        return self.current_position
    
    def get_id(self):
        return self.ue_id

    def move(self):
        if self.speed == 0:
            return
        if self.random == True:
            return self.random_move()
        else:
            return self.line_move()
    
    def random_move(self):
        """
        Move the UE randomly within the environment.

        Returns:
        Updated position of the UE.
        """
        val = random.randint(1, 4)
        size = random.randint(0, math.floor(self.speed*self.sampling_time))
        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()
        if val == 1: 
            if (self.current_position[0] + size) > 0 and (self.current_position[0] + size) < x_lim:
                self.current_position = (self.current_position[0] + size, self.current_position[1], self.current_position[2])
        elif val == 2: 
            if (self.current_position[0] - size) > 0 and (self.current_position[0] - size) < x_lim:
                self.current_position = (self.current_position[0] - size, self.current_position[1], self.current_position[2])
        elif val == 3: 
            if (self.current_position[1] + size) > 0 and (self.current_position[1] + size) < y_lim:
                self.current_position = (self.current_position[0], self.current_position[1] + size, self.current_position[2])
        else: 
            if (self.current_position[1] - size) > 0 and (self.current_position[1] - size) < y_lim:
                self.current_position = (self.current_position[0], self.current_position[1] - size, self.current_position[2])
        return self.current_position

    def line_move(self):
        """
        Move the UE in a linear trajectory within the environment.

        Returns:
        Updated position of the UE.
        """

        new_x = self.current_position[0]+self.speed*self.sampling_time*math.cos(math.radians(self.direction))
        new_y = self.current_position[1]+self.speed*self.sampling_time*math.sin(math.radians(self.direction))
        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()
        # bounce with the same incident angle if a sideo or a corner is reached
        if ((self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (self.current_position[1])/(self.current_position[0])) or (self.direction == 270)) :
            if new_x <= 0 and new_y <= 0:
                # bottom left corner bouncing
                self.direction = 270 - self.direction
                dist = math.sqrt((new_x)**2 + (new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = dist*math.sin(math.radians(self.direction))
        elif ((x_lim - self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (y_lim - self.current_position[1])/(x_lim - self.current_position[0])) or (self.direction == 90)) :
            if new_x >= x_lim and new_y >= y_lim :
                # top right corner bouncing
                self.direction = 270 - self.direction
                dist = math.sqrt((x_lim-new_x)**2 + (y_lim-new_y)**2)
                new_x = x_lim + dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif ((self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (y_lim - self.current_position[1])/(self.current_position[0])) or (self.direction == 90)) :
            if new_x <= 0 and new_y >= y_lim:
                # top left corner bouncing
                self.direction = 450 - self.direction
                dist = math.sqrt((new_x)**2 + (y_lim-new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif ((x_lim - self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (self.current_position[1])/(x_lim - self.current_position[0])) or (self.direction == 270)) :
            if new_x >= x_lim and new_y <= 0 :
                # bottom right corner bouncing
                self.direction = 450 - self.direction
                dist = math.sqrt((new_x)**2 + (y_lim-new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif new_y <= 0:
            # bottom side bouncing
            new_y = 0 - new_y
            self.direction = 360 - self.direction
            if new_x <= 0:
                # there is another bouncing on the left side
                new_x = 0 - new_x
                self.direction = 180 - self.direction
            elif new_x >= x_lim:
                # there is another bouncing on the right side
                new_x = 2*x_lim - new_x
                self.direction = 180 - self.direction
        elif new_x <= 0 :
            # left side bouncing
            new_x = 0 - new_x
            self.direction = 180 - self.direction
            if new_y <= 0:
                # there is another bouncing on the bottom side
                new_y = 0 - new_y
                self.direction = - self.direction
            elif new_y >= y_lim:
                # there is another bouncing on the top side
                new_y = 2*y_lim - new_y
                self.direction = - self.direction
        elif new_y >= y_lim:
            # top side bouncing
            new_y = 2*y_lim - new_y
            self.direction = 360 - self.direction
            if new_x <= 0:
                # there is another bouncing on the left side
                new_x = 0 - new_x
                self.direction = 180 - self.direction
            elif new_x >= x_lim:
                # there is another bouncing on the left side
                new_x = 2*x_lim - new_x
                self.direction = 180 - self.direction
        elif new_x >= x_lim:
            # right side bouncing
            new_x = 2*x_lim - new_x
            self.direction =  180 - self.direction
            if new_y <= 0:
                # there is another bouncing on the bottom side
                new_y = 0 - new_y
                self.direction = -self.direction
            elif new_y >= y_lim:
                # there is another bouncing on the top side
                new_y = 2*y_lim - new_y
                self.direction = - self.direction

        self.current_position = (new_x, new_y, self.current_position[2])
        self.direction = self.direction % 360
        return self.current_position
    
    def measure_rsrp(self):
        # measure RSRP together with the BS
        # the result is in dB
        return self.env.compute_rsrp(self)
    
    def get_current_bs(self):
        if len(self.bs_data_rate_allocation) == 0:
            return None
        else:
            return list(self.bs_data_rate_allocation.keys())[0]

    def advertise_connection(self):
        """
        Advertise connection to base station (BS) or decide to disconnect based on specified Poisson arrival rates.

        If no BS is currently connected:
        - If no connection rate (_lambda_c) is specified, advertise connection immediately.
        - If connection rate is specified and connection time has elapsed, advertise connection.
        - Otherwise, increment time counter.

        If a BS is already connected:
        - If no disconnection rate (_lambda_d) is specified, advertise connection each timestep.
        - If disconnection time has elapsed, disconnect from current BS.
        - Otherwise, increment time counter and advertise connection.

        If disconnection occurs, a new connection attempt may be scheduled based on the connection rate.
        """
        if len(self.bs_data_rate_allocation) == 0:
            # no BS connected, decide if it is time to connect
            if self._lambda_c == None:
                self.env.advertise_connection(self.ue_id)
            elif self.last_time >= self.connection_time_to_wait:
                self.last_time = 0
                self.env.advertise_connection(self.ue_id)
                if self._lambda_d != None:
                    self.disconnection_time_to_wait = numpy.random.poisson(self._lambda_d)
            else:
                self.last_time += 1
        else:
            if self._lambda_d == None:
                # ENABLE THIS LINE IF WE WANT TO RECONNECT EACH TIMESTEP
                self.env.advertise_connection(self.ue_id)
            elif self.last_time >= self.disconnection_time_to_wait:
                self.last_time = 0
                self.disconnect()
                if self._lambda_c != None:
                    self.connection_time_to_wait = numpy.random.poisson(self._lambda_c)
            else:
                self.last_time += 1
                # ENABLE THIS LINE IF WE WANT TO RECONNECT EACH TIMESTEP
                self.env.advertise_connection(self.ue_id)
    
    def step(self, substep = False):
        if not substep:
            self.move()
            self.advertise_connection()
        return
    
    def connect_bs(self, bs):
        rsrp = self.measure_rsrp()
        if len(rsrp) == 0:
            return None
        if bs not in rsrp:
            return None

        return self.connect_(bs, rsrp)


    def connect_max_rsrp(self):
        rsrp = self.measure_rsrp()
        if len(rsrp) == 0:
            return
        best_bs = None
        max_rsrp = -200
        for elem in rsrp:
            if rsrp[elem] > max_rsrp:
                best_bs = elem
                max_rsrp = rsrp[elem]
        return self.connect_(best_bs, rsrp)

    def connect_(self, bs, rsrp):
        actual_data_rate = None
        if len(self.bs_data_rate_allocation) == 0:
            # no BS connected
            bs = self.env.bs_by_id(bs)
            actual_data_rate = bs.connect(self.ue_id, self.data_rate, rsrp)
            self.bs_data_rate_allocation[bs.get_id()] = actual_data_rate
            logging.info("UE %s connected to BS %s with data rate %s", self.ue_id, bs.get_id(), actual_data_rate)
        else:
            current_bs = self.get_current_bs()
            if current_bs != bs:
                self.disconnect()
                bs = self.env.bs_by_id(bs)
                actual_data_rate = bs.connect(self.ue_id, self.data_rate, rsrp)
                self.bs_data_rate_allocation[bs.get_id()] = actual_data_rate
                logging.info("UE %s switched to BS %s with data rate %s", self.ue_id, bs.get_id(), actual_data_rate)
            else:
                current_bs = self.env.bs_by_id(current_bs)
                actual_data_rate = current_bs.update_connection(self.ue_id, self.data_rate, rsrp)
                logging.info("UE %s updated to BS %s with data rate %s --> %s", self.ue_id, current_bs.get_id(), self.bs_data_rate_allocation[current_bs.get_id()], actual_data_rate)
                self.bs_data_rate_allocation[current_bs.get_id()] = actual_data_rate
            self.connected_bs = current_bs
        return actual_data_rate

    def disconnect(self):
        current_bs = self.get_current_bs()
        if current_bs != None:
            self.env.bs_by_id(current_bs).disconnect(self.ue_id) 
            del self.bs_data_rate_allocation[current_bs]
            current_bs = None
            self.connected_bs = None
    
    def requested_disconnect(self):
        # this is called if the env or the BS requested a disconnection
        current_bs = self.get_current_bs()
        del self.bs_data_rate_allocation[current_bs]
        current_bs = None

    def compute_data_rate(self):
        ue_id = self.ue_id
        environment = self.env
        return

    def get_buffer_size(self):
        return self.buffer

    def feed_buffer(self, buffer_size):
        self.buffer = buffer_size

    def reduce_buffer(self, data):
        if(self.buffer - data < 0):
            self.buffer = 0
        else:
            self.buffer = self.buffer - data

    def isEligible(self):
        if self.buffer > 0:
            return True
        else:
            return False

    def update_fairness(self, selected):
        if selected == 1:
            self.fairness = max(self.fairness - 1, 0)
        else:
            if (self.buffer != 0):
                self.fairness = self.fairness + 1

    def calculate_sinr(self, 
                       received_signal_power=10.0, 
                       interference_power=5.0, 
                       noise_power=2.0):
        if self.get_current_bs() is None:
            # raise ValueError("UE is not connected to any base station.")
            return 0
        # Calculate SINR
        sinr = received_signal_power / (interference_power + noise_power)
        
        return sinr
    
    def get_sinr(self):
        return self.sinr

    def calculate_interference(self, neighboring_cells, adjacent_channels):
        if self.get_current_bs() is None:
            # raise ValueError("UE is not connected to any base station.")
            return 0

        interference_power = 0.0

        # Calculate interference power from neighboring cells
        for cell in neighboring_cells:
            if cell != self.get_current_bs():
                # Assume each neighboring cell contributes some interference power
                # Example: interference calculation based on distance, path loss, etc.
                # Replace with actual interference calculation based on your model
                distance = self.calculate_distance(self.connected_bs, cell)
                path_loss = self.calculate_path_loss(distance)
                interference_power += cell.transmit_power / path_loss  # Example formula

        # Calculate interference power from adjacent channels
        for channel in adjacent_channels:
            if channel != self.connected_bs.channel:
                # Assume each adjacent channel contributes some interference power
                # Example: interference calculation based on frequency separation, etc.
                # Replace with actual interference calculation based on your model
                interference_power += channel.transmit_power / self.calculate_frequency_distance(channel)

        return interference_power
    
    def calculate_distance(self, bs1, bs2):
        distance = 0
        if bs1 and bs2:
            # Calculate Euclidean distance between two points in 3D space
            dx = bs1.position[0] - bs2.position[0]
            dy = bs1.position[1] - bs2.position[1]
            dz = bs1.position[2] - bs2.position[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            
        return distance

    def calculate_frequency_distance(self, channel):
        # Example method to calculate frequency distance between channels
        return abs(self.connected_bs.frequency - channel.frequency)  # Replace with actual calculation

    def get_neighboring_cells(self, all_cells, max_distance):
        neighboring_cells = []
        for cell in all_cells:
            if cell != self.connected_bs:
                distance = self.calculate_distance(self.connected_bs, cell)
                if distance <= max_distance:
                    neighboring_cells.append(cell)
        return neighboring_cells
    
    def get_neighboring_channels(self, all_channels, max_frequency_distance):
        neighboring_channels = []
        if self.connected_bs:
            for channel in all_channels:
                if channel != self.connected_bs.channel:
                    frequency_distance = self.calculate_frequency_distance(channel)
                    if frequency_distance <= max_frequency_distance:
                        neighboring_channels.append(channel)
        return neighboring_channels