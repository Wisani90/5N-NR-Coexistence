# simulate.py
import json
import logging
import numpy.random as random
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann


from wns2.basestation.nrbasestation import NRBaseStation
from wns2.environment.cband_environment import Environment
from wns2.environment.channel import Channel
from wns2.renderer.renderer import CustomRenderer
from wns2.renderer.renderer_json import JSONRendererARIES
from wns2.userequipment.multipath_userequipment import MultiPathUserEquipment
from wns2.userequipment.userequipment import UserEquipment as UE
from wns2.basestation.vsatbasestation import VSATBaseStation
from wns2.basestation.generic import BaseStation
from wns2.pathloss.cband_freespace import FreeSpacePathLoss
import wns2.environment.environment

# logger = logging.getLogger()
# logger.setLevel(level=logging.INFO)
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


channel1 = Channel(frequency=3810)
channel2 = Channel(frequency=3850)
channel3 = Channel(frequency=3900)
channel4 = Channel(frequency=3980)
channel5 = Channel(frequency=4040)
channel6 = Channel(frequency=4100)

bs_parm =[{"pos": (500, 500, 30),
    "freq": 3600,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 20,
    # 'beamwidth': 65,     # degrees
    "max_bitrate": 1000},
    
    #BS2
    {"pos": (250, 300, 30),
    "freq": 3800,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    # 'beamwidth': 65,     # degrees
    "max_bitrate": 1000},
    
    #BS3
    {"pos": (780, 810, 30),
    "freq": 4000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    # 'beamwidth': 65,     # degrees
    "max_bitrate": 1000},
    
    #BS4
    {"pos": (900, 900, 50),
    "freq": 4000,
    "numerology": 1, 
    "power": 40,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    # 'beamwidth': 65,     # degrees
    "max_bitrate": 1000}]
        

def place_base_stations(env, base_station_config, channels):
    positions = [(5, 5), (500, 500), (1000, 1000)]
    
    for i in range(1, len(bs_parm)):
        env.add_base_station(
            NRBaseStation(env, f'5G_NR_{i}', 
                          bs_parm[i]["pos"], bs_parm[i]["freq"], 
                          bs_parm[i]["bandwidth"], bs_parm[i]["numerology"], 
                          bs_parm[i]["max_bitrate"], bs_parm[i]["power"], 
                          bs_parm[i]["gain"], bs_parm[i]["loss"], 
                          channels=channels
                          )
            )

    # for i, pos in enumerate(positions):
    #     env.add_base_station(NRBaseStation(env, f'5G_NR_{i+1}', pos))

def place_vsat_systems(env, vsat_config, channels):
    positions = [(800, 800, 10), (800, 600, 15), (900, 700, 20)]
    for i, pos in enumerate(positions):
        env.add_base_station(
            VSATBaseStation(env, f'VSAT_{i+1}', pos, channels=channels))

def simulate_interference(env, duration, cells, channels):
    all_data = []

    for _ in range(duration):
        env.step()
        data = collect_data(env, cells, channels)
        all_data.append(data)
        
        # Optionally log data
        # logging.info(json.dumps(data))
    
    return all_data

def generate_sinr_maps(env, grid_size=100):
    """
    Generate SINR maps for the simulation environment.
    
    Args:
    - env: The simulation environment containing base stations and VSAT systems.
    - grid_size: The size of the grid to calculate SINR values.
    
    Returns:
    - sinr_map: A 2D array representing the SINR values across the grid.
    """
    
    # Define the grid boundaries
    x_min, y_min = 0, 0
    x_max, y_max = 1000, 1000  # Assuming a 1000x1000 area for simplicity
    
    # Create the grid
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Initialize SINR map
    sinr_map = np.zeros((grid_size, grid_size))
    
    # Calculate SINR for each point in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            sinr_map[i, j] = calculate_sinr_at_point(env, xx[i, j], yy[i, j])
    
    # Plot SINR map
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, sinr_map, levels=100, cmap='jet')
    plt.colorbar(label='SINR (dB)')
    plt.title('SINR Map')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.show()
    
    return sinr_map

def generate_coverage_maps(environment, grid_size=100, frequency_band="C-band"):
    """
    Generate coverage maps for all base stations and VSATs in the environment.

    Args:
    - environment: The simulation environment containing base stations and VSATs.
    - grid_size: The number of points along each axis of the grid.
    - frequency_band: The frequency band for the coverage map.

    Returns:
    - coverage_maps: A dictionary containing the coverage maps for each base station and VSAT.
    """
    x_min, x_max, y_min, y_max = environment.get_bounds()
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    coverage_maps = {}
    
    for bs in environment.base_stations + environment.vsats:
        coverage_map = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                distance = np.sqrt((bs.position[0] - X[i, j])**2 + (bs.position[1] - Y[i, j])**2)
                path_loss_model = FreeSpacePathLoss()
                path_loss = path_loss_model.compute_path_loss(distance, bs.carrier_frequency * 1e9)  # Convert GHz to Hz
                received_power = bs.transmission_power - path_loss
                coverage_map[i, j] = received_power
        
        coverage_maps[bs.get_id()] = coverage_map
    
    # Plot the coverage maps
    for bs_id, coverage_map in coverage_maps.items():
        plt.figure()
        plt.contourf(X, Y, coverage_map, cmap='viridis')
        plt.colorbar(label='Received Power (dBm)')
        plt.title(f'Coverage Map for Base Station/VSAT {bs_id} in {frequency_band}')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.show()
    
    return coverage_maps

def generate_interference_heatmaps(env):
    # Implement interference heatmap generation logic
    pass

def collect_data(env, all_cells, all_channels):
    data = {
        'step': env.current_step,
        'base_stations': [],
        'users': []
    }
    
    for bs in env.base_stations:
        bs_data = {
            'id': bs.get_id(),
            'position': bs.get_position(),
            'allocated_data_rate': bs.get_allocated_data_rate(),
            'usage_ratio': bs.get_usage_ratio()
        }
        data['base_stations'].append(bs_data)
    
    for ue in env.users:
        neighboring_cells = ue.get_neighboring_cells(all_cells, max_distance=15.0)
        adjacent_channels = ue.get_neighboring_channels(all_channels, max_frequency_distance=20.0)
        ue_data = {
            'id': ue.get_id(),
            'position': ue.get_position(),
            'connected_bs': ue.get_current_bs(),
            'received_signal_strength': ue.measure_rsrp(),
            'sinr': ue.calculate_sinr(),
            'interference': ue.calculate_interference(neighboring_cells, adjacent_channels)
        }
        data['users'].append(ue_data)
    
    return data

def analyze_data(all_data):
    # Initialize lists to store metrics
    sinr_values = []
    throughput_values = []
    interference_values = []

    # Extract data from each time step
    for data in all_data:
        for ue in data['users']:
            sinr_values.append(ue['sinr'])
            throughput_values.append(ue['received_signal_strength'])  # Assuming throughput is tracked as received_signal_strength
            interference_values.append(ue['interference'])  # Assuming interference is tracked as 'interference'

    # Calculate average metrics
    avg_sinr = np.mean(sinr_values)
    # avg_throughput = np.mean(throughput_values)
    formarted_throughput = [list(d.values())[0] for d in throughput_values]
    avg_throughput = np.mean(formarted_throughput)
    avg_interference = np.mean(interference_values)

    # Log average metrics
    logging.info(f'Average SINR over the simulation: {avg_sinr:.2f} dB')
    logging.info(f'Average Throughput over the simulation: {avg_throughput:.2f} Mbps')
    logging.info(f'Average Interference over the simulation: {avg_interference:.2f} dB')

    # Print average metrics
    print(f'Average SINR over the simulation: {avg_sinr:.2f} dB')
    print(f'Average Throughput over the simulation: {avg_throughput:.2f} Mbps')
    print(f'Average Interference over the simulation: {avg_interference:.2f} dB')

    # Plot metrics
    plt.figure(figsize=(15, 5))

    # Plot SINR
    plt.subplot(1, 3, 1)
    plt.plot(sinr_values, label='SINR')
    plt.xlabel('Time Step')
    plt.ylabel('SINR (dB)')
    plt.title('SINR Over Time')
    plt.legend()

    # Plot Throughput
    plt.subplot(1, 3, 2)
    plt.plot(formarted_throughput, label='Throughput')
    plt.xlabel('Time Step')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Throughput Over Time')
    plt.legend()

    # Plot Interference
    plt.subplot(1, 3, 3)
    plt.plot(interference_values, label='Interference')
    plt.xlabel('Time Step')
    plt.ylabel('Interference (dB)')
    plt.title('Interference Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_sinr_at_point(env, x, y):
    """
    Calculate the SINR at a specific point in the grid.
    
    Args:
    - env: The simulation environment containing base stations and VSAT systems.
    - x: X coordinate of the point.
    - y: Y coordinate of the point.
    
    Returns:
    - sinr: The calculated SINR value at the point.
    """
    
    # Calculate the received signal power from each base station
    received_powers = []
    for bs in env.base_stations:
        received_power = calculate_received_power(bs, x, y)
        received_powers.append(received_power)
    
    # Calculate the interference power from all other base stations
    interference_power = 0
    for bs in env.base_stations:
        interference_power += calculate_interference_power(bs, x, y)
    
    # Calculate thermal noise
    noise_power = Boltzmann * 293.15 * bs.subcarrier_bandwidth * 1e6  # Assuming a temperature of 293.15 K
    
    # Calculate SINR
    signal_power = max(received_powers)
    sinr = signal_power / (interference_power + noise_power)
    
    return 10 * np.log10(sinr)

def calculate_received_power(bs, x, y):
    """
    Calculate the received power from a base station at a specific point.
    
    Args:
    - bs: The base station.
    - x: X coordinate of the point.
    - y: Y coordinate of the point.
    
    Returns:
    - received_power: The received power at the point.
    """
    distance = np.sqrt((bs.position[0] - x)**2 + (bs.position[1] - y)**2)
    path_loss = FreeSpacePathLoss().compute_path_loss(distance, bs.carrier_frequency)
    received_power = bs.transmission_power - path_loss
    return 10 ** (received_power / 10)  # Convert dBm to linear scale

def calculate_interference_power(bs, x, y):
    """
    Calculate the interference power from a base station at a specific point.
    
    Args:
    - bs: The base station.
    - x: X coordinate of the point.
    - y: Y coordinate of the point.
    
    Returns:
    - interference_power: The interference power at the point.
    """
    distance = np.sqrt((bs.position[0] - x)**2 + (bs.position[1] - y)**2)
    path_loss = FreeSpacePathLoss().compute_path_loss(distance, bs.carrier_frequency)
    interference_power = bs.transmission_power - path_loss
    return 10 ** (interference_power / 10)  # Convert dBm to linear scale


def generate_interference_heatmaps(environment, grid_size=100, frequency_band="C-band"):
    """
    Generate interference heatmaps for all base stations and VSATs in the environment.

    Args:
    - environment: The simulation environment containing base stations and VSATs.
    - grid_size: The number of points along each axis of the grid.
    - frequency_band: The frequency band for the interference heatmap.

    Returns:
    - interference_heatmaps: A dictionary containing the interference heatmaps for each base station and VSAT.
    """
    x_min, x_max, y_min, y_max = environment.get_bounds()
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    interference_heatmaps = {}
    
    for target_bs in environment.base_stations + environment.vsats:
        interference_heatmap = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                distance_to_target = np.sqrt((target_bs.position[0] - X[i, j])**2 + (target_bs.position[1] - Y[i, j])**2)
                path_loss_model = FreeSpacePathLoss()
                target_path_loss = path_loss_model.compute_path_loss(distance_to_target, target_bs.carrier_frequency * 1e9)  # Convert GHz to Hz
                target_received_power = target_bs.transmission_power - target_path_loss

                # Calculate interference and noise
                interference = 0
                for interferer in environment.base_stations + environment.vsats:
                    if interferer != target_bs:
                        distance_to_interferer = np.sqrt((interferer.position[0] - X[i, j])**2 + (interferer.position[1] - Y[i, j])**2)
                        interferer_path_loss = path_loss_model.compute_path_loss(distance_to_interferer, interferer.carrier_frequency * 1e9)
                        interferer_received_power = interferer.transmission_power - interferer_path_loss
                        interference += 10 ** (interferer_received_power / 10)
                
                # Calculate thermal noise
                thermal_noise = Boltzmann * 293.15 * target_bs.subcarrier_bandwidth * 1e6
                total_noise_interference = thermal_noise + interference
                sinr = target_received_power - 10 * np.log10(total_noise_interference)
                
                interference_heatmap[i, j] = sinr
        
        interference_heatmaps[target_bs.get_id()] = interference_heatmap
    
    # Plot the interference heatmaps
    for bs_id, interference_heatmap in interference_heatmaps.items():
        plt.figure()
        plt.contourf(X, Y, interference_heatmap, cmap='viridis')
        plt.colorbar(label='SINR (dB)')
        plt.title(f'Interference Heatmap for Base Station/VSAT {bs_id} in {frequency_band}')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.show()
    
    return interference_heatmaps


def main():
    x_lim = 1000
    y_lim = 1000
    all_channels = [channel1, channel2, channel3, channel4, channel5, channel6]
    env = Environment()
    # wns2.environment.environment.MIN_RSRP = -75
    
    base_station_config = {
        'antenna_gain': 15,  # dBi
        'beamwidth': 65,     # degrees
        'transmission_power': 40  # dBm
    }
    
    vsat_config = {
        'antenna_gain': 35,  # dBi
        'beamwidth': 3,      # degrees
        'transmission_power': 30  # dBm
    }
    
    place_base_stations(env, base_station_config, all_channels)
    place_vsat_systems(env, vsat_config, all_channels)
    
    for i in range(0, 5):
        pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
        user_object = UE(env, i, 25, pos, speed = 0, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15)
        env.users.append(user_object)
        # env.add_user()

    
    simulation_duration = 100  # Example duration
    all_data = simulate_interference(env, simulation_duration, env.base_stations, all_channels)
    
    # Generate and analyze SINR maps
    sinr_map = generate_sinr_maps(env)
    
    # Assuming environment is an instance of your simulation environment with base stations and VSATs
    coverage_maps = generate_coverage_maps(env, grid_size=100, frequency_band="3.8-4.2 GHz")
    
    # Assuming environment is an instance of your simulation environment with base stations and VSATs
    interference_heatmaps = generate_interference_heatmaps(env, grid_size=100, frequency_band="3.8-4.2 GHz")

    # Perform post-simulation analysis
    analyze_data(all_data)

if __name__ == '__main__':
    main()
