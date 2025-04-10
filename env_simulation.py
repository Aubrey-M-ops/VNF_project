import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# define x, y coordinate range (san francisco)
SF_X_MIN, SF_X_MAX = -122.5, -122.3
SF_Y_MIN, SF_Y_MAX = 37.7, 37.8

# define parameters
NUM_BASE_STATIONS = 50     # base station number
NUM_TRAJECTORIES = 610000  # user trajectory number
START_DATE = datetime(2023, 1, 1)  # start date for trajectory generation

#####################Simulation Tool Functions###########################

# generate base stations []
def generate_base_stations(num_stations):
    base_stations = []
    for i in range(num_stations):
        # generate random x coordinate
        x = np.random.uniform(SF_X_MIN, SF_X_MAX)
        # generate random y coordinate
        y = np.random.uniform(SF_Y_MIN, SF_Y_MAX)
        base_stations.append((x, y))
    return base_stations

# generate one user trajectory => {id, timestamp, start_point, end_point}
def generate_one_trajectory(base_stations, trajectory_id, start_time):
    # near a base station? => randomly choose
    is_trajectory_near_station = random.choice([True, False])
    if is_trajectory_near_station and base_stations:
        base = random.choice(base_stations)
        # set user start point near a base station
        start_x = base[0] + np.random.normal(0, 0.005)
        start_y = base[1] + np.random.normal(0, 0.005)
    else:
        start_x = np.random.uniform(SF_X_MIN, SF_X_MAX)
        start_y = np.random.uniform(SF_Y_MIN, SF_Y_MAX)

    # simulate moving distance（0.001 ≈ 100 meters，0.05 ≈ 5 kilometers）
    move_distance = np.random.uniform(0.001, 0.05)
    move_angle = np.random.uniform(0, 2 * np.pi)
    end_x = start_x + move_distance * np.cos(move_angle)
    end_y = start_y + move_distance * np.sin(move_angle)

    # make sure the end point is within the range
    end_x = np.clip(end_x, SF_X_MIN, SF_X_MAX)
    end_y = np.clip(end_y, SF_Y_MIN, SF_Y_MAX)

    # generate random timestamp: start_time + random seconds
    time_offset = timedelta(seconds=np.random.randint(0, 24 * 60 * 60))
    timestamp = start_time + time_offset

    return {
        'id': trajectory_id,
        'timestamp': timestamp,
        'start_point': f'POINT({start_x} {start_y})',
        'end_point': f'POINT({end_x} {end_y})'
    }

# generate dataset
def generate_dataset(num_trajectories, num_base_stations):
    print('👉 Generating base stations...')
    base_stations = generate_base_stations(num_base_stations)
    print(f'👉 {num_base_stations} base stations generated!')

    print('👉 Generating user trajectories...')
    trajectories = []
    for i in range(num_trajectories):
        traj = generate_one_trajectory(base_stations, i, START_DATE)
        trajectories.append(traj)
        # print progress every 100000 trajectories
        if (i + 1) % 100000 == 0:
            print(f'👉 {i + 1} trajectories generated...')

    # trasform to DataFrame
    user_movements = pd.DataFrame(trajectories)
    return user_movements, base_stations


############################ Main Function ############################
print('👉 Starting dataset generation...')
user_movements, base_stations = generate_dataset(NUM_TRAJECTORIES, NUM_BASE_STATIONS)
print('👉 Dataset generation completed!')

# save user movements to CSV
output_path = './datasets/simulated_sf_users.csv'
user_movements.to_csv(output_path, index=False)
print(f'👉 User trajectories saved to {output_path}')

# save base stations to CSV
base_station_df = pd.DataFrame(base_stations, columns=['x', 'y'])
base_station_path = './datasets/simulated_base_stations.csv'
base_station_df.to_csv(base_station_path, index=False)
print(f'👉 Base stations saved to {base_station_path}')

# show sample of generated dataset
print('\n👉 Sample of generated dataset:')
print(user_movements.head())
