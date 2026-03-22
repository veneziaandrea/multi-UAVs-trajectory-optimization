# Map generation and visualization
from utils.mapgen_v2 import generate_drone_map
from utils.mapgen_v2 import map_and_grid_visualization
#from utils.mapgen_v2 import generate_occupancy_grid
from utils.kmeans import kmeans_clustering
from utils.save_map import save_map
from utils.dwa import drone_model
from utils.dwa import sample_acc
from utils.dwa import compute_obstacles_cost
from utils.dwa import plot_dwa_results
from utils.dwa import Drone

# to save and reload maps and occupancy grids
import pickle
from pathlib import Path

from scipy.spatial import KDTree

# Voronoi partitioning and visualization
from partition.voronoi import voronoi_partition, plot_voronoi

import numpy as np
import matplotlib.pyplot as plt

# MODIFY THIS AT THE START !
reload_map= True
# size of the map sizexsize
size = 50  

if reload_map == False:
    # 2D map generation parameters                         # size of the map (20x20)
    maxheight = 10                      # maximum height of obstacles
    num_obstacles = 40                  # number of obstacles to generate
    density = 0.5                       # density of obstacles
    num_drones = 5                      # number of drones

    # drone starting position selection
    drone_start = 'centered'  # 'random' or 'centered'


    # 2D map generation
    '''
    free_space is used for Voronoi tessellation.
    obstacles are used for pyBullet simulation.
    occupancy grid is used for trajectory optimization.
    '''

    workspace, obstacles, free_space, drone_starts, occupancy_grid =  generate_drone_map(size, maxheight, num_obstacles, density, num_drones)
    print("workspace:", workspace)
    # print("obstacles:", obstacles)
    # print("free_space:", free_space)    
    print("drone_starts:", drone_starts)

    #occupancy_grid = generate_occupancy_grid(workspace, obstacles, size)

    # k-means clustering to find optimal starting points for Voronoi partitioning
    if drone_start != 'random':
        centroids = kmeans_clustering(free_space, num_drones)
        print("Centroids:", centroids)
    else:
        centroids = drone_starts

        # Map visualization
    map_and_grid_visualization(workspace, obstacles, drone_starts , occupancy_grid, centroids)

    user_choice = input("Do you want to save this map? (y/n): ").lower()
        
    if user_choice == 'y':
        filename = input("Enter map name: ")
        save_map(workspace, occupancy_grid, obstacles, drone_starts, centroids, filename)
        print(f"Map saved as {filename}.pkl")
    else:
        print("Map discarded.")

else: 
    filename= input("Write name of the map you want to load (without .pkl): ")
    filename= f"{filename}.pkl"
    maps_path= Path("data")
    file_path= maps_path / filename
    # Reload the map object
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    # Assign back to variables
    workspace = loaded_data["map"]
    occupancy_grid = loaded_data["grid"]
    obstacles= loaded_data["obstacles"]
    drone_starts= loaded_data["drone_starts"]
    centroids= loaded_data["centroids"]

    map_and_grid_visualization(workspace, obstacles, drone_starts , occupancy_grid, centroids)

# Voronoi partitioning
vor, voronoi_cells = voronoi_partition(centroids, size)

# Visualize the Voronoi partitioning
plot_voronoi(vor, drone_starts, size)

# -------------------------------------------------------------------------------------------------------
# REGION ASSIGNEMENT TO BE DONE (BUT COULD WORK WITHOUT MAYBE)
# ---------------------------------------------------------------------------------------------------------
# DWA WAYPOINT GENERATION

# Crea l'albero di ricerca spaziale 
kd_tree_obstacles = KDTree(obstacles)
# safety margin for each drone
safe_radius = 0.5 # [m]

# DWA PIPELINE TO BE IMPLEMENTED
for i in range(len(drone_starts)):
    drone= Drone(drone, i, drone_starts[i])
    for _ in range(3):
        sample_acc()







