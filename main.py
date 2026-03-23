# Map generation and visualization
from utils.mapgen_v2 import generate_drone_map
from utils.mapgen_v2 import map_and_grid_visualization
#from utils.mapgen_v2 import generate_occupancy_grid
from utils.kmeans import kmeans_clustering
from utils.save_map import save_map
from utils.dwa import drone_model
from utils.dwa import sample_acc
from utils.dwa import compute_obstacles_cost
from utils.dwa import plot_final_trajectories
from utils.dwa import Drone

# to save and reload maps and occupancy grids
import pickle
from pathlib import Path

from scipy.spatial import KDTree

# Voronoi partitioning and visualization
from partition.voronoi import voronoi_partition, plot_voronoi

import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
# ... rest of your imports

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

'''TO BE TESTED'''
# Extract 3D coordinates (x, y, and half the height for the z-center)
obstacle_coords = np.array([[obs.x, obs.y, obs.height / 2.0] for obs in obstacles])
# Create an array of radii to match the order of the tree
obs_radii = np.array([obs.radius for obs in obstacles])
# Create obstacles object in a way that is actually fast to use 
obs_tree = KDTree(obstacle_coords)

# Make drone starts actually in 3D  
# Loop through each [x, y] pair and create a new 3D numpy array
drone_starts = [np.array([start[0], start[1], 0.0]) for start in drone_starts]

# Problem initialization
safe_radius = 0.5 
N_tot = 300
num_iter = 1000 
iter_count = 0

# Convert physical states to lists so EVERY drone has its own independent history
pos_i = list(drone_starts) 
vel_i = [np.zeros(3) for _ in range(len(drone_starts))]
a_prev = [np.zeros(3) for _ in range(len(drone_starts))]

lim = [Drone.vel_lim, Drone.acc_lim]

# Calculate distances to centroids
dist_centr = np.zeros((len(drone_starts), len(centroids))) 

closer_centroids = [np.argmin(dist_centr[i]) for i in range(len(drone_starts))] 

# Initialize reference array. If ref_j is used by DWA to avoid other drones, 
# it MUST be initialized as their starting 3D positions, not integer indices.
ref_j = list(drone_starts) 
drones = [None] * len(drone_starts) 

# create drone objects and assign as starting reference the closest centroid
for i in range(len(drone_starts)):
    drones[i]= Drone(i, drone_starts[i], lim)
    
    # FIXED: Extract the 2D coordinates using the index, and append 0.0 for the Z-axis
    c_idx = closer_centroids[i]
    ref_j[i] = np.array([centroids[c_idx][0], centroids[c_idx][1], 0.0])

acc_star = [None] * len(drone_starts) 
waypoints = [None] * len(drone_starts) 
best_idx = [None] * len(drone_starts) 
J_min_vec = np.zeros(len(drone_starts)) 

# Opt problem: Now represents physical TIME STEPS, not convergence of a single step
while iter_count <= num_iter: 
    
    # 1. Distributed Planning Phase
    for i in range(len(drone_starts)):
        # Pass the i-th drone's specific state variables
        waypoints[i], acc_star[i], J_min_vec[i], best_idx[i] = drones[i].DWA(
            pos_i[i], 
            vel_i[i], 
            ref_j,       # Current positions of all drones for dispersion
            a_prev[i],   # This drone's specific warm start
            Drone.acc_lim, 
            Drone.T_h, 
            Drone.w1, 
            Drone.w2, 
            obs_tree, 
            safe_radius, 
            obs_radii
        )
    
    # 2. Execution / Kinematic Update Phase
    # We update the states ONLY AFTER all drones have planned, 
    # to maintain synchronous behavior and avoid unfair advantages.
    trajectory_history = [[np.array(start)] for start in drone_starts] # for recording trajectories for plotting at the end
    for i in range(len(drone_starts)):
        # Update physical states for the next time step
        pos_i[i] = waypoints[i] 
        vel_i[i] = vel_i[i] + acc_star[i] * Drone.T_h # Basic Euler integration
        a_prev[i] = acc_star[i]
        
        # Update the shared reference map for the next loop
        ref_j[i] = waypoints[i] 
        # RECORDING: Save the chosen waypoint into that drone's history
        trajectory_history[i].append(np.array(waypoints[i]))

    iter_count += 1
    
    # Optional: Break the loop early if all drones have reached their centroids
    # (You would need to define a distance threshold check here)
print("DWA optimization completed.")

# AFTER the loop finishes, call the final plot
plot_final_trajectories(trajectory_history, loaded_data["obstacles"], [d.id for d in drones])




        







