# Map generation and visualization
from utils.map_generation import generate_drone_map
from utils.map_generation import map_and_grid_visualization
from utils.map_generation import generate_occupancy_grid
from utils.kmeans import kmeans_clustering

# Voronoi partitioning and visualization
from partition.voronoi import voronoi_partition, plot_voronoi

import numpy as np
import matplotlib.pyplot as plt


# 2D map generation parameters
size = 20                           # size of the map (20x20)
maxheight = 10                      # maximum height of obstacles
num_obstacles = 20                  # number of obstacles to generate
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
print("Generating map...")
workspace, obstacles, free_space, drone_starts =  generate_drone_map(size, maxheight, num_obstacles, density, num_drones)
print("Map generation completed.")
print("workspace:", workspace)
# print("obstacles:", obstacles)
# print("free_space:", free_space)    
print("drone_starts:", drone_starts)

occupancy_grid = generate_occupancy_grid(workspace, obstacles, size)

# k-means clustering to find optimal starting points for Voronoi partitioning
if drone_start != 'random':
    centroids = kmeans_clustering(free_space, num_drones)
    print("Centroids:", centroids)
else:
    centroids = drone_starts

# Map visualization

# map_and_grid_visualization(workspace, obstacles, drone_starts, occupancy_grid, centroids)

# Voronoi partitioning
vor, voronoi_cells = voronoi_partition(centroids, size)


# Visualize the Voronoi partitioning
plot_voronoi(vor, drone_starts, size)






