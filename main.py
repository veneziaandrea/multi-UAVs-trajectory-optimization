from utils.map_generation import generate_drone_map
from utils.map_generation import map_and_grid_visualization
from utils.map_generation import generate_occupancy_grid

from partition.voronoi import voronoi_partition
from partition.voronoi import plot_voronoi
import numpy as np

# 2D map generation parameters
size = 20
maxheight = 10
num_obstacles = 20
density = 0.5
num_drones = 5  


# 2D map generation
'''
free_space is used for Voronoi tessellation.
obstacles are used for pyBullet simulation.
occupancy grid is used for trajectory optimization.
'''

workspace, obstacles, free_space, drone_starts =  generate_drone_map(size, maxheight, num_obstacles, density, num_drones)

occupancy_grid = generate_occupancy_grid(workspace, obstacles, size)


# Map visualization
map_and_grid_visualization(workspace, obstacles, drone_starts, occupancy_grid)

# Voronoi partitioning
#voronoi_cells = voronoi_partition(drone_starts, size)
#print("Voronoi partitioning completed.")

# Visualize the Voronoi partitioning
#plot_voronoi(voronoi_cells, drone_starts, size)



