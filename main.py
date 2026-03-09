from utils.map_generation import generate_drone_map
from utils.map_generation import map_visualization
from partition.voronoi import voronoi_partition
from partition.voronoi import plot_voronoi
import numpy as np

# 2D map generation parameters
size = 20
maxheight = 10
num_obstacles = 20
density = 0.05
num_drones = 5  

# Generate the map and visualize it
workspace, obstacles, free_space, drone_starts =  generate_drone_map(size, maxheight, num_obstacles, density, num_drones)

map_visualization(workspace, obstacles, drone_starts)

# Voronoi partitioning
#voronoi_cells = voronoi_partition(drone_starts, size)
#print("Voronoi partitioning completed.")

# Visualize the Voronoi partitioning
#plot_voronoi(voronoi_cells, drone_starts, size)



