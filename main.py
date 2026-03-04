from utils.map_generation import generate_drone_map
from partition.voronoi import compute_voronoi_cells
from utils.plot_environment import plot_environment

workspace, obstacles, free_space, drone_starts =  generate_drone_map(40, 0.05, 15, 5)

cells = compute_voronoi_cells(drone_starts, free_space)

plot_environment(workspace, obstacles, cells, drone_starts)