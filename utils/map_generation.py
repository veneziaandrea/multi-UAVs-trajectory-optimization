import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import random
import matplotlib.pyplot as plt

# --- MAP AND OBSTACLE CLASSES ---
class map: 
    def __init__(self, size, maxheight, num_obstacles, density, num_drones):
        self.size = size
        self.maxheight = maxheight
        self.num_obstacles = num_obstacles
        self.density = density
        self.num_drones = num_drones

        # geometric representation
        self.workspace = None
        self.obstacles = []  
        self.free_space = None
        self.drone_starts = []

        # grid representation
        self.grid = None


class obstacle:
    def __init__(self, x, y, radius, height):
        self.x = x
        self.y = y
        self.radius = radius
        self.height = height                    # height is not used in 2D map generation but can be relevant for 3D trajectory planning
        self.shape = Point(x, y).buffer(radius) # cylindrical obstacle represented as a circle in 2D

# --- MAP GENERATION ---

def generate_drone_map(size, maxheight, num_obstacles, density, num_drones):

    min_radius = 0.5
    max_radius = 1.5

    '''Generate a 2D map with random obstacles and drone starting positions'''
    workspace = Polygon([(0,0), (size,0), (size,size), (0,size)])

    # Obstacle Generation
    obstacles = []

    center = size / 2
    inflation = 0.6 # To avoid placing obstacles too close to the edges 

    for _ in range(num_obstacles):
        valid_obstacle = False
        attempts = 0
        while not valid_obstacle and attempts < 50:
            attempts += 1

            r = random.uniform(min_radius, max_radius)  # Random radius for obstacles
            x = random.uniform(inflation, size - inflation)
            y = random.uniform(inflation, size - inflation)
            h = random.uniform(1, maxheight)  # Random height for obstacles

            valid_obstacle = True
            for obs in obstacles:
                d = np.sqrt((x-obs.x)**2 + (y-obs.y)**2) 

                if d < r + obs.radius + inflation:  # Ensure new obstacle doesn't overlap with existing ones
                    valid_obstacle = False
                    break
        
        if valid_obstacle:
            obstacles.append(obstacle(x, y, r, h))

    obstacle_union = unary_union([obs.shape for obs in obstacles])

    # Free space: workspace minus obstacles
    free_space = workspace.difference(obstacle_union)

    # Drone Starting Positions
    drone_starts = []
    offset = [0, 0]
    layer = 1
    offsets = []

    while len(offsets) < num_drones:
            candidates = [
                ( layer*inflation, 0),
                (-layer*inflation, 0),
                (0,  layer*inflation),
                (0, -layer*inflation),
                ( layer*inflation,  layer*inflation),
                ( layer*inflation, -layer*inflation),
                (-layer*inflation,  layer*inflation),
                (-layer*inflation, -layer*inflation)
            ]
            for c in candidates:
                if len(offsets) < num_drones:
                    offsets.append(c)
            layer += 1

    for off in offsets[:num_drones]:
        drone_starts.append([center + off[0], center + off[1]])

    drone_starts = np.array(drone_starts)

    return workspace, obstacles, free_space, drone_starts

# --- OCCUPANCY GRID GENERATION ---
def generate_occupancy_grid(workspace, obstacles, size):
    '''Generate a binary occupancy grid from the workspace and obstacles'''
    grid = np.zeros((size, size), dtype=np.uint8)
    cell_size = workspace.bounds[2] / size

    for i in range(size):
        for j in range(size):
            x = i * cell_size + cell_size / 2   # Center of the cell
            y = j * cell_size + cell_size / 2
            point = Point(x, y)

            if any(obs.shape.contains(point) for obs in obstacles):
                grid[i, j] = 1  # Occupied
            else:
                grid[i, j] = 0  # Free

    return grid


# --- VISUALIZATION ---
def map_visualization(workspace, obstacles, drone_starts):
    ax = plt.subplots(figsize=(8, 8))[1]
    # Plot workspace
    x, y = workspace.exterior.xy
    ax.plot(x, y, color='black')
    # Plot obstacles
    for obs in obstacles:
        x, y = obs.shape.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.7)
    # Plot drone starting positions
    if len(drone_starts) > 0:
        ax.scatter(drone_starts[:, 0], drone_starts[:, 1], c='red', marker='x', label='Drone Starts')
    ax.set_title("Drone Map with Obstacles and Starting Positions")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_xlim(0, workspace.bounds[2])
    ax.set_ylim(0, workspace.bounds[3])
    ax.legend()
    ax.grid()
    plt.show()

