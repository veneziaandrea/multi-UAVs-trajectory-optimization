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
    ''' Generate a 2D map with random obstacles and drone starting positions.
        Returns:
        - workspace: Shapely Polygon representing the boundaries of the map
        - obstacles: List of obstacle objects with geometric shapes
        - free_space: Shapely Polygon representing the free space in the map
        - drone_starts: List of starting positions for the drones  
    '''

    min_radius = 0.5
    max_radius = 1.5
    inflation = 0.6 # To avoid placing obstacles too close to the edges and drone starting positions
    center = size / 2


    '''Generate a 2D map with random obstacles and drone starting positions'''
    workspace = Polygon([(0,0), (size,0), (size,size), (0,size)])

    # Drone Starting Positions
    drone_starts = []
    offset = [0, 0]
    layer = 1
    offsets = []

    # Spawning drones in a grid pattern around the center, ensuring they are not too close to each other
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

        # Obstacle Generation + Occupancy Grid
    obstacles = []
    # Fixed: np.zeros expects a tuple for shape: (rows, cols)
    grid = np.zeros((size, size), dtype=np.uint16)
    cell_size = workspace.bounds[2] / size

    for _ in range(num_obstacles):
        valid_obstacle = False
        attempts = 0
        while not valid_obstacle and attempts < 50:
            attempts += 1

            r = random.uniform(min_radius, max_radius)
            # Assuming size here is the coordinate limit; 
            # ensure these units match your workspace bounds
            x = random.uniform(inflation, size - inflation)
            y = random.uniform(inflation, size - inflation)
            h = random.uniform(1, maxheight)

            valid_obstacle = True
            
            # Ensure new obstacle doesn't overlap with existing ones
            for obs in obstacles:
                d = np.sqrt((x - obs.x)**2 + (y - obs.y)**2) 
                if d < r + obs.radius + inflation:  
                    valid_obstacle = False
                    break
            
            if not valid_obstacle: continue

            # Ensure new obstacle doesn't overlap with drone starting positions
            safety_distance = 0.2
            for ds in drone_starts:
                d = np.sqrt((x - ds[0])**2 + (y - ds[1])**2)
                if d < r + safety_distance:
                    valid_obstacle = False
                    break

        if valid_obstacle:
            new_obs = obstacle(x, y, r, h)
            obstacles.append(new_obs)
            
            # OPTIMIZATION: Instead of re-checking every cell in the grid for every obstacle,
            # only check cells within the bounding box of the new obstacle.
            
            x_min = max(0, int((x - r) / cell_size))
            x_max = min(size, int((x + r) / cell_size) + 1)
            y_min = max(0, int((y - r) / cell_size))
            y_max = min(size, int((y + r) / cell_size) + 1)

            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    # Calculate cell center
                    cell_x = i * cell_size + cell_size / 2
                    cell_y = j * cell_size + cell_size / 2
                    point = Point(cell_x, cell_y)

                    if new_obs.shape.contains(point):
                        grid[j, i] = 1

    obstacle_union = unary_union([obs.shape for obs in obstacles])

    # Free space: workspace minus obstacles
    free_space = workspace.difference(obstacle_union)

    
    return workspace, obstacles, free_space, drone_starts, grid

''' # --- OCCUPANCY GRID GENERATION ---
def generate_occupancy_grid(workspace, obstacles, size):
    Generate a binary occupancy grid from the workspace and obstacles
    grid = np.zeros((size, size), dtype=np.uint8)
    cell_size = workspace.bounds[2] / size

    for i in range(size):      # x-axis
        for j in range(size):  # y-axis
            x = i * cell_size + cell_size / 2   # Center of the cell
            y = j * cell_size + cell_size / 2
            point = Point(x, y)

            if any(obs.shape.contains(point) for obs in obstacles):
                grid[j, i] = 1  # Note: swap i and j
            else:
                grid[j, i] = 0  # Free space
    return grid 
    '''


# --- VISUALIZATION ---
def map_and_grid_visualization(workspace, obstacles, drone_starts, occupancy_grid, centroids):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Two subplots side by side

    # ---------------- Map subplot ----------------
    ax_map = axes[0]
    # Workspace boundary
    x, y = workspace.exterior.xy
    ax_map.plot(x, y, color='black')
    # Obstacles
    for obs in obstacles:
        ox, oy = obs.shape.exterior.xy
        ax_map.fill(ox, oy, color='gray', alpha=0.7)
        
    # Drone starting positions
    if drone_starts is not None:
        ax_map.scatter(drone_starts[:, 0], drone_starts[:, 1], c='red', marker='x', label='Drone Starts')
    # Centroids from K-means
    if centroids is not None:
        ax_map.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='o', label='K-means Centroids')
    
    ax_map.set_title("Drone Map with Obstacles")
    ax_map.set_xlabel("X-axis")
    ax_map.set_ylabel("Y-axis")
    ax_map.set_xlim(0, workspace.bounds[2])
    ax_map.set_ylim(0, workspace.bounds[3])
    ax_map.legend()
    ax_map.grid(True, linestyle='--', alpha=0.5)

    # ---------------- Occupancy Grid subplot ----------------
    ax_grid = axes[1]
    ax_grid.imshow(occupancy_grid, origin='lower', cmap='Greys',
                   extent=(0, workspace.bounds[2], 0, workspace.bounds[3]))
    ax_grid.set_title("Occupancy Grid")
    ax_grid.set_xlabel("X-axis")
    ax_grid.set_ylabel("Y-axis")
    ax_grid.grid(True, linestyle='--', alpha=0.5)
    # Optional: show colorbar
    plt.colorbar(ax_grid.images[0], ax=ax_grid, label='Occupied')

    plt.tight_layout()
    plt.show(block=False)
