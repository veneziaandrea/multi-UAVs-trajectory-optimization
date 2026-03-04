import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import random

def generate_drone_map(size, maxheight, num_obstacles, density, num_drones, manual_drone_position=None):
    """
    Generate a random map for drone trajectory optimization.

    Parameters:
    - size: of the map (square area)
    - maxheight: maximum height of obstacles
    - num_obstacles: number of obstacles to generate
    - num_drones: number of drones to generate
    - manual_drone_position: if provided, a list of manual drone positions (x, y, z)

    Returns:
    - obstacles: list of shapely Polygons representing obstacles
    - drone_positions: list of tuples (x, y, z) representing drone positions
    """
    if num_obstacles is None:
        area = size ** 2
        num_obstacles = int(area * density)  # Adjust density as needed
    
    min_radius = 0.5
    max_radius = 2.5
    safety_margin = 0.5

    # Workspace boundaries
    workspace = Polygon([
        (0,0),
        (size,0),
        (size,size),
        (0,size)
    ])

    # -----------------------
    # --- Drone Positions ---
    # -----------------------

    drone_positions = []    
    if manual_drone_position is not None:
        drone_starts = manual_drone_position
    else:
        drone_starts = []
        center = size / 2
        inflation = 0.6 # To avoid placing drones too close to the edges

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
            drone_starts.append([center + off[0],
                                 center + off[1]])

        drone_starts = np.array(drone_starts)

    # ---------------------------
    # --- Obstacle Generation ---
    # ---------------------------
    obstacles = []
    occupied = [(p[0], p[1], 0.6) for p in drone_starts]

    for _ in range(num_obstacles):
        valid = False
        attempts = 0

        while not valid and attempts < 50:
            attempts += 1

            r = random.uniform(min_radius, max_radius)
            x = random.uniform(0, size)
            y = random.uniform(0, size)

            distances = [
                np.linalg.norm(np.array([x,y]) - np.array([ox,oy]))
                for ox,oy,_ in occupied
            ]

            if all(d > (orad + r + safety_margin)
                   for d,(_,_,orad) in zip(distances, occupied)):
                valid = True

        if valid:
            obstacles.append(Point(x,y).buffer(r))
            occupied.append((x,y,r))

    obstacle_union = unary_union(obstacles)
    free_space = workspace.difference(obstacle_union)

    return workspace, obstacles, free_space, drone_starts




