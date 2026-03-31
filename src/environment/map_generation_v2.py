import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

class Obstacle: 
    '''Cylindrical obstacle in 3D space'''
    
    def __init__(self, x, y, radius, height):
        self.x = x
        self.y = y
        self.radius = radius
        self.height = height                    # height is not used in 2D map generation but can be relevant for 3D trajectory planning
        self.shape = Point(x, y).buffer(radius) # cylindrical obstacle represented as a circle in 2D

    @property
    def center_xy(self):
        return np.array([self.x, self.y])

class Map3D:
    '''3D map containing obstacles and drone starting positions'''
    
    def __init__(self, x_bounds, y_bounds, z_bounds, obstacles):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.obstacles = obstacles

        # geometric representation
        self.workspace = np.array(
            [
                [x_bounds[0], y_bounds[0]],
                [x_bounds[1], y_bounds[0]],
                [x_bounds[1], y_bounds[1]],
                [x_bounds[0], y_bounds[1]],
            ],
            dtype=float,
        )
        self.workspace_polygon = Polygon(self.workspace)
        self.obstacle_union = unary_union([obstacle.shape for obstacle in obstacles]) if obstacles else None
        if self.obstacle_union is None or self.obstacle_union.is_empty:
            self.free_space = self.workspace_polygon
        else:
            self.free_space = self.workspace_polygon.difference(self.obstacle_union)
    @classmethod
    def generate_map3D(
        cls,
        x_bounds,
        y_bounds,
        z_bounds,
        num_obstacles,
        obstacle_radius_range,
        obstacle_height_range,
        num_drones,                 
        spacing,               
        seed=None,
    ):
        """Generate a 3D map with obstacles avoiding drone starting positions"""

        rng = np.random.default_rng(seed)

        # --- 1. Generate drone starting positions (come il tuo codice originale) ---
        center = np.array([
            (x_bounds[0] + x_bounds[1]) / 2,
            (y_bounds[0] + y_bounds[1]) / 2
        ])

        drone_starts = []
        layer = 1

        while len(drone_starts) < num_drones:
            offsets = [
                ( layer, 0), (-layer, 0),
                (0,  layer), (0, -layer),
                ( layer,  layer), ( layer, -layer),
                (-layer,  layer), (-layer, -layer)
            ]

            for dx, dy in offsets:
                if len(drone_starts) >= num_drones:
                    break

                pos = np.array([
                    center[0] + dx * spacing,
                    center[1] + dy * spacing
                ])

                drone_starts.append(pos)

            layer += 1

        drone_starts = np.array(drone_starts)

        # --- 2. Generate obstacles ---
        obstacles = []
        max_attempts = num_obstacles * 200
        attempts = 0

        safety_distance = 0.5  # distanza minima dai droni

        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1

            radius = float(rng.uniform(*obstacle_radius_range))
            height = float(rng.uniform(*obstacle_height_range))

            x = float(rng.uniform(x_bounds[0] + radius + 1.0, x_bounds[1] - radius - 1.0))
            y = float(rng.uniform(y_bounds[0] + radius + 1.0, y_bounds[1] - radius - 1.0))

            new_obstacle = Obstacle(x, y, radius, height)

            overlap = False

            # --- check overlap con altri ostacoli ---
            for obs in obstacles:
                dist = np.linalg.norm(new_obstacle.center_xy - obs.center_xy)
                if dist < (new_obstacle.radius + obs.radius + 1.0):
                    overlap = True
                    break

            # --- check distanza da drone starts ---
            if not overlap:
                for ds in drone_starts:
                    dist = np.linalg.norm(new_obstacle.center_xy - ds)
                    if dist < (new_obstacle.radius + safety_distance):
                        overlap = True
                        break

            if not overlap:
                obstacles.append(new_obstacle)

        return cls(x_bounds, y_bounds, z_bounds, obstacles), drone_starts
            


    



