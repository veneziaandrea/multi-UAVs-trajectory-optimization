import numpy as np

from utils.geometry import create_rectangle
from shapely.geometry import Point, Polygon
from dataclasses import dataclass   



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
        self.workspace = create_rectangle(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

    @classmethod
    def generate_map3D(
        cls,
        x_bounds,
        y_bounds,
        z_bounds,
        num_obstacles,
        obstacle_radius_range,
        obstacle_height_range,
        seed=None,
    ):
        ''' Generate a 3D map with random obstacles and drone starting positions'''
        rng = np.random.default_rng(seed)   # use numpy's random generator
        obstacles = []

        # maximum number of attempst to avoid infinite loops
        max_attempts = num_obstacles * 200
        attempts = 0

        # Generate non-overlapping obstacles
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # randomly sample obstacle parameters
            radius = float(rng.uniform(*obstacle_radius_range))
            height = float(rng.uniform(*obstacle_height_range))

            # ensure obstacle is within map's bounds
            x = float(rng.uniform(x_bounds[0] + radius + 1.0, x_bounds[1] - radius - 1.0))
            y = float(rng.uniform(y_bounds[0] + radius + 1.0, y_bounds[1] - radius - 1.0))

            
            new_obstacle = Obstacle(x, y, radius, height)
            
            # check for overlaps with existing obstacles
            overlap = False
            for obs in obstacles:
                dist = np.linalg.norm(new_obstacle.center_xy - obs.center_xy)
                if dist < (new_obstacle.radius + obs.radius + 1.0):
                    overlap = True
                break

            if not overlap:
                obstacles.append(new_obstacle)

        return cls(x_bounds, y_bounds, z_bounds, obstacles)
    
    def is_collision_free(self, point_xyz, margin=0.0):
        '''Check if a point is collision-free considering the obstacles and workspace boundaries'''
        point_2d = Point(point_xyz[0], point_xyz[1])
        
        # Check if point is within workspace boundaries
        if not self.workspace.buffer(-margin).contains(point_2d):
            return False
        
        # Check for collisions with obstacles
        for obs in self.obstacles:
            if obs.shape.buffer(margin).contains(point_2d):
                return False
        
        return True
    
    






    



