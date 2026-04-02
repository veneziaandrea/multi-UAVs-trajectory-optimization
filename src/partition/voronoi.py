import numpy as np
import sys
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import shapely

# Ensure that src is on sys.path even when running this module directly
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment.map_generation_v2 import Map3D

EPS = 1e-9

def clip_polygon_with_half_plane(
    polygon: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    keep_leq: bool = True,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def signed_value(vertex: np.ndarray) -> float:
        value = float(np.dot(normal, vertex) - offset)
        return value if keep_leq else -value

    clipped: list[np.ndarray] = []
    for index in range(len(polygon)):
        current = polygon[index]
        previous = polygon[index - 1]
        value_current = signed_value(current)
        value_previous = signed_value(previous)
        inside_current = value_current <= EPS
        inside_previous = value_previous <= EPS

        if inside_current != inside_previous:
            direction = current - previous
            denominator = float(np.dot(normal, direction))
            if abs(denominator) > EPS:
                t = (offset - float(np.dot(normal, previous))) / denominator
                t = min(1.0, max(0.0, t))
                clipped.append(previous + t * direction)
        if inside_current:
            clipped.append(current)

    if not clipped:
        return np.empty((0, 2), dtype=float)

    return np.asarray(clipped, dtype=float)


def assign_area(vor_partition, drone_positions):
    cell_items = list(vor_partition.Voronoi_Cells.items())
    seeds = np.array([cell.seed for _, cell in cell_items])
    
    # Matrice dei costi (distanze al quadrato)
    dist_matrix = np.sum((seeds[:, np.newaxis, :] - drone_positions[np.newaxis, :, :])**2, axis=2)
    
    # Risolve il problema dell'assegnazione ottima (Minimizza distanza totale)
    # Garantisce 1 drone -> 1 cella senza duplicati
    cell_indices, drone_indices = linear_sum_assignment(dist_matrix)
    
    new_cells = {}
    for c_idx, d_idx in zip(cell_indices, drone_indices):
        cell = cell_items[c_idx][1]
        cell.drone_id = int(d_idx)
        new_cells[cell.drone_id] = cell
        
    vor_partition.Voronoi_Cells = new_cells

def get_waypoints_in_partition(waypoints_np, partition_polygon):
    """
    Filters a numpy array of waypoints [N x 3] to find those inside a Shapely polygon.
    """
    # This returns a boolean mask [True, False, True...]
    mask = shapely.contains_xy(partition_polygon, waypoints_np[:, 0], waypoints_np[:, 1])

    # Apply the mask to get the waypoints in one shot
    local_wps_np = waypoints_np[mask]

    local_wps_np = np.atleast_2d(local_wps_np)
            
    return local_wps_np

@dataclass
class Voronoi_Cell: 
    ''' Voronoi cell assigned to a drone, considered in the 2D plane for partitioning the workspace. '''
    
    drone_id: int       # drone ID associated to the cell
    seed: np.ndarray    # (x, y) coordinates of the seed point for this cell
    polygon: np.ndarray # vertices of the polygon representing the Voronoi cell in 2D


@dataclass
class Voronoi_Partition:
    Voronoi_Cells: dict[int, Voronoi_Cell]      # mapping from drone ID to its corresponding Voronoi cell
        
    @classmethod
    def build(cls, cells, seeds_xy, map3D: Map3D):
        ''' Build a Voronoi partition of the environment workspace based on the given seed points. 
            Each cell in the partition corresponds to a seed point and contains all points in the 
            workspace that are closer to that seed than to any other seed.
             
            Args:
                seeds_xy: An array of shape (N, 2) containing the (x, y) coordinates of the seed points for the Voronoi partition.
                map3D: An instance of Map3D representing the environment in which the Voronoi partition is to be constructed.
                
            Returns:
                Voronoi_Partition: An instance of Voronoi_Partition representing the constructed Voronoi partition. '''

        seeds_xy = np.asarray(seeds_xy, dtype=float)
        if seeds_xy.ndim != 2 or seeds_xy.shape[1] != 2:
            raise ValueError("seeds_xy must have shape (N, 2).")

        workspace_limits = map3D.workspace # array of shape (4,2) representing the corners of the workspace rectangle
        cells = {}
        for index, seed in enumerate(seeds_xy):
            polygon = np.array(workspace_limits, copy=True)
            for other_index, other_seed in enumerate(seeds_xy):
                if index == other_index:
                    continue
                # Compute the normal vector for the half-plane defined by the seed and the other seed
                normal = np.asarray(other_seed - seed, dtype=float)
                # Compute the offset for the half-plane
                offset = 0.5 * (float(np.dot(other_seed, other_seed)) - float(np.dot(seed, seed)))
                # Clip the current polygon with the half-plane defined by the normal and offset
                polygon = clip_polygon_with_half_plane(polygon, normal, offset, keep_leq=True)
                if len(polygon) == 0:
                    break
            cells[index] = Voronoi_Cell(drone_id=index, seed=np.asarray(seed, dtype=float), polygon=polygon)
        return cls(Voronoi_Cells=cells)
    

    



