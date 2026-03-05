import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import time

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    Parameters:
    - vor: Voronoi object from scipy.spatial
    - radius: Distance to 'points at infinity'

    Returns:
    - regions: List of Voronoi regions as lists of vertices
    - vertices: Array of Voronoi vertices
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map containing all ridges for a point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)


def clip_to_bbox(polygon, bbox):
    '''Clip a polygon to a bounding box'''  

    min_x, min_y, max_x, max_y = bbox
    clipped = []
    for x, y in polygon:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            clipped.append((x, y))
    return clipped


def voronoi_partition(drone_starts, size):
    '''Compute Voronoi partitioning for drone starting positions within the free space'''
    start = time.time()
    vor = Voronoi(drone_starts)
    print("voronoi:", time.time()-start)

    start = time.time()
    regions, vertices = voronoi_finite_polygons_2d(vor)
    print("finite_polygons:", time.time()-start)
    bbox = (0, size, 0, size)
    
    voronoi_cells = []
    for region in regions:
        polygon = vertices[region]
        clipped_polygon = clip_to_bbox(polygon, bbox)
        voronoi_cells.append(clipped_polygon)
    
    return voronoi_cells


def plot_voronoi(voronoi_cells, drone_starts, size):
    '''Visualize Voronoi cells and drone starting positions'''
    plt.figure(figsize=(8, 8))
    for cell in voronoi_cells:
        if len(cell) > 2:  # Only plot valid polygons
            plt.fill(*zip(*cell), alpha=0.4)
    plt.scatter(drone_starts[:, 0], drone_starts[:, 1], c='red', marker='x')
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.title("Voronoi Partitioning of Drone Starting Positions")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()