import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def vornoi_finite_polygons(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions.

    """
    # 2D input check
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist()) 
        new_regions.append(new_region)  

    return new_regions, np.asarray(new_vertices)




def voronoi_partition(points, free_space):
    vor = Voronoi(points)
    regions, vertices = vornoi_finite_polygons(vor)

    polygons = []
    for region in regions:
        polygon = vertices[region]

        poly = Polygon(polygon)
        if poly.is_valid:
            polygons.append(poly)

    union_poly = unary_union(polygons)
    partitioned_space = union_poly.intersection(free_space)

    return partitioned_space


def plot_environment(workspace, obstacles, cells, points):

    fig, ax = plt.subplots()

    # Workspace
    x,y = workspace.exterior.xy
    ax.plot(x,y)

    # Obstacles
    for obs in obstacles:
        x,y = obs.exterior.xy
        ax.fill(x,y, alpha=0.5)

    # Cells
    for cell in cells:
        if not cell.is_empty:
            x,y = cell.exterior.xy
            ax.fill(x,y, alpha=0.3)

    ax.scatter(points[:,0], points[:,1], c='red')
    ax.set_aspect('equal')
    plt.show()