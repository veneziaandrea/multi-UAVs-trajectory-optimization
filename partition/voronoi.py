import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
from shapely.geometry import Polygon
from shapely.geometry import box
import matplotlib.pyplot as plt
import time

def voronoi_partition(centroids, size):
    """
    Compute finite Voronoi regions clipped to a square of dimension `size`.
    """
    vor = Voronoi(centroids)
    bbox_poly = box(0, 0, size, size)
    radius = size * 2  # estende abbastanza le regioni infinite

    finite_polygons = []

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 not in region:
            # regione già finita
            poly_coords = vor.vertices[region]
        else:
            # regione infinita → sostituiamo -1 con punto lontano
            poly_coords = []
            for v_idx in region:
                if v_idx == -1:
                    # aggiunge un punto lontano verso l'esterno della mappa
                    poly_coords.append(vor.points[point_idx] + np.random.rand(2)*radius - radius/2)
                else:
                    poly_coords.append(vor.vertices[v_idx])
            poly_coords = np.array(poly_coords)

        poly = Polygon(poly_coords)
        if poly.is_valid and poly.intersects(bbox_poly):
            finite_polygons.append(poly.intersection(bbox_poly))

    return vor, finite_polygons



def plot_voronoi(vor, drone_starts, size):
    '''Visualize Voronoi cells and drone starting positions'''
    fig = voronoi_plot_2d(vor,
                          show_vertices=False,
                          line_colors='orange',
                          line_width=2,
                          line_alpha=0.6,
                          point_size=5)
    plt.plot(drone_starts[:, 0], drone_starts[:, 1], 'ro', label='Drone Starts')
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Voronoi Diagram (2D)")   
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.show(block=False)

    while True:
        plt.pause(1)
    
