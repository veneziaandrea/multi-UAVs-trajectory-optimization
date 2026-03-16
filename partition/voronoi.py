import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
import matplotlib.pyplot as plt
import time


def voronoi_partition(drone_starts):
    '''Compute Voronoi partitioning for drone starting positions within the free space'''
    start = time.time()
    vor = Voronoi(drone_starts)
    print("voronoi:", time.time()-start)

    return vor



def plot_voronoi(vor, drone_starts):
    '''Visualize Voronoi cells and drone starting positions'''
    fig = voronoi_plot_2d(vor, show_vertices=False,
                          line_colors='orange',
                          line_width=2,
                          line_alpha=0.6,
                          point_size=5)
    plt.plot(drone_starts[:, 0], drone_starts[:, 1], 'ro', label='Drone Starts')
    plt.title("Voronoi Diagram (2D)")   
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.show(block=False)

    
