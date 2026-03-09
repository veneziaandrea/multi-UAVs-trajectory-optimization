# code to obtain the centroids of the clusters of the free space, to be used as starting points for the 
# voronoi partitioning instead of the drone starting positions, in case the initial position of the drones is 
# fixed and not optimal for the partitioning. 

import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Point


def kmeans_clustering(free_space, num_drones):
    ''' Perform K-means clustering on the free space to find optimal starting points for Voronoi partitioning.
        Parameters:
        - free_space: Shapely Polygon representing the free space in the map    
        - num_drones: Number of drones (clusters) to generate
        Returns:
        - centroids: List of (x, y) coordinates representing the centroids of the clusters
    '''
    # Sample points from the free space polygon
    minx, miny, maxx, maxy = free_space.bounds
    points = []
    while len(points) < 1000:  # Sample 1000 points for clustering
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        if free_space.contains(point):
            points.append([x, y])
    points = np.array(points)       
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_drones, random_state=0).fit(points)
    centroids = kmeans.cluster_centers_
    return centroids


