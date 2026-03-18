import numpy as np
import matplotlib.pyplot as plt

# PCA for dimensionality reduction and clustering for waypoint generation. 
# This module can be used to analyze the free space and identify key waypoints for trajectory optimization.
# PCA can help in reducing the dimensionality of the free space representation, making it easier to identify clusters of free space and generate waypoints for the drones to follow.

def pca(vertices):
    '''Perform PCA on the Voronoi cells to identify key waypoints for trajectory optimization
    
    Input:
    - vertices: Array of vertices representing the free space
    Output:
    - waypoints: List of key waypoints identified through PCA
    '''
    
    # Centroid calculation
    centroid = np.mean(vertices, axis=0)

    # Center the data
    centered_data = vertices - centroid     # Centering the data around the mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False) 

    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # taking the principal components
    idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvectors = eigenvectors[:, idx]    # Sort eigenvectors according to eigenvalues
    eigenvalues = eigenvalues[idx]

    # Select the top k eigenvectors (for 2D, we can take the top 2)
    k = 2
    principal_components = eigenvectors[:, :k]
    waypoints = np.dot(centered_data, principal_components) + centroid  # Project back to original space

    return waypoints


def plot_pca(waypoints):
    '''Visualize the waypoints identified through PCA'''
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', marker='o', label='PCA Waypoints')
    plt.title("PCA Waypoints (2D)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.show(block=False)

    while True:
        plt.pause(1)
