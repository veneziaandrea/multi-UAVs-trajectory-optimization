import matplotlib.pyplot as plt
import numpy as np

def plot_initial_environment(map3d, drone_positions):
    '''plot the 3D environment with obstacles and drone starting positions'''
    min_x = map3d.x_bounds[0]
    max_x = map3d.x_bounds[1]
    min_y = map3d.y_bounds[0]
    max_y = map3d.y_bounds[1]
    min_z = map3d.z_bounds[0]
    max_z = map3d.z_bounds[1]

    fig = plt. figure(figsize=(max_x - min_x, max_y - min_y))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)

    # Plot obstacles
    for obs in map3d.obstacles:
        x = obs.x
        y = obs.y
        r = obs.radius
        h = obs.height

        # Create a cylinder for the obstacle
        z_range = np.linspace(0, obs.height, 10)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z_range)

        x_grid = r * np.cos(theta_grid) + x
        y_grid = r * np.sin(theta_grid) + y
        
        ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.5)

    # Plot drone starting positions
    drone_positions = np.array(drone_positions)
    ax.scatter(drone_positions[:, 0], drone_positions[:, 1], 0, c='blue', s=100)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Environment with Obstacles and Drone Starting Positions')
    plt.show()
    