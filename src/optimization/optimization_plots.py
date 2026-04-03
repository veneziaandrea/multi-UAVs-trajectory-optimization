from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_results(drones, obstacles):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. Plot Obstacles as Actual 3D Cylinders ---
    for obs in obstacles:
        # Create cylinder geometry
        z_range = np.linspace(0, obs.height, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z_range)
        x_grid = obs.radius * np.cos(theta_grid) + obs.x
        y_grid = obs.radius * np.sin(theta_grid) + obs.y
        
        # --- THE FIX ---
        # 1. Use facecolors='r' 
        # 2. Set shade=False to avoid the internal broadcast error
        ax.plot_surface(x_grid, y_grid, z_grid, 
                        color='r', 
                        alpha=0.3, 
                        shade=False)
        # Optional: Add a cap on top
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='r', alpha=0.1, linewidth=0.5)

    # --- 2. Plot Waypoints ---
    # We use a different color/marker to distinguish them from drones
    for drone in drones:
        wps = drone.waypoints # Assuming shape [N x 3] (x, y, seen)
        # Filter only active waypoints (seen == 0) or plot all with different alpha
        ax.scatter(wps[:, 0], wps[:, 1], 0.2, marker='*', s=100, 
                   color='gold', edgecolors='k', label=f'WPs Drone {drone.id}')

    # --- 3. Plot Drone Trajectories ---
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    for i, drone in enumerate(drones):
        path = np.array(drone.history_p)
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                    color=colors[i % len(colors)], linewidth=3, label=f'Drone {drone.id}')
            
            # Mark the current/final position
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], 
                       color=colors[i % len(colors)], s=50)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title("3D Swarm Mission Visualizer")
    
    # Set axis limits to match your map bounds [0, 40]
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 20) # Adjust based on your max flying height
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def animate_simulation(drones, obstacles, map_limits):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        # Set bounds
        ax.set_xlim(map_limits[0])
        ax.set_ylim(map_limits[1])
        ax.set_aspect('equal')
        
        # Draw Obstacles as real circles
        for obs in obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.4)
            ax.add_patch(circle)
            ax.text(obs.x, obs.y, "OBS", fontsize=8, ha='center')

        # Draw Waypoints
        for drone in drones:
            wps = drone.waypoints
            # Plot waypoints: Green if seen (1.0), Yellow if not (0.0)
            for wp in wps:
                color = 'green' if wp[2] == 1.0 else 'gold'
                ax.plot(wp[0], wp[1], marker='*', color=color, markersize=10)

        # Draw Drone positions and Predicted Horizons
        for drone in drones:
            if frame < len(drone.history_p):
                pos = drone.history_p[frame]
                pred = drone.history_predictions[frame]
                
                # Drone body
                ax.plot(pos[0], pos[1], 'ko', markersize=6)
                # The "Look Ahead" line from MPC
                ax.plot(pred[0, :], pred[1, :], color='blue', linestyle='--', alpha=0.4)
                
        ax.set_title(f"Time Step: {frame}")

    max_frames = max(len(d.history_p) for d in drones)
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50)
    plt.show()