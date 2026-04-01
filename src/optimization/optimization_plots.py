from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_results(history_p, waypoints, obstacles=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Actual Trajectory (History)
    hp = np.array(history_p)
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], 'b-', label='Actual Path', linewidth=2)

    # Plot Waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
               c='red', marker='x', s=100, label='Waypoints')

    # Plot Obstacles if provided (as spheres or points)
    if obstacles is not None:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], 
                   c='black', alpha=0.5, s=50, label='Obstacles')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    plt.title("Static MPC Trajectory")
    plt.show()

def animate_simulation(history_p, history_predictions, waypoints):
    """
    history_p: List of actual positions reached [N_steps x 3]
    history_predictions: List of full predicted trajectories at each step [N_steps x 3 x N+1]
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize plot elements
    path_line, = ax.plot([], [], [], 'b-', label='Actual Path')
    pred_line, = ax.plot([], [], [], 'r--', alpha=0.5, label='MPC Prediction')
    drone_dot, = ax.plot([], [], [], 'go', markersize=10)
    
    # Static elements
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', marker='x')

    # Set axis limits based on data
    all_data = np.array(history_p)
    ax.set_xlim(np.min(all_data[:,0])-1, np.max(all_data[:,0])+1)
    ax.set_ylim(np.min(all_data[:,1])-1, np.max(all_data[:,1])+1)
    ax.set_zlim(0, np.max(all_data[:,2])+2)

    def update(frame):
        # Update actual path up to current frame
        curr_path = np.array(history_p[:frame+1])
        path_line.set_data(curr_path[:, 0], curr_path[:, 1])
        path_line.set_3d_properties(curr_path[:, 2])
        
        # Update the drone's current position
        drone_dot.set_data([history_p[frame][0]], [history_p[frame][1]])
        drone_dot.set_3d_properties([history_p[frame][2]])
        
        # Update the MPC prediction horizon for this frame
        pred = history_predictions[frame] # Shape (3, N+1)
        pred_line.set_data(pred[0, :], pred[1, :])
        pred_line.set_3d_properties(pred[2, :])
        
        return path_line, pred_line, drone_dot

    ani = FuncAnimation(fig, update, frames=len(history_p), interval=100, blit=False)
    plt.legend()
    plt.show()