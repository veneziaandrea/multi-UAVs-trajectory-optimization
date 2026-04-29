from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle

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
        
        ax.plot_surface(x_grid, y_grid, z_grid, 
                        color='r', 
                        alpha=0.3, 
                        shade=False)
        # Optional: Add a cap on top
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='r', alpha=0.1, linewidth=0.5)

    # --- 2. Plot Waypoints ---
    for drone in drones:
        wps = drone.waypoints 
        # Placed at Z=0.2 so they don't clip into the floor
        ax.scatter(wps[:, 0], wps[:, 1], 0.2, marker='*', s=100, 
                   color='gold', edgecolors='k', label=f'WPs Drone {drone.id}')

    # --- 3. Plot Drone Trajectories ---
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    for i, drone in enumerate(drones):
        path = np.array(drone.history_p)
        if len(path) > 0:
            # Plot the actual 3D path
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                    color=colors[i % len(colors)], linewidth=3, label=f'Drone {drone.id}')
            
            # Mark the current/final position
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], 
                       color=colors[i % len(colors)], s=50)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title("3D Swarm Mission Visualizer (Bird's-Eye View)")
    
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 20) 

    # --- THE FIX: ORTHOGRAPHIC BIRD'S EYE VIEW ---
    # 1. Orthographic projection removes perspective distortion (leaning cylinders)
    ax.set_proj_type('ortho') 
    
    # 2. elev=90 points the camera straight down. azim=-90 aligns X/Y to a standard 2D grid.
    ax.view_init(elev=90, azim=-90)

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
                
                # Drone body
                ax.plot(pos[0], pos[1], 'ko', markersize=6)
                
                # --- NEW: Safely fetch the prediction ---
                if frame < len(drone.history_predictions):
                    pred = drone.history_predictions[frame]
                elif len(drone.history_predictions) > 0:
                    # If we run out of predictions, just show the final parked one
                    pred = drone.history_predictions[-1] 
                else:
                    pred = None
                
                # The "Look Ahead" line from MPC
                if pred is not None:
                    ax.plot(pred[0, :], pred[1, :], color='blue', linestyle='--', alpha=0.4)
                
        ax.set_title(f"Time Step: {frame}")

    max_frames = max(len(d.history_p) for d in drones)
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50)
    plt.show()

def plot_kinematics(drones, dt):
    """
    Plots the velocity and acceleration profiles of the drones over time,
    displaying their average values in the legend.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']

    for i, drone in enumerate(drones):
        path = np.array(drone.history_p)
        
        # Need at least 3 points to calculate acceleration
        if len(path) > 2:
            # 1. Calculate Velocity (Derivative of Position)
            velocities = np.diff(path, axis=0) / dt
            v_mag = np.linalg.norm(velocities, axis=1)
            time_v = np.arange(len(v_mag)) * dt
            avg_v = np.mean(v_mag)
            
            ax1.plot(time_v, v_mag, color=colors[i % len(colors)], linewidth=2,
                     label=f'Drone {drone.id} (Avg: {avg_v:.2f} m/s)')

            # Formatting Velocity Plot
            ax1.set_title("Drone Velocity Profile", fontsize=14)
            ax1.set_ylabel("Velocity Magnitude [m/s]", fontsize=12)
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            ax1.legend(loc="upper right")

            # 2. Plot Acceleration (From the solver directly)
            # Make sure we have acceleration data logged
            if hasattr(drone, 'history_a') and len(drone.history_a) > 0:
                true_accel = np.array(drone.history_a)
                a_mag = np.linalg.norm(true_accel, axis=1)
                
                # Match the time array length
                time_a = np.arange(len(a_mag)) * dt
                avg_a = np.mean(a_mag)
                
                ax2.plot(time_a, a_mag, color=colors[i % len(colors)], linewidth=2,
                            label=f'Drone {drone.id} (Avg: {avg_a:.2f} m/s²)')
                ax2.set_title("Drone Acceleration Profile", fontsize = 14)
                 # Formatting Acceleration Plot
                ax2.set_ylabel("Acceleration Magnitude [m/s]", fontsize=12)
                ax2.grid(True, which="both", ls="--", alpha=0.5)
                ax2.legend(loc="upper right")

def calculate_final_coverage(drones, map_limits, L, W, res=0.5):
    x_range = np.arange(map_limits[0][0], map_limits[0][1], res)
    y_range = np.arange(map_limits[1][0], map_limits[1][1], res)
    grid = np.zeros((len(x_range), len(y_range)), dtype=bool)
    
    # Pre-calculate grid coordinates for vectorized checking
    X_grid, Y_grid = np.meshgrid(x_range, y_range, indexing='ij')

    for drone in drones:
        # Convert history to arrays
        pos_hist = np.array(drone.history_p) 
        # Calculate headings from positions if velocity history isn't explicit
        # or use drone.history_v if you logged it[cite: 14]
        
        for i in range(1, len(pos_hist)):
            curr_p = pos_hist[i]
            prev_p = pos_hist[i-1]
            
            # 1. Determine rotation (heading)
            dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
            theta = np.arctan2(dy, dx) if (abs(dx) > 1e-3 or abs(dy) > 1e-3) else 0
            
            # 2. Local frame transformation
            # Translate grid so drone is at origin
            dx_grid = X_grid - curr_p[0]
            dy_grid = Y_grid - curr_p[1]
            
            # Rotate grid points by -theta to align with drone axes
            x_local = dx_grid * np.cos(theta) + dy_grid * np.sin(theta)
            y_local = -dx_grid * np.sin(theta) + dy_grid * np.cos(theta)
            
            # 3. Apply Coverage Mask
            mask = (np.abs(x_local) <= L/2) & (np.abs(y_local) <= W/2)
            grid |= mask # Logical OR to accumulate coverage

    coverage_pct = (np.sum(grid) / grid.size) * 100

    return coverage_pct, grid

def plot_coverage_map(coverage_grid, map_limits, res, obstacles, drones):
    """
    Disegna la mappa di copertura evidenziando le aree viste e non viste.
    """
    x_range = np.arange(map_limits[0][0], map_limits[0][1], res)
    y_range = np.arange(map_limits[1][0], map_limits[1][1], res)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Disegna la Coverage Heatmap
    # coverage_grid è una matrice Booleana (True=Visto, False=Non Visto).
    # Usiamo la colormap 'RdYlGn' (Red-Yellow-Green): Rosso=0 (Non visto), Verde=1 (Visto)
    # Trasponiamo la griglia (.T) perché imshow inverte gli assi X e Y per default.
    cmap = plt.get_cmap('RdYlGn')
    ax.imshow(
        coverage_grid.T, 
        origin='lower', 
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
        cmap=cmap, 
        alpha=0.5 # Trasparenza per vedere la griglia sotto
    )
    
    # 2. Disegna gli Ostacoli
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color='black', alpha=0.7)
        ax.add_patch(circle)
        
    # 3. Sovrapponi le Traiettorie dei Droni
    # Questo aiuta a capire PERCHÉ un'area è stata coperta o mancata
    colors = ['blue', 'cyan', 'magenta', 'yellow', 'white']
    for i, drone in enumerate(drones):
        if len(drone.history_p) > 0:
            traj = np.array(drone.history_p)
            ax.plot(
                traj[:, 0], traj[:, 1], 
                color=colors[i % len(colors)], 
                linewidth=1.5, 
                label=f'Drone {drone.id} Path'
            )
            
    # 4. Formattazione del Grafico
    ax.set_title("UAV Swarm Coverage Map\n(Green = Seen, Red = Unseen)", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_xlim(map_limits[0][0], map_limits[0][1])
    ax.set_ylim(map_limits[1][0], map_limits[1][1])
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
