from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
import numpy as np
import os 
import csv
from pathlib import Path

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
    Plots velocity (top row) and acceleration (bottom row) in a grid,
    where each column corresponds to one drone.
    """
    num_drones = len(drones)
    # Create a 2-row grid: Row 0 for Velocity, Row 1 for Acceleration
    fig, axes = plt.subplots(2, num_drones, figsize=(4 * num_drones, 8), sharex=True)
    
    # Ensure axes is a 2D array even if there is only one drone
    if num_drones == 1:
        axes = axes.reshape(2, 1)

    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']

    for i, drone in enumerate(drones):
        path = np.array(drone.history_p)
        col_color = colors[i % len(colors)]
        
        # 1. Calculate and Plot Velocity (Top Row)
        ax_v = axes[0, i]
        if len(path) > 1:
            velocities = np.diff(path, axis=0) / dt
            v_mag = np.linalg.norm(velocities, axis=1)
            time_v = np.arange(len(v_mag)) * dt
            avg_v = np.mean(v_mag)
            
            ax_v.plot(time_v, v_mag, color=col_color, linewidth=2,
                      label=f'Avg: {avg_v:.2f} m/s')
            
            ax_v.set_title(f"Drone {drone.id} Velocity", fontsize=12)
            ax_v.set_ylabel("Vel [m/s]", fontsize=10)
            ax_v.grid(True, ls="--", alpha=0.5)
            ax_v.legend(loc="upper right", fontsize='small')

        # 2. Plot Acceleration (Bottom Row)
        ax_a = axes[1, i]
        if hasattr(drone, 'history_a') and len(drone.history_a) > 0:
            true_accel = np.array(drone.history_a)
            a_mag = np.linalg.norm(true_accel, axis=1)
            time_a = np.arange(len(a_mag)) * dt
            avg_a = np.mean(a_mag)
            
            ax_a.plot(time_a, a_mag, color=col_color, linewidth=2,
                      label=f'Avg: {avg_a:.2f} m/s²')
            
            ax_a.set_title(f"Drone {drone.id} Acceleration", fontsize=12)
            ax_a.set_ylabel("Acc [m/s²]", fontsize=10)
            ax_a.set_xlabel("Time [s]", fontsize=10)
            ax_a.grid(True, ls="--", alpha=0.5)
            ax_a.legend(loc="upper right", fontsize='small')

    plt.tight_layout()
    plt.show()

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

def calculate_trajectory_energy(history_v, history_a, dt, mass=1.0):
    """
    Estimates the power (Watts) and total energy (Joules) of a drone's flight.
    
    history_v: List or (N, 3) numpy array of velocities [vx, vy, vz]
    history_a: List or (N, 3) numpy array of accelerations [ax, ay, az]
    dt: Timestep of the simulation (e.g., 0.05s)
    mass: Mass of the drone in kg
    """
    v_arr = np.array(history_v)
    a_arr = np.array(history_a)
    
    # Constants
    g = np.array([0.0, 0.0, 9.81])  # Gravity vector [m/s^2]
    rho = 1.225                     # Air density [kg/m^3]
    prop_area = 0.05                # Total swept area of propellers [m^2]
    c_drag = 0.25                   # Empirical drag coefficient of the drone frame
    efficiency = 0.7                # Motor/Propeller electrical efficiency
    
    # HOVER POWER (Constant)
    # Based on Momentum Theory: P = T * sqrt(T / 2*rho*A)
    thrust_hover = mass * 9.81
    p_hover = thrust_hover * np.sqrt(thrust_hover / (2 * rho * prop_area))
    
    # MECHANICAL POWER (Dynamic)
    # Force required to generate the commanded acceleration AND fight gravity
    # F = m * (a + g)
    F_thrust = mass * (a_arr + g)
    
    # Power is the dot product of Force and Velocity (P = F dot V)
    # Using np.sum with axis=1 does a row-wise dot product
    p_mech = np.sum(F_thrust * v_arr, axis=1)
    
    # NOTE: Multirotors cannot efficiently regenerate power when braking. 
    # If p_mech is negative, the energy is mostly lost as heat. We clamp it to 0.
    p_mech = np.maximum(p_mech, 0)
    
    # DRAG POWER (Dynamic)
    # P_drag = drag_coeff * |v|^3
    v_norm = np.linalg.norm(v_arr, axis=1)
    p_drag = c_drag * (v_norm ** 3)
    
    # --- TOTALS ---
    # Divide mechanical/drag work by motor efficiency to get electrical draw
    power_watts = p_hover + ((p_mech + p_drag) / efficiency)
    
    # Energy (Joules) = Power * Time
    total_energy_joules = np.sum(power_watts) * dt
    
    return power_watts, total_energy_joules

def plot_energy_consumption(drones, dt, mass=1.0):
    """
    Plots instantaneous power and total energy for the entire swarm.
    """
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    drone_ids = []
    total_energies = []
    # Safe, universal way to pull the colormap
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(drones))) # Distinct colors for each drone
    
    for i, drone in enumerate(drones):
        # Safety check
        if not hasattr(drone, 'history_v') or len(drone.history_v) == 0:
            print(f"Warning: No velocity data for Drone {drone.id}. Skipping plot.")
            continue
            
        power_watts, total_energy = calculate_trajectory_energy(
            drone.history_v, drone.history_a, dt, mass
        )
        
        time_axis = np.arange(len(power_watts)) * dt
        
        # --- TOP PLOT: Instantaneous Power (Watts) ---
        ax1.plot(time_axis, power_watts, label=f'Drone {drone.id} ({total_energy:.0f} J)', 
                 color=colors[i], linewidth=2, alpha=0.8)
        
        # Store data for the bar chart
        drone_ids.append(f"Drone {drone.id}")
        total_energies.append(total_energy)

   # Calculate mean energy using the existing total_energies list
    mean_energy = sum(total_energies) / len(total_energies)
        
    # Formatting Top Plot
    ax1.set_title("Swarm Instantaneous Power Consumption", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (Seconds)", fontsize=12)
    ax1.set_ylabel("Power (Watts)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # --- BOTTOM PLOT: Total Energy Bar Chart (Joules) ---
    if total_energies:
        bars = ax2.bar(drone_ids, total_energies, color=colors, alpha=0.8, edgecolor='black')
        
        ax2.set_title("Total Energy Consumed Per Drone", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Total Energy (Joules)", fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add the exact Joule value on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(total_energies)*0.05), 
                     f'{yval:.0f} J', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add the mean energy dashed line and legend
        ax2.axhline(y=mean_energy, color='red', linestyle='--', linewidth=2, label=f'Mean Energy: {mean_energy:.0f} J')
        ax2.legend(loc='upper right')
             
    plt.tight_layout()
    plt.show()

    return mean_energy

def evaluate_trajectory_performance(drone, dt):
    """
    Analyzes the flight logs to quantify the Elastic Band effect and Coverage Trade-off.
    """
    p_arr = np.array(drone.history_p)
    v_arr = np.array(drone.history_v)
    a_arr = np.array(drone.history_a)
    
    # Exclude the "home" waypoint if it was added dynamically at the end
    mission_wps = drone.waypoints[:-1] if drone.returning_home else drone.waypoints
    
    cornering_speeds = []
    miss_distances = []
    
    # 1. EVALUATE WAYPOINTS (Cornering Speed & Miss Distance)
    for i in range(mission_wps.shape[0]):
        target_wp = mission_wps[i, :3]
        
        # Calculate distance from drone to this waypoint at every timestep
        distances = np.linalg.norm(p_arr - target_wp, axis=1)
        
        # Find the timestep where the drone was closest to the waypoint
        idx_closest = np.argmin(distances)
        
        # Record the Miss Distance (Coverage Trade-off)
        min_dist = distances[idx_closest]
        miss_distances.append(min_dist)
        
        # Record the Cornering Speed (Kinetic Energy Retention)
        speed_at_wp = np.linalg.norm(v_arr[idx_closest])
        cornering_speeds.append(speed_at_wp)

    # 2. EVALUATE SMOOTHNESS (Cumulative Jerk)
    # Jerk is the derivative of acceleration: (a[k] - a[k-1]) / dt
    jerk_vectors = np.diff(a_arr, axis=0) / dt
    jerk_magnitudes = np.linalg.norm(jerk_vectors, axis=1)
    
    # Cumulative sum of squared jerk
    total_jerk_effort = np.sum(jerk_magnitudes**2) * dt
    
    # 3. EVALUATE MISSION TIME
    total_flight_time = len(p_arr) * dt
    
    # --- PRINT REPORT ---
    print(f"\n=== FLIGHT DYNAMICS REPORT (Drone {drone.id}) ===")
    print(f"Total Flight Time:      {total_flight_time:.2f} seconds")
    print(f"Cumulative Jerk Effort: {total_jerk_effort:.2f} m^2/s^5")
    print("-" * 40)
    print("Waypoint Analysis:")
    for i in range(len(cornering_speeds)):
        print(f"  WP {i+1}: Missed Center by {miss_distances[i]:.2f}m | Cornering Speed: {cornering_speeds[i]:.2f} m/s")
        
    avg_miss = np.mean(miss_distances)
    avg_speed = np.mean(cornering_speeds)
    print("-" * 40)
    print(f"AVERAGE MISS DISTANCE:  {avg_miss:.2f} meters")
    print(f"AVERAGE CORNERING SPD:  {avg_speed:.2f} m/s")
    
    return {
        "time": total_flight_time,
        "jerk": total_jerk_effort,
        "avg_miss_distance": avg_miss,
        "avg_cornering_speed": avg_speed
    }

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_offline_csv_comparison(csv_filepath):
    """
    Reads the offline CSV flight logs, averages the performance across all map seeds,
    and generates a 3-panel grouped bar chart comparing the two algorithms.
    """
    # 1. Load the database
    df = pd.read_csv(csv_filepath)
    
    # Optional but highly recommended: Filter out "Stuck" drones 
    # so they don't poison the average flight times and speeds
    df_success = df[df['Final_State'] == 'Success']
    
    # 2. Identify the algorithms being compared (e.g., 'Normal' and 'Early')
    algos = df_success['Algorithm'].unique()
    if len(algos) != 2:
        print(f"Warning: Expected exactly 2 algorithms, found {len(algos)}: {algos}")
        return
    name_a, name_b = algos[0], algos[1]
    
    # 3. Calculate Global Map Averages (Coverage)
    # Since coverage is duplicated across drone rows for a single map, drop duplicates first
    global_df = df_success[['Map_Seed', 'Algorithm', 'Coverage_pct']].drop_duplicates()
    global_stats = global_df.groupby('Algorithm')['Coverage_pct'].mean().to_dict()
    
    # 4. Calculate Per-Drone Averages
    # Group by Algorithm and Drone_ID, then calculate the mean across all Map Seeds
    drone_stats = df_success.groupby(['Algorithm', 'Drone_ID'])[['Speed_m_s', 'Jerk_m2_s5', 'Flight_Time_s']].mean().reset_index()
    
    # Split the data back into the two algorithms and sort to ensure Drone 0 to N align
    data_a = drone_stats[drone_stats['Algorithm'] == name_a].sort_values('Drone_ID')
    data_b = drone_stats[drone_stats['Algorithm'] == name_b].sort_values('Drone_ID')
    
    drone_ids = data_a['Drone_ID'].tolist()
    
    # ==========================================
    # PLOTTING
    # ==========================================
    x = np.arange(len(drone_ids))
    width = 0.35
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Global Title (Displays the Averaged Coverage)
    title_str = (f"Offline Trajectory Analysis: {name_a} vs {name_b} (Averaged across maps)\n"
                 f"Mean Coverage: {name_a} ({global_stats.get(name_a, 0):.2f}%) vs "
                 f"{name_b} ({global_stats.get(name_b, 0):.2f}%)")
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.95)
    
    color_a, color_b = '#2ca02c', '#1f77b4' # Green vs Blue
    
    # --- Subplot 1: Cornering Speed ---
    rects1_a = ax1.bar(x - width/2, data_a['Speed_m_s'], width, label=name_a, color=color_a, edgecolor='black')
    rects1_b = ax1.bar(x + width/2, data_b['Speed_m_s'], width, label=name_b, color=color_b, edgecolor='black')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Average Cornering Speed (Higher is usually more efficient)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(drone_ids)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Subplot 2: Cumulative Jerk ---
    rects2_a = ax2.bar(x - width/2, data_a['Jerk_m2_s5'], width, color=color_a, edgecolor='black')
    rects2_b = ax2.bar(x + width/2, data_b['Jerk_m2_s5'], width, color=color_b, edgecolor='black')
    ax2.set_ylabel('Jerk ($m^2/s^5$)')
    ax2.set_title('Average Cumulative Jerk (Lower means less actuator wear)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(drone_ids)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Subplot 3: Flight Time ---
    rects3_a = ax3.bar(x - width/2, data_a['Flight_Time_s'], width, color=color_a, edgecolor='black')
    rects3_b = ax3.bar(x + width/2, data_b['Flight_Time_s'], width, color=color_b, edgecolor='black')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Average Flight Time per Drone')
    ax3.set_xticks(x)
    ax3.set_xticklabels(drone_ids)
    
    # Dynamic Y-Limit to zoom in on the exact time range differences
    min_time = min(data_a['Flight_Time_s'].min(), data_b['Flight_Time_s'].min())
    max_time = max(data_a['Flight_Time_s'].max(), data_b['Flight_Time_s'].max())
    padding = (max_time - min_time) * 0.5 if max_time != min_time else 5
    ax3.set_ylim(max(0, min_time - padding), max_time + padding)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Utility: Add Data Labels ---
    def autolabel(rects, ax, format_str='{:.2f}'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Apply labels
    autolabel(rects1_a, ax1)
    autolabel(rects1_b, ax1)
    autolabel(rects2_a, ax2, '{:.0f}')
    autolabel(rects2_b, ax2, '{:.0f}')
    autolabel(rects3_a, ax3)
    autolabel(rects3_b, ax3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def save_metrics_to_csv(filepath, map_seed, overlap_factor, mode, drone_ids, metrics, global_cov):
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.is_file()
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "Map_Seed", "Overlap_Factor", "Algorithm", "Drone_ID", 
                "Final_State", "Speed_m_s", "Jerk_m2_s5", "Miss_Distance_m", 
                "Energy_Joules", "Flight_Time_s", 
                "Collisions", # <--- NEW COLUMN
                "Coverage_pct"
            ])
            
        for i in range(len(drone_ids)):
            writer.writerow([
                map_seed, overlap_factor, mode, drone_ids[i], metrics["state"][i],
                round(metrics["speed"][i], 4),
                round(metrics["jerk"][i], 4),
                round(metrics["miss"][i], 4),
                round(metrics["energy"][i], 2),
                round(metrics["time"][i], 2),
                metrics["collisions"][i],  # <--- NEW DATA
                round(global_cov, 2)
            ])