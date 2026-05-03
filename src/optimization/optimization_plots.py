from pathlib import Path
from contextlib import contextmanager
import os
import sys
import webbrowser

PYTHON_BASE = Path(sys.base_prefix)
if "TCL_LIBRARY" not in os.environ:
    tcl_library = PYTHON_BASE / "tcl" / "tcl8.6"
    if tcl_library.exists():
        os.environ["TCL_LIBRARY"] = str(tcl_library)
if "TK_LIBRARY" not in os.environ:
    tk_library = PYTHON_BASE / "tcl" / "tk8.6"
    if tk_library.exists():
        os.environ["TK_LIBRARY"] = str(tk_library)

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
<<<<<<< Updated upstream

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
=======
from matplotlib.patches import Circle
import matplotlib as mpl

PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"
_backend_fallback_announced = False

def _patch_tk_foreground_restore():
    try:
        import matplotlib.backends._backend_tk as backend_tk
    except Exception:
        return

    original_context = backend_tk._restore_foreground_window_at_end
    if getattr(original_context, "_codex_safe_patch", False):
        return

    @contextmanager
    def safe_restore_foreground_window_at_end():
        foreground = None
        try:
            foreground = backend_tk._c_internal_utils.Win32_GetForegroundWindow()
        except ValueError:
            foreground = None

        try:
            yield
        finally:
            if foreground is not None and mpl.rcParams.get("tk.window_focus", False):
                try:
                    backend_tk._c_internal_utils.Win32_SetForegroundWindow(foreground)
                except ValueError:
                    pass

    safe_restore_foreground_window_at_end._codex_safe_patch = True
    backend_tk._restore_foreground_window_at_end = safe_restore_foreground_window_at_end

_patch_tk_foreground_restore()

def _switch_to_agg_backend():
    global _backend_fallback_announced
    if plt.get_backend().lower() != "agg":
        plt.close("all")
        plt.switch_backend("Agg")
    if not _backend_fallback_announced:
        print(f"Matplotlib GUI backend failed; saving plots to {PLOTS_DIR}")
        _backend_fallback_announced = True

def _safe_figure(*args, **kwargs):
    try:
        return plt.figure(*args, **kwargs)
    except ValueError as exc:
        if "PyCapsule_New called with null pointer" not in str(exc):
            raise
        _switch_to_agg_backend()
        return plt.figure(*args, **kwargs)

def _safe_subplots(*args, **kwargs):
    try:
        return plt.subplots(*args, **kwargs)
    except ValueError as exc:
        if "PyCapsule_New called with null pointer" not in str(exc):
            raise
        _switch_to_agg_backend()
        return plt.subplots(*args, **kwargs)

def _finalize_plot(fig, filename):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOTS_DIR / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {output_path}")
        _open_saved_plot(output_path)
        return

    plt.show()

def _open_saved_plot(path):
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)
        else:
            webbrowser.open(path.as_uri())
    except Exception as exc:
        print(f"Could not open saved plot automatically: {exc}")
>>>>>>> Stashed changes

def plot_results(drones, obstacles):
    fig = _safe_figure(figsize=(12, 9))
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
    _finalize_plot(fig, "trajectory_overview.png")

def animate_simulation(drones, obstacles, map_limits):
    fig, ax = _safe_subplots(figsize=(8, 8))
    
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
    if "agg" in plt.get_backend().lower():
        update(max_frames - 1)
        _finalize_plot(fig, "simulation_animation_preview.png")
        return

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50)
    fig._animation = ani
    _finalize_plot(fig, "simulation_animation_preview.png")

def plot_kinematics(drones, dt):
    """
    Plots the velocity and acceleration profiles of the drones over time,
    displaying their average values in the legend.
    """
<<<<<<< Updated upstream
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
=======
    num_drones = len(drones)
    # Create a 2-row grid: Row 0 for Velocity, Row 1 for Acceleration
    fig, axes = _safe_subplots(2, num_drones, figsize=(4 * num_drones, 8), sharex=True)
    
    # Ensure axes is a 2D array even if there is only one drone
    if num_drones == 1:
        axes = axes.reshape(2, 1)

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
    plt.tight_layout()
    _finalize_plot(fig, "kinematics.png")

def plot_tracking_error(drones, dt):
    '''Plot the position tracking error between the ideal MPC state and the PyBullet state for each drone.'''
    drones_with_tracking = [drone for drone in drones if len(drone.history_tracking_error) > 0]
    if not drones_with_tracking:
        print("No simulator tracking data available for plotting.")
        return

    fig, axes = _safe_subplots(
        len(drones_with_tracking),
        1,
        figsize=(11, 3.2 * len(drones_with_tracking)),
        sharex=True,
    )

    if len(drones_with_tracking) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(drones_with_tracking)))

    for ax, color, drone in zip(axes, colors, drones_with_tracking):
        errors = np.asarray(drone.history_tracking_error, dtype=float)
        time_axis = np.arange(len(errors)) * dt
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))

        ax.plot(time_axis, errors, color=color, linewidth=2.0, label="Position error")
        ax.axhline(mean_error, color=color, linestyle="--", alpha=0.7, label=f"Mean: {mean_error:.3f} m")
        ax.set_title(f"Drone {drone.id} Tracking Error", fontsize=12)
        ax.set_ylabel("Error [m]", fontsize=10)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize="small", title=f"Max: {max_error:.3f} m")

    axes[-1].set_xlabel("Time [s]", fontsize=10)
    plt.suptitle("MPC Ideal vs PyBullet Tracking Error", fontsize=14, y=1.02)
    plt.tight_layout()
    _finalize_plot(fig, "tracking_error.png")

def print_collision_report(collision_events):
    '''Print a summary and timeline of collision events detected during the MPC loop.'''
    print("\n" + "=" * 40)
    print("        PYBULLET COLLISION REPORT")
    print("=" * 40)

    if not collision_events:
        print("No collisions detected during the MPC loop.")
        return

    summary = {}
    for event in collision_events:
        drone_id = event["drone_id"]
        for label in event["labels"]:
            key = (drone_id, label)
            summary[key] = summary.get(key, 0) + 1

    print(f"Collision events detected: {len(collision_events)}")
    print("-" * 40)
    print("Counts by drone and contact:")
    for (drone_id, label), count in sorted(summary.items()):
        print(f"Drone {drone_id} vs {label:<18} : {count}")

    print("-" * 40)
    print("Event timeline:")
    for event in collision_events:
        labels = ", ".join(event["labels"])
        print(f"Step {event['step']:<5} | Drone {event['drone_id']} -> {labels}")

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
    
    fig, ax = _safe_subplots(figsize=(10, 10))
    
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
    _finalize_plot(fig, "coverage_map.png")

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
    fig, (ax1, ax2) = _safe_subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    drone_ids = []
    total_energies = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(drones))) # Distinct colors for each drone
    
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
    _finalize_plot(fig, "energy_consumption.png")

    return mean_energy

if __name__ == "__main__":
    print(
        "This module only defines plotting utilities. "
        "Run main.py to generate plots, or import the functions from another script."
    )
>>>>>>> Stashed changes
