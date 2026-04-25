import numpy as np
import sys
from pathlib import Path
import math
from scipy.spatial import KDTree
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Set the root directory and add the source directory to the Python path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, seed_everything
from environment.map_generation_v2 import Map3D
from utils.plot_initial_envronment import plot_initial_environment
from utils.kmeans import kmeans_clustering
from utils.plot_voronoi import plot_voronoi_partition
from partition.voronoi import Voronoi_Partition, assign_area, get_waypoints_in_partition
from optimization.mpc import setup_MPC_QP, run_mpc_iteration, setup_MPC_NLP, setup_test_MPC 
from utils.drones import Drone
from optimization.optimization_plots import plot_results, animate_simulation, plot_kinematics

def build_demo(config): 
    # Load environment configuration
    map_cfg = config["map"]
    uav_cfg = config["uav"]
    seed = config["seed"]

    # Generate the 3D map and get drone starting positions
    map3d, drone_positions = Map3D.generate_map3D(
        x_bounds=map_cfg["x_bounds"],
        y_bounds=map_cfg["y_bounds"],
        z_bounds=map_cfg["z_bounds"],
        num_obstacles=map_cfg["num_obstacles"],
        obstacle_radius_range=map_cfg["obstacle_radius_range"],
        obstacle_height_range=map_cfg["obstacle_height_range"],
        num_drones=uav_cfg["num_uavs"],                 
        spacing=uav_cfg["start_separation"],                       
        seed=seed,
    )
    print(f"Generated map with {len(map3d.obstacles)} obstacles and {len(drone_positions)} drone starting positions.")
    print("Map bounds:", map3d.x_bounds, map3d.y_bounds, map3d.z_bounds)
    print("Drone starting positions:")
    for i, pos in enumerate(drone_positions):
        print(f"  Drone {i+1}: {pos}")

    #Sto assumendo camera con un FOV diagonale di 84° (abbastanza largo) a un'altezza h dal terreno
    H_FOV = 71.1 #Horizontal FOV is equal to 71.1°
    V_FOV = 56.3 #Vertical FOV is equal to 56.3°
    h = 5 #Distanza del drone dal livello 0 del terreno

    L = 2* h * math.tan(math.radians(H_FOV/2)) # Largehzza immagine
    W = 2* h * math.tan(math.radians(V_FOV/2)) # Altezza immagine

    A_FOV = L * W
    A_map = map3d.x_bounds[1] * map3d.y_bounds[1]
    k = math.ceil(1.2 * A_map/A_FOV)

    print(f"Output image dimensions: {L} meters of width and {W} meters of height.")
    print(f"Map area = {A_map} m^2")
    print(f"FOV area = {A_FOV} m^2")
    print(f"k = {k}")

    waypoints = kmeans_clustering(
            map3d.free_space,
            k,
            seed=seed,
        )

    # --- Voronoi Partition --- 
    print("Computing Voronoi partition for the generated map and drone starting positions...")
    seeds_xy = kmeans_clustering(
        map3d.free_space,
        uav_cfg["num_uavs"],
        seed=seed,
        waypoints=waypoints
    )

    print("Voronoi seeds from k-means on free space:")
    for i, seed_xy in enumerate(seeds_xy):
        print(f"  Seed {i+1}: {seed_xy}")

    vor = Voronoi_Partition.build(
        cells=None,  # cells will be computed inside the build method
        seeds_xy=seeds_xy,
        map3D=map3d,
    )

    return map3d, vor, drone_positions, waypoints


if __name__ == "__main__":
    # Load configuration
    config_path = ROOT / "configs" / "demo_parameters.json"
    config = load_config(config_path)

    # Set random seed for reproducibility
    seed_everything(config["seed"])

    # Build the demo environment and get initial drone positions
    map3d, vor, drone_positions, waypoints = build_demo(config)
    
    # Visualize the Voronoi partition together with obstacles and initial drone positions
    plot_voronoi_partition(
        map3d,
        vor,
        drone_positions=drone_positions,
        waypoints=waypoints,
        title="Voronoi Partition of the Workspace",
    )

    
    # Extract 3D coordinates (x, y, and half the height for the z-center)
    obstacle_coords = np.array([[obs.x, obs.y, obs.height / 2.0] for obs in map3d.obstacles])
    # Create an array of radii to match the order of the tree
    obs_radii = np.array([obs.radius for obs in map3d.obstacles])
    # Create obstacles object in a way that is actually fast to use 
    obs_tree = KDTree(obstacle_coords)
    wp_tree = KDTree(waypoints)

    # SETUP MPC
    # Imposta ogni quante iterazioni vuoi vedere il report
    PRINT_INTERVAL = 10
    num_neighbors = len(drone_positions) - 1

    # take the prediction horizon and time interval from config file
    config_path = ROOT / "configs" / "optimization_params.json"
    config = load_config(config_path)
    mpc_cfg = config["mpc"]
    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]
    max_iter = mpc_cfg["max_iter"]

    # --- INITIALIZATION ---
    # Assume 'drones' is a list of objects containing 
    # the state, mpc_vars, and trajectory for each drone ID.
    drones = []
    drone_ids = [0] * len(drone_positions)
    for i in range(len(drone_positions)):
        drone_ids[i] = i

    # assign the waypoints to the associated drone
    assign_area(vor, drone_positions)

    # Add the 'seen' column to the global waypoints matrix ---
    # If waypoints is [N x 2], this makes it [N x 3]
    if waypoints.shape[1] == 2:
        seen_column = np.zeros((waypoints.shape[0], 1)) # Create column of 0s
        waypoints = np.hstack((waypoints, seen_column)) # Attach it

    # --- Assignment Phase ---
    for id_d in drone_ids:
        current_cell = vor.Voronoi_Cells[id_d]
        partition_shape = Polygon(current_cell.polygon)
        
        waypoints_assigned = get_waypoints_in_partition(waypoints, partition_shape)

        # Initialize Drone
        vars_ = setup_test_MPC(num_neighbors=0, enable_obstacles= True) 
        new_drone = Drone(id_d, drone_positions[id_d], waypoints_assigned, vars_, N)
        drones.append(new_drone)
        
    # --- MAIN MPC LOOP ---
    num_iter = 0
    dist_threshold = 0.8 # Distance to mark a waypoint as 'seen' [m]
    dJ_thresh = 1e-6
    prev_total_cost = 1e6
    num_iter_mpc_prev = 0
    t_solve_avg_prev = 0
    
    # Initialize the history tracker for plot 
    cost_history = {
        "total": [],
        "waypoints": [],
        "effort": [],
        "battery": [],
        "z_ref": [],
        "barrier": [] # Add any new components you defined in mpc.py
    }

    # --- MAIN MPC LOOP ---
    while num_iter <= max_iter:
        total_loop_cost = 0 # Track sum for the whole fleet
        # Check if ALL drones have finished their tasks
        all_done = all(np.all(d.waypoints[:, 2] == 1.0) for d in drones)
        
        if all_done:
            print(f"Mission accomplished in {num_iter} steps!")
            break
        
        for drone in drones:
            
            # Collect trajectories from OTHER drones
            neighbor_trajs = [d.last_traj for d in drones if d.id != drone.id]
    
            # Explicitly check if there is at least one neighbor to stack
            if len(neighbor_trajs) > 0:
                neighbor_trajs_array = np.stack(neighbor_trajs, axis=2)
            else:
                # Fallback for solitary drone
                neighbor_trajs_array = np.empty((3, N + 1, 0))

            current_flags = drone.waypoints[:, 2]      # All rows, 3rd column
            current_coords = drone.waypoints[:, :2]    # All rows, first 2 columns

            # Run MPC
            accel, new_traj, current_cost_value, n_iter_mpc, t_solve_mpc, cost_breakdown = run_mpc_iteration(
                drone.mpc_vars, drone.state, 
                drone.waypoints, 
                drone.last_traj, neighbor_trajs_array, obs_tree, 
                num_iter_mpc_prev, t_solve_avg_prev
            )

            # Record the costs for this iteration
            if current_cost_value != np.inf: # Don't track crashes/fallbacks
                cost_history["total"].append(current_cost_value)
                for key, val in cost_breakdown.items():
                    if key not in cost_history:
                        cost_history[key] = []
                    cost_history[key].append(val)

            total_loop_cost += current_cost_value

            # Update physics and internal logs
            drone.drone_model(accel, dt)
            drone.check_waypoints(dist_threshold)
            drone.log_telemetry(new_traj)
            drone.last_traj = new_traj
        
        
        if abs(prev_total_cost - total_loop_cost) < dJ_thresh:
            print("Converged!")
            break
           
        prev_total_cost = total_loop_cost
        num_iter += 1
        num_iter_mpc_prev = n_iter_mpc
        t_solve_avg_prev = t_solve_mpc

        if num_iter % PRINT_INTERVAL == 0:
            print(f"\n--- Step {num_iter} | Drone {drone.id} ---")
            print(f"Total Cost:  {current_cost_value:.2f}")
            print(f"  Waypoints: {cost_breakdown['waypoints']:.2f}")
            print(f"  Effort:    {cost_breakdown['effort']:.2f}")
            print(f"  Battery:   {cost_breakdown['battery']:.2f}")
            print(f"  Z-Ref:     {cost_breakdown['z_ref']:.2f}")
            print(f"  Barrier:     {cost_breakdown['barrier']:.2f}")


print("optimization completed")
print(f"Avg mpc loop solve time: {t_solve_avg_prev/5}")

# --- OPTIMIZATION RECAP ---
print("\n" + "="*40)
print("       OPTIMIZATION COST RECAP")
print("="*40)
print(f"{'Component':<15} | {'Mean':<10} | {'Max':<10}")
print("-" * 40)

# Calculate and print stats
for key, values in cost_history.items():
    if len(values) > 0:
        mean_val = np.mean(values)
        max_val = np.max(values)
        print(f"{key.capitalize():<15} | {mean_val:<10.2f} | {max_val:<10.2f}")

plt.figure(figsize=(12, 6))
for key, values in cost_history.items():
    if len(values) > 0:
        # Use log scale if total cost dwarfs the other components
        plt.plot(values, label=f"{key.capitalize()} (Max: {np.max(values):.1f})")

plt.title("MPC Cost Components Over Time", fontsize=14)
plt.xlabel("Iteration Step", fontsize=12)
plt.ylabel("Cost Value (Log Scale)", fontsize=12)
plt.yscale("log") # Log scale is highly recommended for optimization costs
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# This will open a window you can rotate to see the 3D flight paths
plot_results(drones, map3d.obstacles)

plot_kinematics(drones, dt)
# 2. 2D Animation

# This will show the "movie" of the drones moving
config_path = ROOT / "configs" / "demo_parameters.json"
config = load_config(config_path)
map_cfg = config["map"]
map_limits = [ map_cfg["x_bounds"],
            map_cfg["y_bounds"],
            map_cfg["z_bounds"]
            ]
animate_simulation(drones, map3d.obstacles, map_limits)



