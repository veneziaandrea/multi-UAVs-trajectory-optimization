import numpy as np
import sys
from pathlib import Path
import math
from scipy.spatial import KDTree
from shapely.geometry import Polygon

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
from optimization.mpc import setup_MPC_QP, run_mpc_iteration 
from utils.drones import Drone
from optimization.optimization_plots import plot_results, animate_simulation

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
    max_iter = 1000
    num_neighbors = len(drone_positions) - 1

    # take the prediction horizon and time interval from config file
    config_path = ROOT / "configs" / "optimization_params.json"
    config = load_config(config_path)
    mpc_cfg = config["mpc"]
    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]

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
        vars_ = setup_MPC_QP(num_neighbors) # Setup with k=3 as discussed
        new_drone = Drone(id_d, drone_positions[id_d], waypoints_assigned, vars_, N)
        drones.append(new_drone)
        

    # --- MAIN MPC LOOP ---
    num_iter = 0
    dt = 0.05 # Should match your MPC config
    dist_threshold = 0.5 # Distance to mark a waypoint as 'seen' [m]
    dJ_thresh = 1e-6
    prev_total_cost = 1e6

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
            neighbor_trajs_array = np.stack(neighbor_trajs, axis=2)

            current_flags = drone.waypoints[:, 2]      # All rows, 3rd column
            current_coords = drone.waypoints[:, :2]    # All rows, first 2 columns

            # Run MPC
            accel, new_traj, current_cost_value = run_mpc_iteration(
                drone.mpc_vars, drone.state, 
                drone.waypoints, 
                drone.last_traj, neighbor_trajs_array, obs_tree
            )

            total_loop_cost += current_cost_value

            # Update physics and internal logs
            drone.drone_model(accel, dt)
            drone.check_waypoints()
            drone.log_telemetry(new_traj)
            drone.last_traj = new_traj
        
        if abs(prev_total_cost - total_loop_cost) < dJ_thresh:
            print("Converged!")
            break
            
        prev_total_cost = total_loop_cost
        num_iter += 1

print("optimization completed")

# This will open a window you can rotate to see the 3D flight paths
plot_results(drones, map3d.obstacles)
# 2. 2D Animation

# This will show the "movie" of the drones dodging obstacles
config_path = ROOT / "configs" / "demo_parameters.json"
config = load_config(config_path)
map_cfg = config["map"]
map_limits = [ map_cfg["x_bounds"],
            map_cfg["y_bounds"],
            map_cfg["z_bounds"]
            ]
animate_simulation(drones, map3d.obstacles, map_limits)



