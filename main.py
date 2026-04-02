import numpy as np
import sys
from pathlib import Path
import math
from scipy.spatial import KDTree

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

    seen = 0 # flag to mark if a waypoint has been visited or not
    for wp in waypoints:
        wp.append(seen)

    # SETUP MPC
    max_iter = 1000
    num_neighbors = len(drone_positions) - 1

    # --- INITIALIZATION ---
    # Assume 'drones' is a list of objects or a dictionary containing 
    # the state, mpc_vars, and trajectory for each drone ID.
        # ... inside your initialization ...
    drones = []
    drone_ids = []
    for i in range(len(drone_positions)):
        drone_ids[i] = i
    
    # assign the waypoints to the associated drone
    assign_area(vor, drone_positions)
    for id_d in drone_ids:
        waypoints_assigned= get_waypoints_in_partition(waypoints, vor.Voronoi_Cells.drone_id[id_d])
    print(f'Aree assegnate: {vor.Voronoi_Cells}')
        
    for d_id in drone_ids:
        vars_ = setup_MPC_QP(waypoints_assigned[d_id], num_neighbors) # num neighbors is the number of other drones
        new_drone = Drone(d_id, drone_positions[d_id], waypoints_assigned[d_id], vars_, N)
        drones.append(new_drone)

    # --- MAIN MPC LOOP ---
    num_iter = 0
    dt = 0.1 # Should match your MPC config
    dist_threshold = 0.5 # Distance to mark a waypoint as 'seen' [m]

    while num_iter <= max_iter:
        
        for drone in drones:
            # 1. Collect trajectories from OTHER drones
            neighbor_trajs = [d.last_traj for d in drones if d.id != drone.id]
            neighbor_trajs_array = np.stack(neighbor_trajs, axis=2)

            # 2. Run MPC
            accel, new_traj = run_mpc_iteration(
                drone.mpc_vars, drone.state, 
                np.array([wp.seen for wp in drone.waypoints]),
                np.array([wp.coords for wp in drone.waypoints]),
                drone.last_traj, neighbor_trajs_array, obs_tree
            )

            # 3. Update physics and internal logs
            drone.drone_model(accel, dt)
            drone.check_waypoints()
            drone.log_telemetry(new_traj)
            drone.last_traj = new_traj
            
        num_iter += 1