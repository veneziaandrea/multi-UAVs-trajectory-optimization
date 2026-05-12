import numpy as np
import sys
from pathlib import Path
import math
from scipy.spatial import KDTree
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd

# Set the root directory and add the source directory to the Python path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIGS = ROOT / "configs"

from config import load_config, seed_everything
from environment.map_generation_v2 import Map3D
from utils.plot_initial_envronment import plot_initial_environment
from utils.kmeans import kmeans_clustering, sanitize_waypoints
from utils.plot_voronoi import plot_voronoi_partition
from partition.voronoi import Voronoi_Partition, assign_area, get_waypoints_in_partition
from optimization.mpc import run_mpc_iteration, setup_MPC_NLP, setup_test_MPC, setup_test_MPC_QP, run_swarm_simulation
from optimization.waypoints_sorter import sort_waypoints_tsp
from utils.drones import Drone
from optimization.optimization_plots import plot_results, animate_simulation, plot_kinematics, calculate_final_coverage, plot_offline_csv_comparison, plot_coverage_map, plot_energy_consumption, evaluate_trajectory_performance, save_metrics_to_csv

def spawn_swarm():
    """Generates a brand new set of drones"""

    fresh_drones = []
    for id_d in drone_ids:
        # Use your existing Voronoi and TSP logic
        current_cell = vor.Voronoi_Cells[id_d]
        partition_shape = Polygon(current_cell.polygon)
        waypoints_assigned = get_waypoints_in_partition(waypoints, partition_shape)
        waypoints_ordered = sort_waypoints_tsp(drone_positions[id_d], waypoints_assigned)

        vars_ = setup_test_MPC_QP(num_neighbors=num_neighbors, enable_obstacles=True) 
        new_drone = Drone(id_d, drone_positions[id_d], waypoints_ordered, vars_, N)
        
        new_drone.returning_home = False
        new_drone.is_parked = False
        new_drone.home_pos = drone_positions[id_d]
        fresh_drones.append(new_drone)
        
    return fresh_drones

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

    # COMPUTATION OF AREA COVERED BY EACH CAMERA AND DEDUCTION OF K WAYPOINTS
    # "FOV" is the diagonal FOV of the cameras (84° in this case)
    FOV_rad = math.radians(uav_cfg["camera_FOV"])
    aspect_ratio = uav_cfg["camera_aspect_ratio"]
    config_path = CONFIGS / "optimization_params.json"
    opt_config = load_config(config_path)
    h = opt_config["cost"]["z_ref"] # Distanza del drone dal livello 0 del terreno
    overlap = opt_config["constraints"]["overlap_factor"] # Overlap factor to increase redundancy and ensure a better coverage
    d = 2 * h * math.tan(FOV_rad/2) # Lunghezza diagonale dell'immagine delle camere

    W = d / math.sqrt(aspect_ratio**2 + 1) # Altezza immagine
    L = W * aspect_ratio # Largehzza immagine

    A_FOV = L * W # Area covered by each camera
    A_map = map3d.x_bounds[1] * map3d.y_bounds[1] # Total map area
    k = math.ceil(A_map/(A_FOV*(1 - overlap)))

    print(f"Output image dimensions: {L} meters of width and {W} meters of height.")
    print(f"Map area = {A_map} m^2")
    print(f"FOV area = {A_FOV} m^2")
    print(f"k = {k}")

    waypoints = kmeans_clustering(
            map3d.free_space,
            k,
            seed=seed,
        )   

    # 2. Pulizia Waypoint (Margine super safe di prova: safe_distance del JSON + 0.5m)
    safe_margin = opt_config["constraints"]["safe_distance"] + 0.5
    waypoints = sanitize_waypoints(waypoints, map3d.obstacles, safety_margin=safe_margin)

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

    return L, W, map3d, vor, drone_positions, waypoints

if __name__ == "__main__":
    # Load configuration
    config_path = ROOT / "configs" / "demo_parameters.json"
    config = load_config(config_path)

    map_limits = [config["map"]["x_bounds"], config["map"]["y_bounds"], config["map"]["z_bounds"]]
    csv_filepath = ROOT / "logs" / "switch_stats_seen_dist*2.csv"

    seed_list = [3, 27, 51, 13, 93, 42, 84, 79, 32, 25, 33, 41, 69, 55, 99, 1, 7, 77, 11, 62]
    # seed_list = [3]
    for test_seed in seed_list:
        # Set random seed for reproducibility
        # seed_everything(config["seed"])
        config["seed"] = test_seed
        seed_everything(test_seed)
        # Build the demo environment and get initial drone positions
        L, W, map3d, vor, drone_positions, waypoints = build_demo(config)
        '''
        # Visualize the Voronoi partition together with obstacles and initial drone positions
        plot_voronoi_partition(
            map3d,
            vor,
            drone_positions=drone_positions,
            waypoints=waypoints,
            title="Voronoi Partition of the Workspace",
        )
        '''

        # Extract 3D coordinates (x, y, and half the height for the z-center)
        obstacle_coords = np.array([[obs.x, obs.y, obs.height / 2.0] for obs in map3d.obstacles])
        # Create an array of radii to match the order of the tree
        obs_radii = np.array([obs.radius for obs in map3d.obstacles])
        # Create obstacles object in a way that is actually fast to use 
        obs_tree = KDTree(obstacle_coords)
        obstacles = map3d.obstacles

        wp_tree = KDTree(waypoints)

        # SETUP MPC
        # Imposta ogni quante iterazioni vuoi vedere il report
        PRINT_INTERVAL = 10
        num_neighbors = len(drone_positions) - 1

        # take the prediction horizon and time interval from config file
        config_path = ROOT / "configs" / "optimization_params.json"
        opt_config = load_config(config_path)
        mpc_cfg = opt_config["mpc"]
        N = mpc_cfg["prediction_horizon"]
        dt = mpc_cfg["timestep"]
        max_iter = mpc_cfg["max_iter"]
        current_overlap = opt_config["constraints"]["overlap_factor"]

        # --- INITIALIZATION ---
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

        # --- MAIN MPC LOOP ---
        dist_threshold = 0.5 # Distance to mark a waypoint as 'seen' [m]
        ego_accel_prev = 0
        t_solve_avg = 0
        early_swtiching_flag = False
        
        drones_normal = spawn_swarm()
        drones_normal, cost_hist_normal, avg_solve_time = run_swarm_simulation(
            drones_normal, dt, max_iter, opt_config, map3d.obstacles, obs_tree, dist_threshold, early_swtiching_flag, PRINT_INTERVAL
        )
        
        # Extract Normal Metrics
        normal_metrics = {"speed": [], "jerk": [], "miss": [], "state": [], "time": []}
        drone_labels = []
        for drone in drones_normal:
            report = evaluate_trajectory_performance(drone, dt)
            normal_metrics["speed"].append(report["avg_cornering_speed"])
            normal_metrics["jerk"].append(report["jerk"])
            normal_metrics["miss"].append(report["avg_miss_distance"])
            
            # --- EXACT TIME & STATE CALCULATION ---
            if drone.is_parked:
                status = "Success"
                
                # Calculate the magnitude of velocity at every timestep
                v_mag = np.linalg.norm(drone.history_v, axis=1)
                
                # Find the very last index where velocity was not zero
                if np.any(v_mag > 1e-5):
                    # np.max(np.nonzero) gets the last active index. Add 1 for total steps.
                    active_steps = np.max(np.nonzero(v_mag > 1e-5)) + 1
                else:
                    active_steps = 0
                    
                drone_time = active_steps * dt
            else:
                status = "Stuck"
                # If it got stuck, it spent the entire simulation flying/trying
                drone_time = len(drone.history_p) * dt

            normal_metrics["state"].append(status)
            normal_metrics["time"].append(drone_time) # Append the exact time
            drone_labels.append(f"Drone {drone.id}")
            
        # Calculate global coverage 
        res = 0.2
        normal_cov, _ = calculate_final_coverage(drones_normal, map_limits, L, W, res)
        
        # Save 
        save_metrics_to_csv(csv_filepath, test_seed, current_overlap, "Normal", 
                            drone_labels, normal_metrics, normal_cov)

        # ==========================================
        # RUN 2: EARLY SWITCHING
        # ==========================================
        print("\n" + "="*50)
        print(" STARTING RUN 2: EARLY SWITCHING")
        print("="*50)

        early_swtiching_flag = True
        
        drones_early = spawn_swarm() 
        drones_early, cost_hist_early, early_avg_solve_time = run_swarm_simulation(
            drones_early, dt, max_iter, opt_config, map3d.obstacles, obs_tree, dist_threshold, early_swtiching_flag, PRINT_INTERVAL
        )
        
        # Extract Early Metrics
        early_metrics = {"speed": [], "jerk": [], "miss": [], "state": [], "time": []}
        drone_labels = []
        for drone in drones_early:
            report = evaluate_trajectory_performance(drone, dt)
            early_metrics["speed"].append(report["avg_cornering_speed"])
            early_metrics["jerk"].append(report["jerk"])
            early_metrics["miss"].append(report["avg_miss_distance"])
            
            # ---> 2. EXACT TIME & STATE CALCULATION <---
            if drone.is_parked:
                status = "Success"
                
                # Calculate the magnitude of velocity at every timestep
                v_mag = np.linalg.norm(drone.history_v, axis=1)
                
                # Find the very last index where velocity was not zero
                if np.any(v_mag > 1e-5):
                    # np.max(np.nonzero) gets the last active index. Add 1 for total steps.
                    active_steps = np.max(np.nonzero(v_mag > 1e-5)) + 1
                else:
                    active_steps = 0
                    
                drone_time = active_steps * dt
            else:
                status = "Stuck"
                # If it got stuck, it spent the entire simulation flying/trying
                drone_time = len(drone.history_p) * dt

            early_metrics["state"].append(status)
            early_metrics["time"].append(drone_time) # Append the exact time
            drone_labels.append(f"Drone {drone.id}")
            
        # Calculate global coverage 
        res = 0.2
        early_cov, _ = calculate_final_coverage(drones_early, map_limits, L, W, res)
        
        # Save (notice global_time is removed from the arguments)
        save_metrics_to_csv(csv_filepath, test_seed, current_overlap, "Early", 
                            drone_labels, early_metrics, early_cov)

        # ==========================================
        # PHASE 3: AUTOMATED COMPARISON PLOT
        # ==========================================
        # print("\nGenerating Final Performance Comparison...")
        
        early_time = max(early_metrics["time"])
        normal_time = max(normal_metrics["time"])
        '''
        plot_algorithm_comparison(
            drone_ids=drone_labels,
            data_a=early_metrics,  
            name_a="Early Switching",  
            globals_a={"time": early_time, "coverage": early_cov},
            
            data_b=normal_metrics, 
            name_b="Normal Switching", 
            globals_b={"time": normal_time, "coverage": normal_cov}
        )
        '''

    # plot_offline_csv_comparison(csv_filepath)
      
    # (Optional: Show the 3D map or animation for the Early Switching run)
    plot_results(drones_early, map3d.obstacles)

    # Plot the apllied inputs and velocities
    plot_kinematics(drones_early, dt)

    animate_simulation(drones_early, map3d.obstacles, map_limits)

    res = 0.2 # Resolution
    final_coverage_pct, coverage_grid = calculate_final_coverage(drones_early, map_limits, L, W, res)
    print(f"Final Map Coverage: {final_coverage_pct:.2f}%")
    print(f"Early Switch Average solve time: {early_avg_solve_time} ms")
    print(f"Normal Switch Average solve time: {avg_solve_time} ms")