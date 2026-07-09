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
from logs.read_stats import plot_offline_csv_comparison 
from utils.plot_initial_envronment import plot_initial_environment
from utils.kmeans import kmeans_clustering, sanitize_waypoints
from utils.plot_voronoi import plot_voronoi_partition
from partition.voronoi import Voronoi_Partition, assign_area, get_waypoints_in_partition
from optimization.mpc import run_mpc_iteration, setup_MPC_QP, run_swarm_simulation
from optimization.waypoints_sorter import sort_waypoints_tsp
from utils.drones import Drone
from optimization.optimization_plots import plot_results, animate_simulation, plot_kinematics, calculate_final_coverage, plot_coverage_map, plot_energy_consumption, evaluate_trajectory_performance, save_metrics_to_csv

def spawn_swarm():
    """Generates a brand new set of drones"""

    fresh_drones = []
    for id_d in drone_ids:
        # Drone creation and Voronoi cell and waypoints assignment and ordering
        current_cell = vor.Voronoi_Cells[id_d]
        partition_shape = Polygon(current_cell.polygon)
        waypoints_assigned = get_waypoints_in_partition(waypoints, partition_shape)
        waypoints_ordered = sort_waypoints_tsp(drone_positions[id_d], waypoints_assigned)

        vars_ = setup_MPC_QP(num_neighbors=num_neighbors, enable_obstacles=True) 
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
    #print("Map bounds:", map3d.x_bounds, map3d.y_bounds, map3d.z_bounds)
    #print("Drone starting positions:")
    #for i, pos in enumerate(drone_positions):
    #    print(f"  Drone {i+1}: {pos}")

    # COMPUTATION OF AREA COVERED BY EACH CAMERA AND DEDUCTION OF K WAYPOINTS
    # "FOV" is the diagonal FOV of the cameras
    FOV_rad = math.radians(uav_cfg["camera_FOV"])
    aspect_ratio = uav_cfg["camera_aspect_ratio"]
    config_path = CONFIGS / "optimization_params.json"
    opt_config = load_config(config_path)
    h = opt_config["cost"]["z_ref"] # Distance of the drone from the terrain 
    overlap = opt_config["constraints"]["overlap_factor"] # Overlap factor to increase redundancy and ensure a better coverage
    d = 2 * h * math.tan(FOV_rad/2) # Diagonal length of the cameras' images

    W = d / math.sqrt(aspect_ratio**2 + 1) # Image height
    L = W * aspect_ratio # Image width

    A_FOV = L * W # Area covered by each camera
    A_map = map3d.x_bounds[1] * map3d.y_bounds[1] # Total map area
    k = math.ceil(A_map/(A_FOV*(1 - overlap))) # number of waypoints

    # print(f"Output image dimensions: {L} meters of width and {W} meters of height.")
    # print(f"Map area = {A_map} m^2")
    # print(f"FOV area = {A_FOV} m^2")
    print(f"Number of waypoints k = {k}")

    waypoints = kmeans_clustering(
            map3d.free_space,
            k,
            seed=seed,
        )   

    # Remove the waypoints too close to the obstacles (Margin = safe_distance from the JSON configuration file + increase of choice [m])
    safe_margin = opt_config["constraints"]["safe_distance"] + 0.5
    waypoints = sanitize_waypoints(waypoints, map3d.obstacles, safety_margin=safe_margin)

    # Voronoi Partition 
    print("Computing Voronoi partition for the generated map and drone starting positions...")
    seeds_xy = kmeans_clustering(
        map3d.free_space,
        uav_cfg["num_uavs"],
        seed=seed,
        waypoints=waypoints
    )

    # print("Voronoi seeds from k-means on free space:")
    # for i, seed_xy in enumerate(seeds_xy):
    #    print(f"  Seed {i+1}: {seed_xy}")

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
    log_name = config["log_name"]
    # TO BE CHANGED BEFORE RUNNING THE SCRIPT
    # logs file filename, if not existing the script will create one with such name
    csv_filepath = ROOT / "logs" / f"{log_name}.csv"

    # choose on how many maps do you want to simulate the optimization problem
    # seed_list = [3, 27, 51, 13, 93, 42, 84, 79, 32, 25, 33, 41, 69, 55, 99, 1, 7, 77, 11, 62, 76, 48, 82, 26, 64]
    seed_list = config["seed"]
    
    for test_seed in seed_list:
        # Set random seed for reproducibility
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

        # Transform obstacles objects in a structure that is actually fast to use for obstacle, drones avoidance and next waypoints search
        obs_tree = KDTree(obstacle_coords)
        obstacles = map3d.obstacles

        wp_tree = KDTree(waypoints)

        # SETUP MPC
        # After how many iterations a recap of the current situation is shown
        PRINT_INTERVAL = 10
        # number of other drones apart from the one which is running the algorithm
        num_neighbors = len(drone_positions) - 1

        # take the parameters from config file
        config_path = ROOT / "configs" / "optimization_params.json"
        opt_config = load_config(config_path)
        mpc_cfg = opt_config["mpc"]
        N = mpc_cfg["prediction_horizon"]
        dt = mpc_cfg["timestep"]
        max_iter = mpc_cfg["max_iter"]
        current_overlap = opt_config["constraints"]["overlap_factor"]
        safety_radius = opt_config["constraints"]["safe_distance"]

        # INITIALIZATION 
        drones = []
        drone_ids = [0] * len(drone_positions)
        for i in range(len(drone_positions)):
            drone_ids[i] = i

        # assign the waypoints to the associated drone
        assign_area(vor, drone_positions)

        # Add the 'seen' flag column to the global waypoints matrix
        if waypoints.shape[1] == 2:
            seen_column = np.zeros((waypoints.shape[0], 1)) # Create column of 0s
            waypoints = np.hstack((waypoints, seen_column)) # Attach it

        # MAIN MPC LOOP
        # initialization of the run without early switching for a posteriori comparison
        dist_threshold = 0.5 # Distance to mark a waypoint as 'seen' -> flag switched to 1 [m]
        ego_accel_prev = 0 # previous acceleration (input) of the drone
        t_solve_avg = 0 # average solve time of the problem
        early_swtiching_flag = False
        
        drones_normal = spawn_swarm()
        drones_normal, cost_hist_normal, avg_solve_time = run_swarm_simulation(
            drones_normal, dt, max_iter, opt_config, map3d.obstacles, obs_tree, dist_threshold, early_swtiching_flag, PRINT_INTERVAL
        )
        
        # Extract Normal Metrics
        normal_metrics = {"speed": [], "jerk": [], "energy": [], "miss": [], "state": [], "time": [], "collisions": [], "solve_time": []}
        drone_labels = []
        mass = 1.0 # kg

        # create logs
        for drone in drones_normal:
            report = evaluate_trajectory_performance(drone, dt)
            normal_metrics["speed"].append(report["avg_cornering_speed"])
            normal_metrics["jerk"].append(report["jerk"])
            normal_metrics["miss"].append(report["avg_miss_distance"])

            # EXACT TIME & ENERGY CALCULATION
            v_mag = np.linalg.norm(drone.history_v, axis=1)
            a_mag = np.linalg.norm(drone.history_a, axis=1)
            
            if drone.is_parked:
                status = "Success"
                active_steps = np.max(np.nonzero(v_mag > 1e-5)) + 1 if np.any(v_mag > 1e-5) else 0
                drone_time = active_steps * dt
                
                # Get the 3D acceleration vectors for the active flight time
                active_a_3d = np.array(drone.history_a[:active_steps], dtype=float)
                
                # Add gravity to the Z-axis 
                active_a_3d[:, 2] += 9.81 
                
                # Calculate total thrust magnitude
                thrust_mag = mass * np.linalg.norm(active_a_3d, axis=1)
                
                # 4. Integrate squared thrust over time (Energy Proxy)
                drone_energy = np.sum(thrust_mag**2) * dt
                
            else:
                status = "Stuck"
                drone_time = len(drone.history_p) * dt
                
                # Same math, but for the entire trapped duration
                a_3d = np.array(drone.history_a, dtype=float)
                a_3d[:, 2] += 9.81
                thrust_mag = mass * np.linalg.norm(a_3d, axis=1)
                drone_energy = np.sum(thrust_mag**2) * dt
            
            # KD-TREE DISCRETE COLLISION COUNTER
            collision_events = 0
            in_collision = False
            
            # Find the max radius so the KD-Tree knows how wide to cast its net
            max_search_radius = np.max(obs_radii)
            
            for p in drone.history_p:
                step_collision = False
                
                # Ask the KD-Tree for the indices of obstacles that are strictly nearby
                # (p is the [x,y,z] coordinate from the history)
                nearby_obs_indices = obs_tree.query_ball_point(p, r=max_search_radius)
                
                # Only loop through the 1 or 2 obstacles the tree found
                for idx in nearby_obs_indices:
                    obs = map3d.obstacles[idx]
                    
                    # Exact 2D distance check against the specific obstacle's actual radius
                    dist = np.hypot(p[0] - obs.x, p[1] - obs.y)
                    if dist <= (obs.radius + 0.25*safety_radius):
                        step_collision = True
                        break # Found a hit, no need to check other nearby obstacles
                
                # Discrete event tracking logic
                if step_collision and not in_collision:
                    collision_events += 1
                    in_collision = True
                elif not step_collision:
                    in_collision = False

            
            normal_metrics["state"].append(status)
            normal_metrics["time"].append(drone_time) 
            normal_metrics["energy"].append(drone_energy) 
            normal_metrics["collisions"].append(collision_events)
            normal_metrics["solve_time"].append(avg_solve_time)
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
        # print("\n" + "="*50)
        print(" STARTING RUN 2: EARLY SWITCHING")
        # print("="*50)

        early_swtiching_flag = True
        
        drones_early = spawn_swarm() 
        drones_early, cost_hist_early, early_avg_solve_time = run_swarm_simulation(
            drones_early, dt, max_iter, opt_config, map3d.obstacles, obs_tree, dist_threshold, early_swtiching_flag, PRINT_INTERVAL
        )
        
        # Extract early metrics
        early_metrics = {"speed": [], "jerk": [], "energy": [], "miss": [], "state": [], "time": [], "collisions": [], "solve_time": []}
        drone_labels = []
        for drone in drones_early:
            report = evaluate_trajectory_performance(drone, dt)
            early_metrics["speed"].append(report["avg_cornering_speed"])
            early_metrics["jerk"].append(report["jerk"])
            early_metrics["miss"].append(report["avg_miss_distance"])
            
            # TIME & ENERGY CALCULATION
            v_mag = np.linalg.norm(drone.history_v, axis=1)
            a_mag = np.linalg.norm(drone.history_a, axis=1)
            
            if drone.is_parked:
                status = "Success"
                active_steps = np.max(np.nonzero(v_mag > 1e-5)) + 1 if np.any(v_mag > 1e-5) else 0
                drone_time = active_steps * dt
                
                # Get the 3D acceleration vectors for the active flight time
                active_a_3d = np.array(drone.history_a, float)
                
                # Add gravity to the Z-axis
                active_a_3d[:, 2] += 9.81 
                
                # Calculate total thrust magnitude: 
                thrust_mag = mass * np.linalg.norm(active_a_3d, axis=1)
                
                # Integrate squared thrust over time (Energy Proxy)
                drone_energy = np.sum(thrust_mag**2) * dt
                
            else:
                status = "Stuck"
                drone_time = len(drone.history_p) * dt
                
                # Same math, but for the entire trapped duration
                a_3d = np.array(drone.history_a, float)
                a_3d[:, 2] += 9.81
                thrust_mag = mass * np.linalg.norm(a_3d, axis=1)
                drone_energy = np.sum(thrust_mag**2) * dt

            # KD-TREE DISCRETE COLLISION COUNTER
            collision_events = 0
            in_collision = False
            
            # Find the max radius so the KD-Tree knows how wide to cast its net
            max_search_radius = np.max(obs_radii) 
            
            for p in drone.history_p:
                step_collision = False
                
                # Ask the KD-Tree for the indices of obstacles that are strictly nearby
                # (p is [x,y,z] coordinate from the history)
                nearby_obs_indices = obs_tree.query_ball_point(p, r=max_search_radius)
                
                # Only loop through the 1 or 2 obstacles the tree found
                for idx in nearby_obs_indices:
                    obs = map3d.obstacles[idx]
                    
                    # Exact 2D distance check against the specific obstacle's actual radius
                    dist = np.hypot(p[0] - obs.x, p[1] - obs.y)
                    if dist <= (obs.radius + 0.25*safety_radius):
                        step_collision = True
                        break # Found a hit, no need to check other nearby obstacles
                
                # Discrete event tracking logic
                if step_collision and not in_collision:
                    collision_events += 1
                    in_collision = True
                elif not step_collision:
                    in_collision = False

            early_metrics["state"].append(status)
            early_metrics["time"].append(drone_time) 
            early_metrics["energy"].append(drone_energy)
            early_metrics["collisions"].append(collision_events) 
            early_metrics["solve_time"].append(early_avg_solve_time)
            drone_labels.append(f"Drone {drone.id}")
            
        # Calculate global coverage 
        res = 0.2
        early_cov, _ = calculate_final_coverage(drones_early, map_limits, L, W, res)
        
        # Save logs to csv
        save_metrics_to_csv(csv_filepath, test_seed, current_overlap, "Early", 
                            drone_labels, early_metrics, early_cov)

        # find time of mission completion        
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
   
    # Show the 3D map or animation for the Early Switching run
    plot_results(drones_early, map3d.obstacles)

    # Plot the applied inputs and velocities
    plot_kinematics(drones_early, dt)

    animate_simulation(drones_early, map3d.obstacles, map_limits)

    res = 0.2 # Resolution
    final_coverage_pct, coverage_grid = calculate_final_coverage(drones_early, map_limits, L, W, res)
    plot_coverage_map(coverage_grid, map_limits, res, obstacles, drones_early)
    plot_offline_csv_comparison(csv_filepath)
    print(f"Final Map Coverage: {final_coverage_pct:.2f}%")
    print(f"Early Switch Average solve time: {early_avg_solve_time} s")
    print(f"Normal Switch Average solve time: {avg_solve_time} s")