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
from utils.kmeans import kmeans_clustering, sanitize_waypoints
from utils.plot_voronoi import plot_voronoi_partition
from partition.voronoi import Voronoi_Partition, assign_area, get_waypoints_in_partition
from optimization.mpc import setup_MPC_QP, run_mpc_iteration, setup_MPC_NLP, setup_test_MPC, setup_test_MPC_QP 
from optimization.waypoints_sorter import sort_waypoints_tsp
from utils.drones import Drone
from optimization.optimization_plots import plot_results, animate_simulation, plot_kinematics
from simulator.pybullet_simulator import Simulator

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

    # 2. FIX: Pulizia Waypoint (Margine super safe di prova: safe_distance del JSON + 0.25m)
    # ho messo 1 invece di importare safe_distance dal file config perchè non avevo sbatti
    safe_margin = 1.5 + 0.25   
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

    return map3d, vor, drone_positions, waypoints


if __name__ == "__main__":
    env_config_path = ROOT / "configs" / "demo_parameters.json"
    env_config = load_config(env_config_path)

    seed_everything(env_config["seed"])

    map3d, vor, drone_positions, waypoints = build_demo(env_config)

    plot_voronoi_partition(
        map3d,
        vor,
        drone_positions=drone_positions,
        waypoints=waypoints,
        title="Voronoi Partition of the Workspace",
    )

    obstacle_coords = np.array([[obs.x, obs.y, obs.height / 2.0] for obs in map3d.obstacles])
    obs_tree = KDTree(obstacle_coords)

    PRINT_INTERVAL = 10
    num_neighbors = len(drone_positions) - 1

    opt_config_path = ROOT / "configs" / "optimization_params.json"
    opt_config = load_config(opt_config_path)
    mpc_cfg = opt_config["mpc"]
    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]
    max_iter = mpc_cfg["max_iter"]

    drones = []
    drone_ids = list(range(len(drone_positions)))

    assign_area(vor, drone_positions)

    if waypoints.shape[1] == 2:
        seen_column = np.zeros((waypoints.shape[0], 1))
        waypoints = np.hstack((waypoints, seen_column))

    for id_d in drone_ids:
        current_cell = vor.Voronoi_Cells[id_d]
        partition_shape = Polygon(current_cell.polygon)
        waypoints_assigned = get_waypoints_in_partition(waypoints, partition_shape)
        waypoints_ordered = sort_waypoints_tsp(drone_positions[id_d], waypoints_assigned)

        vars_ = setup_test_MPC_QP(num_neighbors=num_neighbors, enable_obstacles=True)
        new_drone = Drone(id_d, drone_positions[id_d], waypoints_ordered, vars_, N)
        drones.append(new_drone)
        new_drone.returning_home = False
        new_drone.is_parked = False
        new_drone.home_pos = drone_positions[id_d]

    pybullet_cfg = env_config.get("pybullet", {})
    simulator = None
    if pybullet_cfg.get("enabled", True):
        simulator = Simulator(
            drones=drones,
            map3d=map3d,
            dt=dt,
            gui=pybullet_cfg.get("gui", True),
            real_time=pybullet_cfg.get("real_time", True),
            drone_radius=pybullet_cfg.get("drone_radius", 0.1),
            drone_visual=pybullet_cfg.get("drone_visual", "sphere"),
        )
        simulator.sync_all_drones(drones)

    num_iter = 0
    dist_threshold = 0.8
    dJ_thresh = 1e-6
    prev_total_cost = 1e6

    cost_history = {
        "total": [],
        "waypoints": [],
        "effort": [],
        "battery": [],
        "z_ref": [],
        "barrier": [],
    }

    total_solver_time = 0.0
    total_solver_calls = 0

    while num_iter <= max_iter:
        total_loop_cost = 0.0

        if all(d.is_parked for d in drones):
            print(f"\nMission accomplished in {num_iter} steps!")
            if total_solver_calls > 0:
                average_time = total_solver_time / total_solver_calls
                print(f"Avg Solve Time: {average_time:.5f} seconds")
            break

        for drone in drones:
            unseen_mask = drone.waypoints[:, 2] == 0
            if not np.any(unseen_mask) and not drone.returning_home:
                print(f"Drone {drone.id} finished mission! Returning home.")
                home_wp = np.array([drone.home_pos[0], drone.home_pos[1], 0])
                drone.waypoints = np.vstack([drone.waypoints, home_wp])
                drone.returning_home = True

            if drone.returning_home and not drone.is_parked:
                dist_to_home = np.linalg.norm(drone.state["p"][:2] - drone.home_pos[:2])
                if dist_to_home < dist_threshold:
                    print(f"Drone {drone.id} has parked safely!")
                    drone.is_parked = True

            if drone.is_parked:
                drone.state["v"] = np.zeros(3)
                drone.state["a"] = np.zeros(3)

                horizon_n = drone.mpc_vars["p"].shape[1] - 1
                stationary_traj = np.tile(drone.state["p"], (horizon_n + 1, 1)).T
                drone.last_traj = stationary_traj

                drone.history_p.append(drone.state["p"].copy())
                drone.history_a.append(np.zeros(3))
                if hasattr(drone, "history_predictions"):
                    drone.history_predictions.append(stationary_traj)
                if simulator is not None:
                    simulator.sync_drone_state(drone)
                continue

            neighbor_trajs = [d.last_traj for d in drones if d.id != drone.id]
            if neighbor_trajs:
                neighbor_trajs_array = np.stack(neighbor_trajs, axis=2)
            else:
                neighbor_trajs_array = np.empty((3, N + 1, 0))

            accel, new_traj, current_cost_value, cost_breakdown, t_solve_mpc = run_mpc_iteration(
                drone.mpc_vars,
                drone.state,
                drone.waypoints,
                drone.last_traj,
                neighbor_trajs_array,
                obs_tree,
            )

            total_solver_time += t_solve_mpc
            total_solver_calls += 1

            if current_cost_value != np.inf:
                cost_history["total"].append(current_cost_value)
                for key, val in cost_breakdown.items():
                    if key not in cost_history:
                        cost_history[key] = []
                    cost_history[key].append(val)

            total_loop_cost += current_cost_value

            drone.drone_model(accel, dt)
            drone.check_waypoints(dist_threshold)
            drone.log_telemetry(new_traj)
            drone.last_traj = new_traj
            drone.history_a.append(accel)
            drone.state["a"] = accel

            if simulator is not None:
                simulator.sync_drone_state(drone)

            if num_iter % PRINT_INTERVAL == 0:
                print(f"\n--- Step {num_iter} | Drone {drone.id} ---")
                print(f"Total Cost:  {current_cost_value:.2f}")
                print(f"  Waypoints: {cost_breakdown['waypoints']:.2f}")
                print(f"  Effort:    {cost_breakdown['effort']:.2f}")
                print(f"  Battery:   {cost_breakdown['battery']:.2f}")
                print(f"  Slack:     {cost_breakdown['slack']:.2f}")
                print(f"  Barrier:   {cost_breakdown['barrier']:.2f}")

        if simulator is not None:
            simulator.step()
            collisions = simulator.get_new_collisions(drones)
            for drone_id, labels in collisions.items():
                print(f"[PyBullet] Drone {drone_id} collisioni rilevate con: {', '.join(labels)}")

        if abs(prev_total_cost - total_loop_cost) < dJ_thresh:
            print("Converged!")
            break

        prev_total_cost = total_loop_cost
        num_iter += 1

    print("optimization completed")

    print("\n" + "=" * 40)
    print("       OPTIMIZATION COST RECAP")
    print("=" * 40)
    print(f"{'Component':<15} | {'Mean':<10} | {'Max':<10}")
    print("-" * 40)

    for key, values in cost_history.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            max_val = np.max(values)
            print(f"{key.capitalize():<15} | {mean_val:<10.2f} | {max_val:<10.2f}")

    valid_items = [(key, values) for key, values in cost_history.items() if len(values) > 0]
    num_plots = len(valid_items)

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, valid_items):
        ax.plot(values, label=f"{key.capitalize()} (Max: {np.max(values):.1f})")
        ax.set_title(f"{key.capitalize()} Cost", fontsize=12)
        ax.set_ylabel("Cost (Log)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right")

    plt.xlabel("Iteration Step", fontsize=12)
    plt.suptitle("MPC Cost Components Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    plot_results(drones, map3d.obstacles)
    plot_kinematics(drones, dt)

    map_cfg = env_config["map"]
    map_limits = [
        map_cfg["x_bounds"],
        map_cfg["y_bounds"],
        map_cfg["z_bounds"],
    ]
    animate_simulation(drones, map3d.obstacles, map_limits)
