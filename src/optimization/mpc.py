from config import load_config
import sys
from pathlib import Path
import casadi as ca
import numpy as np
from scipy.spatial import KDTree

def setup_MPC_QP(waypoints, num_neighbors): 
    """
    Initializes the CasADi Opti stack. Run this ONCE at the start.
    """
    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    config_path = ROOT / "configs" / "optimization.json"
    
    # Load configuration
    config = load_config(config_path)

    # Unpack variables from config
    cost_cfg = config["cost"]
    constraints_cfg = config["constraints"]
    mpc_cfg = config["mpc"]

    # Cost function weights
    w_seen = cost_cfg["w_seen"]
    w_effort = cost_cfg["w_effort"]

    # Physical constraints
    max_vel = constraints_cfg["max_speed"]
    max_acc = constraints_cfg["max_acceleration"]
    safe_rad = constraints_cfg["safe_distance"] 

    # MPC parameters
    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["dt"]
    num_regions = waypoints.shape[0] 

    opti = ca.Opti()

    # --- Optimization Variables ---
    p = opti.variable(3, N+1)  # Position
    v = opti.variable(3, N+1)  # Velocity
    B = opti.variable(1, N+1)  # Battery state
    a = opti.variable(3, N)    # Acceleration input

    # --- Parameters (Updated at each MPC iteration) ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    # Closest obstacles for each step of the horizon
    p_obs_closest = opti.parameter(3, N+1)
    # Predicted trajectories of neighboring drones (3D, Horizon, Neighbor ID)
    p_neighbors = opti.parameter(3, N+1, num_neighbors) 

    # Target waypoints and their active/inactive flags
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)

    # Previous solution used for Taylor expansion (linearization)
    p_ego_prev = opti.parameter(3, N+1)

    # --- Cost Function ---
    cost = 0
  
    # Task cost: Reach waypoints (if flag is 0)
    for i in range(num_regions):
        # Minimize distance between the end of horizon and the waypoint
        cost += (1 - flag[i]) * ca.sumsqr(p[:, N] - p_wp[:, i]) * w_seen

    # Control effort cost: Reduce acceleration magnitude
    for k in range(N):
        cost += w_effort * ca.sumsqr(a[:, k])
        
    opti.minimize(cost)

    # --- Dynamics Constraints (Multiple Shooting) ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)
    opti.subject_to(B[:, 0] == B_init)

    for k in range(N):
        # Kinematic model
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)
        
        # NOTE: Battery dynamics B[k+1] = B[k] - c*||a||^2 is non-linear.
        # To keep this as a QP for OSQP/QRQP, we treat Battery as a simple 
        # state bound here. You can calculate the drop after solving.

    # --- Physical Bounds ---
    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))
    opti.subject_to(opti.bounded(0, B, 100)) 

    # --- Linearized Obstacle Avoidance ---
    for k in range(N+1):
        dp_bar = p_ego_prev[:, k] - p_obs_closest[:, k]
        dist_bar_sqr = ca.sumsqr(dp_bar)
        # First-order Taylor expansion around p_ego_prev
        linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
        opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)
    
    # --- Linearized Neighbor Collision Avoidance ---
    for k in range(N+1):
        for j in range(num_neighbors):
            dp_bar = p_ego_prev[:, k] - p_neighbors[:, k, j]
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)

    # --- Solver Choice ---
    # Using 'osqp' for QP or 'ipopt' for NLP
    # 'expand': True speeds up the solver by evaluating the graph once
    opti.solver("osqp", {"expand": True})

    # Return a dictionary containing the symbolic objects to be used in the loop
    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors
    }

def run_mpc_iteration(mpc_vars, current_state, local_flags, waypoint_coords, 
                      last_traj, neighbor_trajs, obs_tree):
    """
    Executes one step of the MPC. Call this inside your main loop.
    """
    opti = mpc_vars["opti"]
    
    # 1. Query KDTree for closest obstacles along the previous trajectory
    # Transpose last_traj to match (N_points, 3) for KDTree
    distances_obs, indices_obs = obs_tree.query(np.array(last_traj).T)
    closest_obs_coords = obs_tree.data[indices_obs].T 
    waypoints_tree = KDTree(waypoint_coords)
    distances_wp, indices_wp = waypoints_tree.query(np.array(last_traj).T, k = 3)
    closest_wp_coords = obs_tree.data[indices_wp].T 

    # 2. Update CasADi parameters with current numerical values
    opti.set_value(mpc_vars["p_init"], current_state["p"])
    opti.set_value(mpc_vars["v_init"], current_state["v"])
    opti.set_value(mpc_vars["B_init"], current_state["B"])
    opti.set_value(mpc_vars["flag"], local_flags)
    opti.set_value(mpc_vars["p_wp"], closest_wp_coords)
    opti.set_value(mpc_vars["p_ego_prev"], last_traj)
    opti.set_value(mpc_vars["p_obs_closest"], closest_obs_coords)
    opti.set_value(mpc_vars["p_neighbors"], neighbor_trajs)

    try:
        # Solve the QP
        sol = opti.solve()
        
        # Extract numerical results
        new_trajectory = sol.value(mpc_vars["p"])
        optimal_accel = sol.value(mpc_vars["a"])
        
        # Performance profiling
        solve_time = sol.stats()['t_wall_total']
        print(f"MPC solve successful: {solve_time:.4f}s")
        
        return optimal_accel[:, 0], new_trajectory

    except RuntimeError:
        print("MPC solve failed! Using safety fallback.")
        # Debugging: check the state of the solver at failure
        # return_zero_accel, keep_old_trajectory
        return np.array([0.0, 0.0, 0.0]), last_traj