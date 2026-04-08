from config import load_config
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import os
import time

# This forces Python to look specifically in your site-packages for the DLLs
try:
    import casadi as ca
    casadi_path = Path(ca.__file__).parent
    os.add_dll_directory(str(casadi_path))
    # Also add the root of the venv just in case
    os.add_dll_directory(str(Path(sys.executable).parent))
except Exception:
    pass

def get_project_root() -> Path:
    """Climbs up from the current file until it finds the folder containing 'src'."""
    current = Path(__file__).resolve()
    # Check current folder and all parents
    for parent in [current] + list(current.parents):
        if (parent / "src").is_dir():
            return parent
    # Fallback to the script's parent if 'src' isn't found
    return current.parent

ROOT = get_project_root()
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"

# Add SRC to sys.path if it's not there
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Now your config path will ALWAYS be correct
config_path = CONFIGS / "optimization_params.json"

def setup_MPC_QP(num_neighbors): 
    """
    Initializes the CasADi Opti stack. Run this ONCE at the start.
    """
    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    config_path = CONFIGS / "optimization_params.json"
    
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
    dt = mpc_cfg["timestep"]
    num_regions = mpc_cfg["k_wp_search"]
    k_obs = mpc_cfg["k_obs_search"]

    opti = ca.Opti("conic")

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
    p_obs_closest = opti.parameter(3, (N+1)* k_obs) 
    # Predicted trajectories of neighboring drones (3D, Horizon, Neighbor ID)
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)

    # Target waypoints and their active/inactive flags
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)

    # Previous solution used for Taylor expansion (linearization)
    p_ego_prev = opti.parameter(3, N+1)

    # --- Cost Function ---
    cost = 0
  
    # Task cost: Reach waypoints (if flag is 0)
    for i in range(num_regions):
        # Accumulate error along all the horizon
        for k in range(1, N + 1): 
            cost += (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * w_seen

    # Control effort cost: Reduce acceleration magnitude
    for k in range(N):
        cost += w_effort * ca.sumsqr(a[:, k])
        
    opti.minimize(cost)

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
    for j in range(num_neighbors):
        for k in range(N+1):
            # Slice the wide matrix to get the k-th step of the j-th neighbor
            # The column index logic: j * (N+1) + k
            col_idx = j * (N+1) + k
            dp_bar = p_ego_prev[:, k] - p_neighbors[:, col_idx]
            
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)

    # --- Solver Choice ---
    # Using 'qrqp' or 'osqp' but has to be installed for QP or 'ipopt' for NLP
    # 'expand': True speeds up the solver by evaluating the graph once
    opti.solver("osqp", {"expand": True})

    # Return a dictionary containing the symbolic objects to be used in the loop
    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions, "k_obs": k_obs
    }

def run_mpc_iteration(mpc_vars, current_state, waypoint_coords,  
                      last_traj, neighbor_trajs, obs_tree):
    """
    Executes one step of the MPC.
    waypoint_coords: [M x 3] numpy array [x, y, seen_flag]
    """

    opti = mpc_vars["opti"]
    k_limit = mpc_vars["k_search"]
    k_obs = mpc_vars["k_obs"] # Assicurati che questo sia nel dizionario mpc_vars!

    # --- 1. WAYPOINT SEARCH ---
    num_available = waypoint_coords.shape[0]
    k_query = min(k_limit, num_available)
    
    # Rimosso k=k_obs da qui, va solo nella query
    wp_tree = KDTree(waypoint_coords[:, :2])
    dist, indices = wp_tree.query(current_state["p"][:2], k=k_query)
    
    if k_query == 1: 
        indices = [indices]
    
    # Definizione corretta delle coordinate locali
    closest_coords_2d = waypoint_coords[indices, :2]
    closest_flags = waypoint_coords[indices, 2]

    # --- 2. DYNAMIC PADDING ---
    final_coords_2d = closest_coords_2d
    final_flags = closest_flags
    
    if k_query < k_limit:
        padding_count = k_limit - k_query
        last_coord = closest_coords_2d[-1:, :]
        final_coords_2d = np.vstack([closest_coords_2d] + [last_coord] * padding_count)
        final_flags = np.append(closest_flags, [1.0] * padding_count)
    
    # Conversione in 3D (p_wp)
    z_padding = np.zeros((k_limit, 1))
    closest_coords_3d = np.hstack((final_coords_2d, z_padding))

    # --- 3. OBSTACLE SEARCH (K-NEAREST) ---
    # Query per ogni punto della traiettoria precedente
    # last_traj shape: (3, N+1)
    dist_obs, indices_obs = obs_tree.query(last_traj.T, k=k_obs)
    
    # Se k_obs=1, indices_obs è (N+1,), forziamo (N+1, 1)
    if k_obs == 1:
        indices_obs = indices_obs.reshape(-1, 1)

    # Costruiamo la matrice per il solver (3 righe, (N+1)*k_obs colonne)
    num_points = last_traj.shape[1]
    closest_obs_coords = np.zeros((3, num_points * k_obs))
    
    for k in range(num_points):
        for j in range(k_obs):
            obs_idx = indices_obs[k, j]
            col_idx = k * k_obs + j
            closest_obs_coords[:, col_idx] = obs_tree.data[obs_idx]

    # --- 4. SET PARAMETERS ---
    opti.set_value(mpc_vars["p_init"], current_state["p"])
    opti.set_value(mpc_vars["v_init"], current_state["v"])
    opti.set_value(mpc_vars["B_init"], current_state["B"])
    
    opti.set_value(mpc_vars["p_wp"], closest_coords_3d.T)
    opti.set_value(mpc_vars["flag"], final_flags)
    
    opti.set_value(mpc_vars["p_ego_prev"], last_traj)
    opti.set_value(mpc_vars["p_obs_closest"], closest_obs_coords)
    
    # ... resto della funzione (neighbor trajs e solve) ...
    
    # Neighbor trajectories: flatten (3, N+1, num_neighbors) -> (3, (N+1)*num_neighbors)
    flattened_neighbors = neighbor_trajs.reshape((3, -1), order='F')
    opti.set_value(mpc_vars["p_neighbors"], flattened_neighbors)

    try:

        # Start the high-resolution timer right before the solve step
        start_time = time.perf_counter()
        sol = opti.solve()
        # Stop the timer immediately after
        end_time = time.perf_counter()

        # Calculate the elapsed time
        solve_time = end_time - start_time
        cost_value = sol.value(mpc_vars["opti"].f)
        new_trajectory = sol.value(mpc_vars["p"])
        optimal_accel = sol.value(mpc_vars["a"])
        
        # solver stats
        # when using ipopt 
        # solve_time = sol.stats()['t_wall_total']
        print(f"MPC solve successful: {solve_time:.4f}s") 
        
        return optimal_accel[:, 0], new_trajectory, cost_value

    except RuntimeError:
        print("MPC solve failed! Using safety fallback.")
        return np.array([0.0, 0.0, 0.0]), last_traj, cost_value