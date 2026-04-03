from config import load_config
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import os

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
    num_regions = mpc_cfg["k_tree_search"]

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
    opti.solver("ipopt", {"expand": True})

    # Return a dictionary containing the symbolic objects to be used in the loop
    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions
    }

def run_mpc_iteration(mpc_vars, current_state, waypoint_coords,  
                      last_traj, neighbor_trajs, obs_tree):
    """
    Executes one step of the MPC.
    waypoint_coords: [M x 3] numpy array [x, y, seen_flag]
    """
    opti = mpc_vars["opti"]
    k_limit = mpc_vars["k_search"] # Get the '3' (or '5') from the dict

    # 1. Waypoint Search
    num_available = waypoint_coords.shape[0]
    # We can't query more than we have, but we must return exactly k_limit
    k_query = min(k_limit, num_available)
    
    wp_tree = KDTree(waypoint_coords[:, :2])
    dist, indices = wp_tree.query(current_state["p"][:2], k=k_query)
    
    # Handle single-index return if k=1
    if k_query == 1: indices = [indices]
    
    closest_coords_2d = waypoint_coords[indices, :2]
    closest_flags = waypoint_coords[indices, 2]
    
    # 2. Dynamic Padding
    if k_query < k_limit:
        padding_count = k_limit - k_query
        last_coord = closest_coords_2d[-1:, :]
        closest_coords_2d = np.vstack([closest_coords_2d] + [last_coord]*padding_count)
        closest_flags = np.append(closest_flags, [1.0]*padding_count)
    
    # Convert to 3D for CasADi (3 rows, k_limit columns)
    z_padding = np.zeros((k_limit, 1))
    closest_coords_3d = np.hstack((closest_coords_2d, z_padding))

    distances_obs, indices_obs = obs_tree.query(np.array(last_traj).T)
    
    # Extract the actual 3D coordinates of those closest obstacles
    # Resulting shape: (3, N+1)
    closest_obs_coords = obs_tree.data[indices_obs].T

    # 3. Parameter Update
    opti.set_value(mpc_vars["p_wp"], closest_coords_3d.T)
    opti.set_value(mpc_vars["flag"], closest_flags)

    # --- 3. PARAMETER UPDATE ---
    opti.set_value(mpc_vars["p_init"], current_state["p"])
    opti.set_value(mpc_vars["v_init"], current_state["v"])
    opti.set_value(mpc_vars["B_init"], current_state["B"])
    
    # These now strictly match the (3, 3) and (3,) parameters
    opti.set_value(mpc_vars["p_wp"], closest_coords_3d.T) 
    opti.set_value(mpc_vars["flag"], closest_flags)
    
    opti.set_value(mpc_vars["p_ego_prev"], last_traj)
    opti.set_value(mpc_vars["p_obs_closest"], closest_obs_coords)
    
    # Neighbor trajectories: flatten (3, N+1, num_neighbors) -> (3, (N+1)*num_neighbors)
    flattened_neighbors = neighbor_trajs.reshape((3, -1), order='F')
    opti.set_value(mpc_vars["p_neighbors"], flattened_neighbors)

    try:
        sol = opti.solve()
        cost_value = sol.value(mpc_vars["opti"].f)
        new_trajectory = sol.value(mpc_vars["p"])
        optimal_accel = sol.value(mpc_vars["a"])
        
        # solver stats
        solve_time = sol.stats()['t_wall_total']
        print(f"MPC solve successful: {solve_time:.4f}s") 
        
        return optimal_accel[:, 0], new_trajectory, cost_value

    except RuntimeError:
        print("MPC solve failed! Using safety fallback.")
        return np.array([0.0, 0.0, 0.0]), last_traj