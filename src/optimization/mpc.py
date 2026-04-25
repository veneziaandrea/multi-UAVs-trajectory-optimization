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
config = load_config(config_path)

map_config = load_config(CONFIGS/"demo_parameters.json")
bounds_cfg = map_config["map"]

def setup_MPC_QP(num_neighbors): 
    # ... [Inizializzazione ROOT, SRC, config...] ...

    cost_cfg = config["cost"]
    constraints_cfg = config["constraints"]
    mpc_cfg = config["mpc"]

    w_seen = cost_cfg["w_seen"]
    w_effort = cost_cfg["w_effort"]
    w_batt = cost_cfg["w_battery"]
    p_hover = cost_cfg["p_hover"]
    z_ref = cost_cfg["z_ref"]
    w_z = cost_cfg["w_z"] # Aggiungi questo al JSON, o usa un default
    w_slack = cost_cfg["w_slack_collision"] # Peso ENORME per le collisioni

    max_vel = constraints_cfg["max_speed"]
    max_acc = constraints_cfg["max_acceleration"]
    safe_rad = constraints_cfg["safe_distance"] 

    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]
    num_regions = mpc_cfg["k_wp_search"]
    k_obs = mpc_cfg["k_obs_search"]

    opti = ca.Opti("conic")

    # --- Variables ---
    p = opti.variable(3, N+1)  
    v = opti.variable(3, N+1)  
    B = opti.variable(1, N+1)  
    a = opti.variable(3, N)    
    
    # SLACK VARIABLES per evitare i minimi locali
    eps_obs = opti.variable(k_obs, N+1)

    # --- Parameters ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    p_obs_closest = opti.parameter(3, (N+1) * k_obs) 
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)
    p_ego_prev = opti.parameter(3, N+1)

    # --- COST FUNCTION ---
    cost = 0
  
    # 1. Waypoints & Hovering (Battery)
    for i in range(num_regions):
        for k in range(1, N + 1): 
            cost += (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * w_seen

    # 2. Control Effort, Battery (Velocity), and Z-Reference Tracking
    for k in range(N):
        cost += w_effort * ca.sumsqr(a[:, k])
        cost += w_batt * ca.sumsqr(v[:, k])
        cost += w_z * ca.sumsqr(p[2, k] - z_ref) # Mantieni la quota!

    # 3. Penalità sulle Slack Variables (Funge da Barrier Function)
    for k in range(1, N + 1):
        for j in range(k_obs):
            cost += w_slack * ca.sumsqr(eps_obs[j, k])

    opti.minimize(cost)

    # --- DYNAMICS CONSTRAINTS ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)

    for k in range(N):
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)

    # --- PHYSICAL BOUNDS ---
    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))
    
    for k in range(1, N+1):
        for j in range(k_obs):
            opti.subject_to(eps_obs[j, k] >= 0) # Le slack non possono essere negative

    # --- LINEARIZED OBSTACLE AVOIDANCE ---
    for k in range(N+1):
        for j in range(k_obs):
            col_idx = k * k_obs + j
            dp_bar = p_ego_prev[:, k] - p_obs_closest[:, col_idx]
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            
            # Qui la slack (eps_obs) funge da ammortizzatore / soft barrier
            opti.subject_to(dist_bar_sqr + linear_term + eps_obs[j, k] >= safe_rad**2)
    
    # --- NEIGHBOR AVOIDANCE ---
    for j in range(num_neighbors):
        for k in range(N+1):
            col_idx = j * (N+1) + k
            dp_bar = p_ego_prev[:, k] - p_neighbors[:, col_idx]
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)

    opti.solver("osqp", {"expand": True})

    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions, "k_obs": k_obs
    }

def setup_MPC_NLP(num_neighbors): 

    cost_cfg = config["cost"]
    constraints_cfg = config["constraints"]
    mpc_cfg = config["mpc"]

    w_seen = cost_cfg["w_seen"]
    w_effort = cost_cfg["w_effort"]
    w_batt = cost_cfg["w_battery"]
    z_ref = cost_cfg["z_ref"]
    w_z = cost_cfg["w_z"] 
    w_slack = cost_cfg["w_slack_collision"] # Peso ENORME per le collisioni

    max_vel = constraints_cfg["max_speed"]
    max_acc = constraints_cfg["max_acceleration"]
    safe_rad = constraints_cfg["safe_distance"] 

    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]
    num_regions = mpc_cfg["k_wp_search"]
    k_obs = mpc_cfg["k_obs_search"]

    x_min, x_max = bounds_cfg["x_bounds"]
    y_min, y_max = bounds_cfg["y_bounds"]
    z_min, z_max = bounds_cfg["z_bounds"]

    opti = ca.Opti()

    # --- Variables ---
    p = opti.variable(3, N+1)  
    v = opti.variable(3, N+1)  
    B = opti.variable(1, N+1)  
    a = opti.variable(3, N)    
    
    # SLACK VARIABLES per evitare i minimi locali
    eps_obs = opti.variable(k_obs, N+1)

    # --- Parameters ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    p_obs_closest = opti.parameter(3, (N+1) * k_obs) 
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)
    p_ego_prev = opti.parameter(3, N+1)

    # --- COST FUNCTION ---
    cost = 0
    # 1. Crea il dizionario per tracciare le componenti
    cost_components = {
        "waypoints": 0,
        "effort": 0,
        "battery": 0,
        "z_ref": 0
    }

    # wp_priorities = [1.0, 0.4, 0.1] 

    wp_term = 0
    eff_term = 0
    z_term = 0
    batt_term = 0

    # Adjust increasing priority of reaching the closest waypoint as the horizon reaches the end
    wp_priorities = np.linspace(0.1, 1, N+1)

    # Task cost: Reach waypoints

    for i in range(num_regions):
        # Determine weight once per region (no longer inside the k-loop)
        weight = w_seen * wp_priorities[N] if i < len(wp_priorities) else w_seen * 0.01
        
        # Calculate term using only the N-th timestep
        # We use p[:, N] instead of p[:, k]
        wp_term = (1 - flag[i]) * ca.sumsqr(p[:, N] - p_wp[:, i]) * weight
        
        # Add to tracker and total cost
        cost_components["waypoints"] += wp_term
        cost += wp_term
    '''
    FOR THE WHOLE TIME HORIZON CONSIDER THIS LOOP INSTEAD
    for i in range(num_regions):
        for k in range(1, N + 1): 
            # Assegna il peso specifico in base all'ordine di vicinanza
            weight = w_seen * wp_priorities[k] if i < len(wp_priorities) else w_seen * 0.01
            wp_term += (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * weight
            # Aggiungilo al tracker e al costo totale
            cost_components["waypoints"] += wp_term
            cost += wp_term
    '''
    '''
    TO WEIGH EACH STEP WITH THE SAME PENALTY CONSIDER THIS LOOP
    for i in range(num_regions):
        for k in range(1, N + 1): 
            cost += (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * w_seen
    '''

       # Control Effort, Battery, and Z-Reference
    for k in range(N):
        eff_term = w_effort * ca.sumsqr(a[:, k])
        cost_components["effort"] += eff_term
        cost += eff_term
        
        batt_term = w_batt * ca.sumsqr(v[:, k])
        cost_components["battery"] += batt_term
        cost += batt_term
        
        z_term = w_z * ca.sumsqr(p[2, k] - z_ref)
        cost_components["z_ref"] += z_term
        cost += z_term

    opti.minimize(cost)

    # --- DYNAMICS CONSTRAINTS ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)

    for k in range(N):
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)

    # --- PHYSICAL BOUNDS ---
    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))
    
    # Nonlinear obstacle distance
    for k in range(1, N+1):
        for j in range(k_obs):
            col_idx = k * k_obs + j
            # Calcola la distanza esatta (non lineare) tra X,Y del drone e X,Y dell'ostacolo
            dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
            
            # Il vincolo puro del cerchio 
            opti.subject_to(dist_sqr  >= safe_rad**2)
    
    # --- NEIGHBOR AVOIDANCE ---
    for j in range(num_neighbors):
        for k in range(N+1):
            col_idx = j * (N+1) + k
            dp_bar = p_ego_prev[:, k] - p_neighbors[:, col_idx]
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)
    
    # --- MAP BOUNDARIES ---
    # Apply the bounds to every step in the prediction horizon
    for k in range(1, N+1):
        opti.subject_to(opti.bounded(x_min, p[0, k], x_max))
        opti.subject_to(opti.bounded(y_min, p[1, k], y_max))
        opti.subject_to(opti.bounded(z_min, p[2, k], z_max))

    # CasADi Plugin Options
    p_opts = {
        "expand": True, 
        "print_time": False  # Disables the CasADi 'Elapsed time' printout
    }

    # IPOPT Solver Options (NO "ipopt." prefix here!)
    s_opts = {
        "print_level": 0,    # Levels 0-12 (0 is silent, 5 is default)
        "sb": "yes"          # Skips the IPOPT banner
    }

    opti.solver("ipopt", p_opts, s_opts)

    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions, "k_obs": k_obs, "cost_components": cost_components
    }

def setup_test_MPC(num_neighbors=0, enable_obstacles=False): 
    """
    Simplified MPC setup for debugging a single drone.
    Set enable_obstacles=False for a completely empty environment.
    """
    cost_cfg = config["cost"]
    constraints_cfg = config["constraints"]
    mpc_cfg = config["mpc"]

    w_seen = cost_cfg["w_seen"]
    w_effort = cost_cfg["w_effort"]
    w_batt = cost_cfg["w_battery"]
    z_ref = cost_cfg["z_ref"]
    w_z = cost_cfg["w_z"] 
    w_slack = cost_cfg["w_slack_collision"] 
    w_barrier = cost_cfg["w_barrier"]

    max_vel = constraints_cfg["max_speed"]
    max_acc = constraints_cfg["max_acceleration"]
    safe_rad = constraints_cfg["safe_distance"] 

    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["timestep"]
    num_regions = mpc_cfg["k_wp_search"]
    k_obs = mpc_cfg["k_obs_search"]

    x_min, x_max = bounds_cfg["x_bounds"]
    y_min, y_max = bounds_cfg["y_bounds"]
    z_min, z_max = bounds_cfg["z_bounds"]

    opti = ca.Opti()

    # --- Variables (Names kept identical for reusability) ---
    p = opti.variable(3, N+1)  
    v = opti.variable(3, N+1)  
    B = opti.variable(1, N+1)  
    a = opti.variable(3, N)    
    eps_obs = opti.variable(k_obs, N+1)

    # --- Parameters ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    p_obs_closest = opti.parameter(3, (N+1) * k_obs) 
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)
    p_ego_prev = opti.parameter(3, N+1)

    # --- COST FUNCTION ---
    cost = 0
    cost_components = {"waypoints": 0, "effort": 0, "battery": 0, "z_ref": 0, "slack": 0, "barrier": 0}
    wp_priorities = np.linspace(0.1, 1, N+1)

    # 1. Waypoints

    for i in range(num_regions):
        # i = 0 is the closest unseen waypoint. We give it 100% focus.
        # Future waypoints in the array get 0% focus so they don't drag the drone backward.
        target_focus = 1.0 if i == 0 else 0.0
        wp_term = 0 
        for k in range(1, N + 1): 
            # Assegna il peso specifico in base all'ordine di vicinanza
            weight = w_seen * wp_priorities[k] if i < len(wp_priorities) else w_seen * 0.01
            wp_term = (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * weight * target_focus
            # Aggiungilo al tracker e al costo totale
            cost_components["waypoints"] += wp_term
            cost += wp_term
    '''
    UNCOMMENT THIS TO USE TERMINAL COST INSTEAD OF RUNNING + PENALTY TO GET TO THE WAYPOINT INCREASING IN THE HORIZON 
    for i in range(num_regions):
    
        # FIX: The Hierarchy. 
        # i = 0 is the closest unseen waypoint. We give it 100% focus.
        # Future waypoints in the array get 0% focus so they don't drag the drone backward.
        target_focus = 1.0 if i == 0 else 0.0 
    
                # Determine weight once per region (no longer inside the k-loop)
        weight = w_seen * wp_priorities[N] if i < len(wp_priorities) else w_seen * 0.01
        
        # Calculate term using only the N-th timestep
        # We use p[:, N] instead of p[:, k]
        wp_term = (1 - flag[i]) * ca.sumsqr(p[:, N] - p_wp[:, i]) * weight * target_focus
    
        # Add to tracker and total cost
        cost_components["waypoints"] += wp_term
            cost += wp_term
        '''
    # 2. Control Effort & Z-Reference
    for k in range(N):
        eff_term = w_effort * ca.sumsqr(a[:, k])
        cost_components["effort"] += eff_term
        cost += eff_term
        
        z_term = w_z * ca.sumsqr(p[2, k] - z_ref)
        cost_components["z_ref"] += z_term
        cost += z_term
        
        batt_term = w_batt * ca.sumsqr(v[:, k])
        cost_components["battery"] += batt_term
        cost += batt_term

    # 3. MICRO-PENALTIES (To prevent IPOPT from crashing on unused variables)
    cost += 1e-8 * ca.sumsqr(B) # B is declared but unused in constraints

    if enable_obstacles:
        pass # UNCOMMENT IF YOU WANT TO INTRODUCE SLACK VARIABLE IN COST FUNCTION FOR THE COLLISIONS
        '''
        slack_term = 0
        for k in range(1, N + 1):
            for j in range(k_obs):
                slack_term += w_slack * ca.sumsqr(eps_obs[j, k])
        cost += slack_term
        cost_components["slack"] += slack_term
        '''
    else:
        # If obstacles are off, eps_obs is a "ghost" variable. We give it a tiny 
        # dummy cost so the matrix isn't singular, completely bypassing the crash.
        cost += 1e-8 * ca.sumsqr(eps_obs)

        # --- OPTIONAL OBSTACLES ---
    if enable_obstacles:
        step_barrier = 0
        for k in range(1, N+1):
            for j in range(k_obs):
                opti.subject_to(eps_obs[j, k] >= 0)
                col_idx = k * k_obs + j
                dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
                
                # 1. Soft Constraint
                opti.subject_to(dist_sqr + eps_obs[j, k] >= safe_rad**2)
                
                # 2. The Exponential Forcefield
                # 'shape_factor' controls how WIDE the forcefield is. 
                # Higher number = wider, softer warning track. Lower = tighter, sharper wall.
                shape_factor = cost_cfg["shape_factor"]
                
                # Formula: e^( (safe_rad^2 - dist_sqr) / shape_factor )
                # As dist_sqr approaches safe_rad^2, the exponent becomes 0, so exp(0) = 1.
                # The penalty at the exact boundary is exactly equal to w_barrier.
                step_barrier = w_barrier * ca.exp((safe_rad**2 - dist_sqr) / shape_factor)

                # FIX: Use += to ACCUMULATE the penalty
                cost_components["barrier"] += step_barrier
                cost += step_barrier
    
    opti.minimize(cost)

    # --- DYNAMICS CONSTRAINTS ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)

    for k in range(N):
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)

    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))

            # --- OPTIONAL OBSTACLES ---
    if enable_obstacles:
        for k in range(1, N+1):
            for j in range(k_obs):
                opti.subject_to(eps_obs[j, k] >= 0)
                col_idx = k * k_obs + j
                dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
                
                # 1. Soft Constraint
                opti.subject_to(dist_sqr + eps_obs[j, k] >= safe_rad**2)
    
    # --- NEIGHBOR AVOIDANCE ---
    
    for j in range(num_neighbors):
        for k in range(N+1):
            col_idx = j * (N+1) + k
            dp_bar = p_ego_prev[:, k] - p_neighbors[:, col_idx]
            dist_bar_sqr = ca.sumsqr(dp_bar)
            linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
            opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)
    
    # --- MAP BOUNDARIES ---
    for k in range(1, N+1):
        opti.subject_to(opti.bounded(x_min, p[0, k], x_max))
        opti.subject_to(opti.bounded(y_min, p[1, k], y_max))
        
        # FIX: Sink the mathematical floor so starting at Z=0 is strictly "inside" the bounds
        opti.subject_to(opti.bounded(-0.1, p[2, k], z_max))

        # CasADi Plugin Options
    p_opts = {
        "expand": True, 
        "print_time": False  # Disables the CasADi 'Elapsed time' printout
    }

    # IPOPT Solver Options (NO "ipopt." prefix here!)
    s_opts = {
        "print_level": 0,    # Levels 0-12 (0 is silent, 5 is default)
        "sb": "yes"          # Skips the IPOPT banner
    }

    opti.solver("ipopt", p_opts, s_opts)

    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "p_obs_closest": p_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions, "k_obs": k_obs, 
        "cost_components": cost_components, "eps_obs": eps_obs
    }

def run_mpc_iteration(mpc_vars, current_state, waypoint_coords,  
                      last_traj, neighbor_trajs, obs_tree, n_iter_mpc, t_solve_avg):
    """
    Executes one step of the MPC.
    waypoint_coords: [M x 3] numpy array [x, y, seen_flag]
    """

    opti = mpc_vars["opti"]
    k_limit = mpc_vars["k_search"]
    k_obs = mpc_vars["k_obs"] 

    # needed inizialization if no obstacles are detected for the waypoint kd tree
    k_query = 0

    # --- 1. WAYPOINT SEARCH ---
    # Create a boolean mask of only the waypoints that have NOT been seen
    unseen_mask = waypoint_coords[:, 2] == 0
    active_waypoints = waypoint_coords[unseen_mask]
    
    num_available = active_waypoints.shape[0]
    
    # If there are no more active waypoints, the drone is done!
    if num_available == 0:
        # Feed the current position as the target so it just hovers in place smoothly
        closest_coords_2d = np.tile(current_state["p"][:2], (k_limit, 1))
        closest_flags = np.ones(k_limit) # Set flags to 1 so cost is 0
        k_query = k_limit
    else:
        k_query = min(k_limit, num_available)
        
        # Build the KDTree ONLY with the active (unseen) waypoints
        wp_tree = KDTree(active_waypoints[:, :2])
        dist, indices = wp_tree.query(current_state["p"][:2], k=k_query)
        
        if k_query == 1: 
            indices = [indices]
        
        # IMPORTANT: Extract from active_waypoints, not the original waypoint_coords!
        closest_coords_2d = active_waypoints[indices, :2]
        closest_flags = active_waypoints[indices, 2]
    
    if k_query == 1: 
        indices = [indices]

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
        
    flattened_neighbors = neighbor_trajs.reshape((3, -1), order='F')
    
    # Check how many columns the CasADi parameter actually expects
    expected_cols = mpc_vars["p_neighbors"].shape[1]
    
    if expected_cols == 0:
        # If the solver expects 0 neighbors (like in our test setup), feed it an empty array
        opti.set_value(mpc_vars["p_neighbors"], np.empty((3, 0)))
    else:
        # Otherwise, feed it the actual neighbor data
        opti.set_value(mpc_vars["p_neighbors"], flattened_neighbors)

    try:

        # Start the high-resolution timer right before the solve step
        start_time = time.perf_counter()
        sol = opti.solve()
        # Stop the timer immediately after
        end_time = time.perf_counter()

        # Calculate the elapsed time
        solve_time = end_time - start_time
        
        # 2. Valuta i singoli componenti numerici
        comp_vals = {}
        for name, sym_term in mpc_vars["cost_components"].items():
            comp_vals[name] = sol.value(sym_term)
        cost_value = sol.value(mpc_vars["opti"].f)
        new_trajectory = sol.value(mpc_vars["p"])
        optimal_accel = sol.value(mpc_vars["a"])
        
        t_solve_avg += solve_time
        n_iter_mpc += 1
        t_solve_avg = t_solve_avg/n_iter_mpc
        # solver stats
        # when using ipopt 
        # solve_time = sol.stats()['t_wall_total']
        # print(f"MPC solve successful: {solve_time:.4f}s") 
        
        return optimal_accel[:, 0], new_trajectory, cost_value, t_solve_avg, n_iter_mpc, comp_vals

    except RuntimeError:
        print("MPC solve failed! Using safety fallback.")
        cost_value = np.inf
        fallback_components = {name: 0.0 for name in mpc_vars["cost_components"].keys()} 
        return np.array([0.0, 0.0, 0.0]), last_traj, cost_value, t_solve_avg, n_iter_mpc, fallback_components