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

def setup_test_MPC_QP(num_neighbors=0, enable_obstacles=False): 
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
    num_wp = mpc_cfg["k_wp_search"]
    k_obs = mpc_cfg["k_obs_search"]

    x_min, x_max = bounds_cfg["x_bounds"]
    y_min, y_max = bounds_cfg["y_bounds"]
    z_min, z_max = bounds_cfg["z_bounds"]

    opti = ca.Opti('conic')

    # --- Variables ---
    p = opti.variable(3, N+1)  
    v = opti.variable(3, N+1)  
    B = opti.variable(1, N+1)  
    a = opti.variable(3, N)    
    eps_obs = opti.variable(k_obs, N+1)
    eps_neigh = opti.variable(num_neighbors, N+1) 

    opti.set_initial(eps_obs, 0.01)
    opti.set_initial(eps_neigh, 0.01)

    # --- Parameters ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    p_obs_closest = opti.parameter(3, (N+1) * k_obs) 
    r_obs_closest = opti.parameter((N+1) * k_obs)
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)
    p_wp = opti.parameter(3, num_wp) 
    flag = opti.parameter(num_wp)
    p_ego_prev = opti.parameter(3, N+1)
    a_ego_prev = opti.parameter(3)

    w_seen = opti.parameter(1)
    target_focus = opti.parameter(num_wp)

    # --- COST FUNCTION ---
    cost = 0
    cost_components = {"waypoints": 0, "effort": 0, "battery": 0, "z_ref": 0, "slack": 0, "barrier": 0}
    #wp_priorities = np.linspace(1,5, N+1)**2
    wp_priorities = np.ones(N+1)

    # 1. Waypoints
    for i in range(num_wp):
        # i = 0 is the closest unseen waypoint. We give it 100% focus.
        # Future waypoints in the array get 0% focus so they don't hold back the drone
        # target_focus = 1.0 if i == 0 else 0.0
        wp_term = 0 
        for k in range(1, N + 1): 
            # Assegna il peso specifico in base all'ordine di vicinanza
            weight = w_seen * wp_priorities[k] if i < len(wp_priorities) else w_seen * 0.01
            wp_term = (1 - flag[i]) * ca.sumsqr(p[:, k] - p_wp[:, i]) * weight * target_focus[i]
            # Aggiungilo al tracker e al costo totale
            cost_components["waypoints"] += wp_term
            cost += wp_term
    '''
    UNCOMMENT THIS TO USE TERMINAL COST INSTEAD OF RUNNING + PENALTY TO GET TO THE WAYPOINT INCREASING IN THE HORIZON 
    for i in range(num_wp):
    
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

    # 2. Control Effort AKA Jerk limitation & Z-Reference & Battery
    for k in range(N):
        
        # --- JERK MATH ---
        if k == 0:
            jerk = a[:, 0] - a_ego_prev
        else:
            jerk = a[:, k] - a[:, k-1]
            
        eff_term = w_effort * ca.sumsqr(jerk)
        cost_components["effort"] += eff_term
        cost += eff_term
        
        z_term = w_z * ca.sumsqr(p[2, k] - z_ref)
        cost_components["z_ref"] += z_term
        cost += z_term
        
        batt_term = w_batt * ca.sumsqr(v[:, k])
        cost_components["battery"] += batt_term
        cost += batt_term
                
        # --- OBSTACLES (Linearized for QP with Dynamic Radii) ---
    if enable_obstacles:
        slack_term = 0
        step_barrier = 0
        
        for k in range(1, N+1):
            for j in range(k_obs):

                # 1. Slack Cost 
                slack_term += w_slack * ca.sumsqr(eps_obs[j, k])
                
                col_idx = k * k_obs + j
                
                # --- DYNAMIC RADII MATH ---
                # Extract the specific radius for this obstacle at this timestep
                current_obs_radius = r_obs_closest[col_idx]
                
                # The "Brick Wall" distance
                total_safe_dist = safe_rad + current_obs_radius
                
                # The "Warning Track" distance (e.g., 1 meter out from the brick wall)
                # You can change the 1.0 to a variable like warn_margin if you put it in JSON
                dist_influence = total_safe_dist + 1.0 
                
                # Vector from obstacle center to drone's PREVIOUS predicted position
                dp_bar = p_ego_prev[:2, k] - p_obs_closest[:2, col_idx]
                dist_bar_sqr = ca.sumsqr(dp_bar)

                # Quadratic Barrier Function (FIXED: **2 instead of *2)
                barrier_val = w_barrier * ca.fmax(0, dist_influence**2 - dist_bar_sqr)**2
                step_barrier += barrier_val
                
                # First-order Taylor Expansion (The Separating Hyperplane)
                linear_term = 2 * ca.dot(dp_bar, (p[:2, k] - p_ego_prev[:2, k]))
            
                # The Convex Constraint
                opti.subject_to(dist_bar_sqr + linear_term + eps_obs[j, k] >= total_safe_dist**2)
                
        # Add to total cost
        cost += slack_term
        cost += step_barrier
        cost_components["slack"] += slack_term
        cost_components["barrier"] += step_barrier
        
    else:
        cost += 1e-8 * ca.sumsqr(eps_obs)
                
    opti.minimize(cost)

    # --- DYNAMICS CONSTRAINTS ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)

    #slack variables must be positive
    opti.subject_to(ca.vec(eps_obs) >= 0)
    opti.subject_to(ca.vec(eps_neigh) >= 0)

    # model kinematics constraints
    for k in range(N):
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)

    # state constraints
    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))

    # --- OPTIONAL OBSTACLES ---
    # UNCOMMENT IF SLACK VARIABLE FOR OBSTACLES ARE REMOVED
    '''
    if enable_obstacles:
        for k in range(1, N+1):
            for j in range(k_obs):
                opti.subject_to(eps_obs[j, k] >= 0)
                col_idx = k * k_obs + j
                dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
                
                # 1. Soft Constraint
                opti.subject_to(dist_sqr + eps_obs[j, k] >= safe_rad**2)
    '''    
    # --- NEIGHBOR AVOIDANCE ---
    if num_neighbors > 0:
        slack_neigh_term = 0
        for j in range(num_neighbors):
            for k in range(1, N+1): # FIX 1: Start at k=1, not k=0
                col_idx = j * (N+1) + k
                dp_bar = p_ego_prev[:, k] - p_neighbors[:, col_idx]
                dist_bar_sqr = ca.sumsqr(dp_bar)
                linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
                
                # FIX 2: Add eps_neigh to soften the constraint
                opti.subject_to(dist_bar_sqr + linear_term + eps_neigh[j, k] >= safe_rad**2)
                
                # Add to local slack accumulator
                slack_neigh_term += w_slack * ca.sumsqr(eps_neigh[j, k])
        
        # Apply the accumulated penalty to the solver cost
        cost += slack_neigh_term
        cost_components["slack"] += slack_neigh_term
    else:
        # Dummy cost to prevent singular matrices when 0 neighbors
        cost += 1e-8 * ca.sumsqr(eps_neigh)

    # --- MAP BOUNDARIES ---
    for k in range(1, N+1):
        opti.subject_to(opti.bounded(x_min, p[0, k], x_max))
        opti.subject_to(opti.bounded(y_min, p[1, k], y_max))
        
        # Sink the mathematical floor so starting at Z=0 is strictly "inside" the bounds
        opti.subject_to(opti.bounded(-0.1, p[2, k], z_max))

   # CasADi Plugin Options (These remain mostly the same)
    p_opts = {
        "expand": True, 
        "print_time": False,
        "error_on_fail": True # Ensures your try/except block catches failures cleanly
    }

    # OSQP Solver Options (Completely different from IPOPT!)
    s_opts = {
        "verbose": False,         # OSQP's version of print_level=0 and sb="yes"
        "max_iter": 10000,        # OSQP takes more micro-iterations than IPOPT. Give it headroom.
        "eps_abs": 1e-6,          # Absolute tolerance (Loosened slightly for stability)
        "eps_rel": 1e-6,          # Relative tolerance
        "polish": True            # CRITICAL: Runs a secondary solver step to guarantee high accuracy
    }

    opti.solver("osqp", p_opts, s_opts)

    return {
        "opti": opti, "p": p, "a": a, "p_init": p_init, 
        "v_init": v_init, "B_init": B_init, "p_wp": p_wp, 
        "flag": flag, "p_ego_prev": p_ego_prev, 
        "a_ego_prev": a_ego_prev, 
        "p_obs_closest": p_obs_closest,"r_obs_closest": r_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_wp, "k_obs": k_obs, 
        "cost_components": cost_components, "eps_obs": eps_obs, "eps_neigh": eps_neigh, 
        "w_seen": w_seen, "target_focus": target_focus
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
    accel_ego_prev = opti.parameter(3, N+1)

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
    #wp_priorities = np.linspace(0.1, 1, N+1)
    wp_priorities = np.ones(N + 1)

    # Task cost: Reach waypoints

    for i in range(num_regions):
        # Determine weight once per region (no longer inside the k-loop)
        weight = w_seen * wp_priorities[N] if i < len(wp_priorities) else w_seen * 0.01
        
        # Calculate term using only the N-th timestep
        # We use p[:, N] instead of p[:, k]
        wp_term = (1 - flag[i]) * ca.sqrt(ca.sumsqr(p[:, k] - p_wp[:, i]) + 1e-4) * weight
        
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
        "flag": flag, "p_ego_prev": p_ego_prev, "a_ego_prev": accel_ego_prev,
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

    opti.set_initial(eps_obs, 0.01)

    # --- Parameters ---
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    
    p_obs_closest = opti.parameter(3, (N+1) * k_obs)
    r_obs_closest = opti.parameter((N+1) * k_obs) 
    p_neighbors = opti.parameter(3, (N+1) * num_neighbors)
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)
    p_ego_prev = opti.parameter(3, N+1)
    a_ego_prev = opti.parameter(3)

    # --- COST FUNCTION ---
    cost = 0
    cost_components = {"waypoints": 0, "effort": 0, "battery": 0, "z_ref": 0, "slack": 0, "barrier": 0}
    wp_priorities = np.ones(N+1)

    # 1. Waypoints

    for i in range(num_regions):
        # i = 0 is the closest unseen waypoint. We give it 100% focus.
        # Future waypoints in the array get 0% focus so they don't drag the drone backward.
        target_focus = 1.0 if i == 0 else 0.0
        wp_term = 0 
        for k in range(1, N + 1): 
            # Assegna il peso specifico in base all'ordine di vicinanza
            weight = w_seen * wp_priorities[k] if i < len(wp_priorities) else w_seen * 0.01
            wp_term = (1 - flag[i]) * ca.sqrt(ca.sumsqr(p[:, k] - p_wp[:, i]) + 1e-4) * weight * target_focus
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

    # 2. Control Effort AKA Jerk & Z-Reference & Battery
    for k in range(N):
        
        # --- JERK MATH ---
        if k == 0:
            jerk = a[:, 0] - a_ego_prev
        else:
            jerk = a[:, k] - a[:, k-1]
            
        eff_term = w_effort * ca.sumsqr(jerk)
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

    # --- OBSTACLES (Slack Cost, Barrier Cost, & Constraints) ---
    if enable_obstacles:
        slack_term = 0
        step_barrier = 0
        
        for k in range(1, N+1):
            for j in range(k_obs):
                # 1. Slack Cost (L1 Linear)
                slack_term += w_slack * ca.sumsqr(eps_obs[j, k])
                
                # 2. Hard Constraints & Soft Escape
                # opti.subject_to(eps_obs[j, k] >= 0)
                col_idx = k * k_obs + j
                dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
                opti.subject_to(dist_sqr + eps_obs[j, k] >= safe_rad**2)
                
                # 3. Exponential Barrier
                shape_factor = cost_cfg["shape_factor"]
                step_barrier_val = w_barrier * ca.exp((safe_rad**2 - dist_sqr) / shape_factor)
                step_barrier += step_barrier_val

        # Add everything to the total cost once the loop finishes
        cost += slack_term
        cost_components["slack"] += slack_term
        
        cost += step_barrier
        cost_components["barrier"] += step_barrier
        
    else:
        # Dummy cost to prevent singular matrices
        cost += 1e-8 * ca.sumsqr(eps_obs)
    
    opti.minimize(cost)

    # --- DYNAMICS CONSTRAINTS ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)

    #slack variables must be positive
    opti.subject_to(ca.vec(eps_obs) >= 0)

    for k in range(N):
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt + 0.5 * a[:, k] * dt**2)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)

    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))

    # --- OPTIONAL OBSTACLES ---
    # UNCOMMENT IF SLACK VARIABLE FOR OBSTACLES ARE REMOVED
    '''
    if enable_obstacles:
        for k in range(1, N+1):
            for j in range(k_obs):
                opti.subject_to(eps_obs[j, k] >= 0)
                col_idx = k * k_obs + j
                dist_sqr = ca.sumsqr(p[:2, k] - p_obs_closest[:2, col_idx])
                
                # 1. Soft Constraint
                opti.subject_to(dist_sqr + eps_obs[j, k] >= safe_rad**2)
    '''
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
        "a_ego_prev": a_ego_prev, 
        "p_obs_closest": p_obs_closest, "r_obs_closest": r_obs_closest, "p_neighbors": p_neighbors,
        "k_search": num_regions, "k_obs": k_obs, 
        "cost_components": cost_components, "eps_obs": eps_obs, "w_seen": w_seen
    }

def run_mpc_iteration(mpc_vars, current_state, waypoint_coords,  
                      last_traj, neighbor_trajs, obs_tree, obstacles, current_w_seen, current_target_focus):
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
        
        # In run_mpc_iteration, sostituisci la ricerca KDTree con:
        unseen_indices = np.where(waypoint_coords[:, 2] == 0)[0]
        if len(unseen_indices) > 0:
            # Prendi i primi 'k_limit' waypoint nell'ordine prestabilito
            top_indices = unseen_indices[:k_limit]
            closest_coords_2d = waypoint_coords[top_indices, :2]
            closest_flags = waypoint_coords[top_indices, 2]

    # --- 2. DYNAMIC PADDING ---
    final_coords_2d = closest_coords_2d
    final_flags = closest_flags
    
    if k_query < k_limit:
        padding_count = k_limit - k_query
        last_coord = closest_coords_2d[-1:, :]
        final_coords_2d = np.vstack([closest_coords_2d] + [last_coord] * padding_count)
        final_flags = np.append(closest_flags, [1.0] * padding_count)
    
    # Conversione in 3D (p_wp)
    # Modificato, ora fa in modo che z_padding sia uguale a z_ref e quindi il riferimento coincida con closest_coords_3d
    z_val = config["cost"]["z_ref"]
    z_padding = np.full((k_limit, 1), z_val)
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
    
    r_obs_closest_array = np.zeros(num_points * k_obs) 
    
    for k in range(num_points):
        for j in range(k_obs):
            obs_idx = indices_obs[k, j]
            col_idx = k * k_obs + j
            
            # Fill the coordinates
            closest_obs_coords[:, col_idx] = obs_tree.data[obs_idx]
            
            # Extract the actual radius from your global obstacles list
            r_obs_closest_array[col_idx] = obstacles[obs_idx].radius 

     # --- 4. SET PARAMETERS ---
    opti.set_value(mpc_vars["p_init"], current_state["p"])
    opti.set_value(mpc_vars["v_init"], current_state["v"])
    opti.set_value(mpc_vars["B_init"], current_state["B"])
    
    opti.set_value(mpc_vars["p_wp"], closest_coords_3d.T)
    opti.set_value(mpc_vars["flag"], final_flags)
    
    opti.set_value(mpc_vars["p_ego_prev"], last_traj)
    opti.set_value(mpc_vars["p_obs_closest"], closest_obs_coords)
    opti.set_value(mpc_vars["r_obs_closest"], r_obs_closest_array) 
    opti.set_value(mpc_vars["a_ego_prev"], current_state["a"])

    opti.set_value(mpc_vars["w_seen"], current_w_seen)
    opti.set_value(mpc_vars["target_focus"], current_target_focus)
        
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
        
        # solver stats
        # when using ipopt 
        # solve_time = sol.stats()['t_wall_total']
        # print(f"MPC solve successful: {solve_time:.4f}s") 
        
        return optimal_accel[:, 0], new_trajectory, cost_value, comp_vals, solve_time

    except RuntimeError:
        print(f"Drone {current_state.get('id', 'unknown')} MPC solve failed! Safety braking.")
        cost_value = np.inf
        # Applica una frenata decisa invece di lasciarlo scivolare
        braking_accel = -current_state["v"] 
        fallback_components = {name: 0.0 for name in mpc_vars["cost_components"].keys()}
        solve_time = 0 # fallback shouldn't contaminate the real value
        return braking_accel, last_traj, cost_value, fallback_components, solve_time   

def run_swarm_simulation(drones, dt, max_iter, config, obstacles, obs_tree, dist_threshold, early_switch_flag, PRINT_INTERVAL=10):
    """
    Executes the MPC loop for the entire swarm until all drones are parked or max_iter is reached.
    """
    num_iter = 0
    switch_distance = dist_threshold + 0.15
    dJ_thresh = 1e-6
    prev_total_cost = 1e6
    
    total_solver_time = 0.0
    total_solver_calls = 0

    # Initialize the history tracker 
    cost_history = {
        "total": [], "waypoints": [], "effort": [], 
        "battery": [], "z_ref": [], "barrier": [], "slack": []
    }

    print("\nStarting Swarm MPC Simulation...")
    
    while num_iter <= max_iter:
        total_loop_cost = 0 
        
        # Check if ALL drones have finished their tasks
        if all(d.is_parked for d in drones):
            print(f"\nMission accomplished in {num_iter} steps!")
            average_time = total_solver_time / total_solver_calls if total_solver_calls > 0 else 0
            print(f"Avg Solve Time: {average_time:.5f} seconds")
            break

        for i, drone in enumerate(drones):
            
            # --- 1. MISSION STATE CHECK ---
            unseen_mask = drone.waypoints[:, 2] == 0
            if not np.any(unseen_mask) and not drone.returning_home:
                print(f"Drone {drone.id} finished mission! Returning home.")
                home_wp = np.array([drone.home_pos[0], drone.home_pos[1], 0])
                drone.waypoints = np.vstack([drone.waypoints, home_wp])
                drone.returning_home = True

            # --- 2. TARGET IDENTIFICATION (Early Switch Logic) ---
            unseen_wps = drone.waypoints[drone.waypoints[:, 2] == 0]
            current_focus_vector = np.zeros(config["mpc"]["k_wp_search"])
            
            if len(unseen_wps) > 0:
                current_target = unseen_wps[0, :3]
                dist_to_current = np.linalg.norm(drone.state["p"][:2] - current_target[:2])
                if early_switch_flag == True:
                    if dist_to_current > switch_distance or len(unseen_wps) < 2:
                        current_focus_vector[0] = 1.0 
                    else:
                        current_focus_vector[1] = 1.0  
                else: 
                    current_focus_vector[0] = 1.0
            
            # --- 3. PARKING LOGIC ---
            if drone.returning_home and not drone.is_parked:
                dist_to_home = np.linalg.norm(drone.state["p"][:2] - drone.home_pos[:2])
                if dist_to_home < dist_threshold:
                    print(f"Drone {drone.id} has parked safely!")
                    drone.is_parked = True 

            if drone.is_parked:
                drone.state["v"] = np.zeros(3)
                drone.state["a"] = np.zeros(3)
                
                N = drone.mpc_vars["p"].shape[1] - 1
                stationary_traj = np.tile(drone.state["p"], (N+1, 1)).T
                drone.last_traj = stationary_traj
                
                drone.history_p.append(drone.state["p"].copy())
                drone.history_a.append(np.zeros(3))
                drone.history_v.append(np.zeros(3))
                if hasattr(drone, 'history_predictions'):
                    drone.history_predictions.append(stationary_traj)
                continue 
            
            # --- 4. RUN MPC ---
            neighbor_trajs = [d.last_traj for d in drones if d.id != drone.id]
            if len(neighbor_trajs) > 0:
                neighbor_trajs_array = np.stack(neighbor_trajs, axis=2)
            else:
                neighbor_trajs_array = np.empty((3, drone.mpc_vars["p"].shape[1], 0))

            current_w_seen = config["cost"]["w_seen_rth"] if drone.returning_home else config["cost"]["w_seen"]

            accel, new_traj, current_cost_value, cost_breakdown, t_solve_mpc = run_mpc_iteration(
                drone.mpc_vars, drone.state, drone.waypoints, 
                drone.last_traj, neighbor_trajs_array, obs_tree, obstacles, 
                current_w_seen, current_focus_vector
            )

            total_solver_time += t_solve_mpc
            total_solver_calls += 1

            # --- 5. LOGGING ---
            if current_cost_value != np.inf: 
                cost_history["total"].append(current_cost_value)
                for key, val in cost_breakdown.items():
                    if key not in cost_history: cost_history[key] = []
                    cost_history[key].append(val)
            total_loop_cost += current_cost_value

            drone.drone_model(accel, dt)
            if early_switch_flag == False:
                drone.check_waypoints(dist_threshold)
            else : 
                drone.check_waypoints(switch_distance)
            drone.log_telemetry(new_traj)
            drone.last_traj = new_traj
            drone.history_a.append(accel)
            drone.history_v.append(drone.state["v"].copy())
            drone.state["a"] = accel

            if num_iter % PRINT_INTERVAL == 0:
                print(f"Step {num_iter} | Drone {drone.id} | Cost: {current_cost_value:.2f}")
        
        # Convergence Check
        if abs(prev_total_cost - total_loop_cost) < dJ_thresh:
            print("Swarm converged to steady state!")
            break
            
        prev_total_cost = total_loop_cost
        num_iter += 1

    return drones, cost_history     