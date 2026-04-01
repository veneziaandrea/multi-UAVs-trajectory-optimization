from config import load_config
import sys
from pathlib import Path
import casadi as ca
import numpy as np
from scipy.spatial import KDTree

# FIRST ATTEMPT: KEEP IT AS A QP BY LINEARIZING THE DISTANCE FROM OBSTACLES/OTHER DRONES CONSTRAINT
def MPC_QP (waypoints, obs_tree): 

    # Set the root directory and add the source directory to the Python path
    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    config_path = ROOT / "configs" / "optimization.json"
    config = load_config(config_path)

    # unpack the variables of interest from the json file
    cost_cfg = config["cost"]
    constraints_cfg = config["constraints"]
    mpc_cfg = config["mpc"]

    # cost function parameters
    w_seen = cost_cfg["w_seen"]
    w_effort = cost_cfg["w_effort"]
    z_ref = cost_cfg["z_ref"]

    # bounds on acceleration, speed, obstacle avoidance
    max_vel = constraints_cfg["max_speed"]
    max_acc = constraints_cfg["max_acceleration"]
    safe_rad = constraints_cfg["safe_distance"] # safety distance from obstacles

    # mpc problem parameters
    N = mpc_cfg["prediction_horizon"]
    dt = mpc_cfg["config"]
    num_regions = waypoints.size(axis = 0) # one region big as FOV per waypoint

    # OPTIMIZATION PROBLEM START
    opti = ca.Opti()

    # --- Optimization variables ---
    p = opti.variable(3, N+1)  
    v = opti.variable(3, N+1)  
    B = opti.variable(1, N+1)  
    a = opti.variable(3, N) # input  

    # --- Parameters (Updated each iteration of the MPC) ---
    # "warm start" or update of initial position
    p_init = opti.parameter(3) 
    v_init = opti.parameter(3)
    B_init = opti.parameter(1)
    p_obs_closest= opti.parameter(3, N+1)

    # Flags and waypoints 
    p_wp = opti.parameter(3, num_regions) 
    flag = opti.parameter(num_regions)

    # Collision constraints linearization
    # p_obs will be the closest obstacle at the end of the prediction horizon

    # Ego drone's planned trajectory from the previous MPC step
    # We need this to evaluate the Taylor expansion at a known point
    p_ego_prev = opti.parameter(3, N+1)

    # --- Cost function computation ---
    cost = 0

    # cost to visit every waypoint
    for i in range(num_regions):
        # Penalize the distance from the postion at the end of predicition horizon (p[:, N]) and the waypoint
        cost += (1 - flag[i]) * ca.sumsqr(p[:, N] - p_wp[:, i]) * w_seen

    # cost to reduce the control effort
    for k in range(N):
        cost += w_effort * ca.sumsqr(a[:, k])
        
    opti.minimize(cost)

    # --- Dynamics Constraints ---
    opti.subject_to(p[:, 0] == p_init)
    opti.subject_to(v[:, 0] == v_init)
    opti.subject_to(B[:, 0] == B_init)

    # Multiple Shooting
    for k in range(N):
        # simple basic drone model
        opti.subject_to(p[:, k+1] == p[:, k] + v[:, k] * dt)
        opti.subject_to(v[:, k+1] == v[:, k] + a[:, k] * dt)
        
        # battery evolution model (simple first try)
        c_batt = 0.01
        opti.subject_to(B[:, k+1] == B[:, k] - c_batt * ca.sumsqr(a[:, k]) * dt)

    # --- Physical bounds ---
    opti.subject_to(opti.bounded(-max_acc, a, max_acc))
    opti.subject_to(opti.bounded(-max_vel, v, max_vel))
    opti.subject_to(opti.bounded(0, B, 100)) # Battery can't go < 0%

    # --- Obstacles and Collisions avoidance ---
    for k in range(N+1):
        # Vector from the closest obstacle at step k to the nominal position at step k
        dp_bar = p_ego_prev[:, k] - p_obs_closest[:, k]
        
        dist_bar_sqr = ca.sumsqr(dp_bar)
        
        linear_term = 2 * ca.dot(dp_bar, (p[:, k] - p_ego_prev[:, k]))
        
        opti.subject_to(dist_bar_sqr + linear_term >= safe_rad**2)

    # --- Inizialization ---
    # Initial conditions
    opti.subject_to(p[:, 0] == [0, 0, 0])
    opti.subject_to(v[:, 0] == [0, 0, 0])
    opti.subject_to(B[:, 0] == 100)

    # solver choice
    # 'ipopt' for NLP;  'qrqp' or 'osqp' for QP 
    p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("osqp", p_opts, s_opts)

    # update closest obstacles and else
    # Assume 'tree' is your pre-built scipy.spatial.KDTree of all map obstacles

    # 1. Query the KDTree for all N+1 points along the predicted trajectory
    # KDTree expects shape (num_points, dimensions), so we transpose (.T)
    distances, indices = obs_tree.query(prev_solution_p.T)

    # 2. Extract the physical coordinates of those closest obstacles
    # Map the indices back to the original obstacle dataset
    # Resulting shape must match CasADi parameter: 3x(N+1)
    closest_obstacles_numeric = obs_tree.data[indices].T 

    # 3. Update CasADi Parameters
    opti.set_value(p_init, current_p)
    opti.set_value(flag, local_flags)
    opti.set_value(p_ego_prev, prev_solution_p)

    # INJECT THE KDTree RESULTS HERE:
    opti.set_value(p_obs_closest, closest_obstacles_numeric)

    try:
        # Run the optimization
        sol = opti.solve()
        
        # 5. Save the new trajectory for the next iteration's KDTree query and linearization
        prev_solution_p = sol.value(p) 
        # Retrieve the exact wall-clock time spent inside the solver
        solve_time = sol.stats()['t_wall_total']
        
        print(f"Optimal solution found in: {solve_time:.4f} seconds")
        
        # Apply optimal control
        a_opt = sol.value(a)

    except RuntimeError:
        print("Solver failed to find a solution!")
        # You can still inspect the stats even if it fails by querying the opti object directly
        fail_time = opti.debug.stats()['t_wall_total']
        print(f"Failed after: {fail_time:.4f} seconds")

