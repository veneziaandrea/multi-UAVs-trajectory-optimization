from config import load_config
import sys
from pathlib import Path
import casadi as ca
import numpy as np

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

# --- Optimization problem parameters ---
opti = ca.Opti()
num_regions = waypoints.size(axis = 0) # one region big as FOV per waypoint

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

# Flags and waypoints 
p_wp = opti.parameter(3, num_regions) 
flag = opti.parameter(num_regions)    

# --- Cost function computation ---
cost = 0

# cost to visit every waypoint
for i in range(num_regions):
    # Penalizziamo la distanza tra la fine dell'orizzonte predittivo (p[:, N]) e il waypoint
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
opti.subject_to(opti.bounded(0, B, 100)) # Battery can't go < 0

# Vincolo intero per il flag (Richiede un solutore MIQP/MINLP)
# Se usi un solutore continuo, rilassalo a: opti.subject_to(opti.bounded(0, flag, 1))

# --- 5. Vincoli di Ostacoli e Collisioni ---
# Esempio di ostacolo sferico (formulazione non-convessa standard)
# p_obs = np.array([5.0, 5.0, 5.0])
# r_obs = 1.5
# for k in range(N+1):
#     dist_sqr = ca.sumsqr(p[:, k] - p_obs)
#     opti.subject_to(dist_sqr >= r_obs**2) 
# NOTA: Per un QP rigoroso, qui andrebbero implementate delle Control Barrier Functions (CBF) lineari rispetto ad 'a'

# --- 6. Inizializzazione e Solutore ---
# Condizioni iniziali
opti.subject_to(p[:, 0] == [0, 0, 0])
opti.subject_to(v[:, 0] == [0, 0, 0])
opti.subject_to(B[:, 0] == 100)

# solver choice
# 'ipopt' for NLP;  'qrqp' for QP 
p_opts = {"expand": True}
s_opts = {"max_iter": 100}
opti.solver("ipopt", p_opts, s_opts) 

# SOLVE THE PROBLEM
sol = opti.solve()
p_opt = sol.value(p)

