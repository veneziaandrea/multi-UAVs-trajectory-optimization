import numpy as np
import scipy as sci
#import casadi as ca
import matplotlib.pyplot as plt

'''
USE EXAMPLE OF CASADI 

# 1. Creiamo l'oggetto Opti
opti = ca.Opti()

# 2. Dichiariamo le variabili di ottimizzazione
x = opti.variable()
y = opti.variable()

# 3. Definiamo la Funzione Obiettivo (da minimizzare)
obj = (x - 1)**2 + (y - 2.5)**2
opti.minimize(obj)

# 4. Aggiungiamo i Vincoli (Constraints)
opti.subject_to(x + y >= 0)
opti.subject_to(opti.bounded(-10, x, 10)) # Vincolo box su x

# 5. Scegliamo il Solver (IPOPT è lo standard per non-lineare)
p_opts = {"expand": True} # Espande il grafo per velocità
s_opts = {"max_iter": 100}
opti.solver('ipopt', p_opts, s_opts)

# 6. Risolviamo
try:
    sol = opti.solve()
    print(f"Soluzione ottima: x={sol.value(x):.2f}, y={sol.value(y):.2f}")
except:
    print("Il solver ha fallito o raggiunto il limite.")
    # Possiamo comunque recuperare i valori correnti con debug
    print(f"Valori debug: x={opti.debug.value(x)}")
    '''

def drone_model(pos, vel, acc, dt):
<<<<<<< HEAD
    pos_nxt= pos.reshape(3,1) + vel.reshape(3,1)*dt + 0.5*acc*(dt**2)
    vel_nxt= vel.reshape(3,1) + acc*dt
=======
    # Enforces strict (3, 1) column vectors using reshape for flawless, zero-cost broadcasting
    pos_nxt = pos.reshape(3, 1) + vel.reshape(3, 1) * dt + 0.5 * acc * (dt**2)
    vel_nxt= vel.reshape(3, 1) + acc*dt
>>>>>>> 7cca7d31dbcb5bbbd18615dfbf27213639672999
    acc_nxt= acc 

    return pos_nxt, vel_nxt

def sample_acc(a_previous, a_dw_min, a_dw_max, N_tot=300, ratio_warm=0.7, sigma=0.3):

    """
    Generate the acceleration samples matrix(3, N_tot) 
    
    a_prev: array (3, 1), acceleration chosen the previous iteration
    a_dw_min: array (3, 1), minimum acceleration
    a_dw_max: array (3, 1), maximum acceleration
    N_tot: int, total number of trajectories to be evaluated
    ratio_warm: float, percentage of samples generated around the warm start accelerations (0.0 - 1.0)
    sigma: float, standard deviation of possible generated samples around a_prev
    """
    
    # Calcolo del numero di campioni per ciascuna strategia
    N_warm = int(N_tot * ratio_warm)
    N_explore = N_tot - N_warm
    
    # 1. WARM START (Sfruttamento locale tramite Gaussiana)
    # Genera N_warm campioni per le 3 componenti (x, y, z) centrati su a_previous.
    # FIXED: Added [:, np.newaxis] to reshape a_previous from (3,) to (3, 1) for broadcasting
    a_sample_warm = np.random.normal(loc=a_previous[:, np.newaxis], scale=sigma, size=(3, N_warm))
    
    # Una distribuzione normale ha code infinite, quindi alcuni campioni potrebbero 
    # finire fuori dalla finestra dinamica ammissibile. Li "tagliamo" ai limiti fisici.
    a_sample_warm = np.clip(a_sample_warm, a_dw_min, a_dw_max)

    # Campionamento Uniforme Globale
    # Genera numeri pseudo-casuali uniformi tra 0 e 1, dimensione (3, N_explore)
    random_base = np.random.rand(3, N_explore)
    
    # Scala questi numeri per coprire esattamente il volume tra a_dw_min e a_dw_max
    a_sample_explore = a_dw_min + random_base * (a_dw_max - a_dw_min)
    
    # Final matrix a_sample (3, N_tot)
    a_sample = np.hstack((a_sample_warm, a_sample_explore))
    
    return a_sample
    
def compute_obstacles_cost (p_i_final, kd_tree, safety_radius, N_tot, obs_radii):
    """
    p_i_final: array (3, N) dei waypoint finali campionati
    kd_tree: obstacles cartesian points converted to scipy.spatial.KDTree
    raggio_sicurezza: float, minimum tolered distance
    N_tot: int, number of acceleration samples
    """
    # FIXED: Removed slicing to query all 3 dimensions (p_i_final.T)
    # FIXED: Captured nearest_idx to look up the corresponding radius
    min_dist, nearest_idx = kd_tree.query(p_i_final.T)
    
    # FIXED: Fetch the specific radius for the closest obstacle
    nearest_radii = obs_radii[nearest_idx]
    
    # FIXED: Calculate the true distance to the obstacle surface
    true_dist = min_dist - nearest_radii
    
    C_obstacle = np.zeros(N_tot)
    in_collision = true_dist <= safety_radius
    safe = true_dist > safety_radius
    C_obstacle[in_collision] = 1e6
    C_obstacle[safe] = 1.0 / (true_dist[safe] - safety_radius + 1e-6)

    return C_obstacle

class Drone:

    acc_lim= 3 # m/s^2
    vel_lim= 10 # m/s
    lim = [vel_lim, acc_lim]

    def __init__ (self, i, position, lim):
        self.id = f"d_{i}"
        self.pos= position
        self.speed= np.zeros(3)
        self.lim= lim
        self.a_prev= np.zeros(3)

    dt= 0.01 # Ts [s] (100 Hz)
    N= 50 # timesteps
    T_h= N*dt
    w1= 0.7 # weight for distance cost function
    w2= 1-w1 # weight for obstacles cost function
    safe_rad= 0.3 # safe distance from obstacles [m]

    # Assuming drone class object with pos, vel, acc variables
    def DWA(self, pos_i, vel_i, ref_j, a_prev, acc_lim, T_h, w1, w2, obs_tree, safe_rad, obs_radii):
<<<<<<< HEAD
=======

>>>>>>> 7cca7d31dbcb5bbbd18615dfbf27213639672999
        N_tot= 300
        a_vec= sample_acc(a_prev, -acc_lim, acc_lim, N_tot, ratio_warm=0.75, sigma=0.3)
        p_fin= drone_model(pos_i, vel_i, a_vec, T_h)

<<<<<<< HEAD
        # FIXED: Ensure p_fin is an array (and grab the first element if drone_model returned a tuple)
=======
        # Ensure p_fin is an array (and grab the first element if drone_model returned a tuple)
>>>>>>> 7cca7d31dbcb5bbbd18615dfbf27213639672999
        if isinstance(p_fin, tuple):
            p_fin = p_fin[0]
        p_fin = np.array(p_fin)

        # FIXED: Convert ref_j from a list to a numpy array, and transpose it to align the coordinates (3, N_targets)
        ref_j = np.array(ref_j).T

        dist= p_fin[:,:,np.newaxis] - ref_j[:, np.newaxis, :]
        sq_dist= np.sum(dist**2, axis=0)
        C_dist= np.sum(1.0 / (sq_dist+ 1e-6), axis=1)
<<<<<<< HEAD
        dist= p_fin[:,:,np.newaxis] - ref_j[:, np.newaxis, :]
        sq_dist= np.sum(dist**2, axis=0)
        C_dist= np.sum(1.0 / (sq_dist+ 1e-6), axis=1)
=======
>>>>>>> 7cca7d31dbcb5bbbd18615dfbf27213639672999
        C_obs= compute_obstacles_cost(p_fin, obs_tree, safe_rad, N_tot, obs_radii)

        J= w1*C_dist + w2*C_obs

<<<<<<< HEAD
        # Identify any trajectory that goes under the map (Z coordinate < 0)
        underground_mask = p_fin[2, :] < 0 
        higher_limit_mask= p_fin[2,:] > 10
        size_x_map_mask= p_fin[0, :] > 50
        size_y_map_mask= p_fin[1,:] > 50

        invalid_point_mask = underground_mask | higher_limit_mask | size_x_map_mask | size_y_map_mask
        
        # Apply a massive cost penalty to those specific trajectories
        J[invalid_point_mask] += 1e6
=======
        # 1. Floor & Ceiling Penalty (Z-axis)
        z_too_low = p_fin[2, :] < 0
        z_too_high = p_fin[2, :] > 10.0  # Match your map height
        
        # 2. Map Boundary Penalty (X and Y axis)
        # Assuming map is 0 to 50 (size of the map), adjust if different
        out_of_bounds_x = (p_fin[0, :] < 0) | (p_fin[0, :] > 50)
        out_of_bounds_y = (p_fin[1, :] < 0) | (p_fin[1, :] > 50)

        # Apply massive cost to any trajectory that leaves the "safety box"
        invalid_mask = z_too_low | z_too_high | out_of_bounds_x | out_of_bounds_y
        J[invalid_mask] += 1e6

        best_idx = np.argmin(J)
        return p_fin[:, best_idx], a_vec[:, best_idx], J[best_idx], best_idx
>>>>>>> 7cca7d31dbcb5bbbd18615dfbf27213639672999

        best_idx= np.argmin(J)

        return p_fin[:, best_idx], a_vec[:, best_idx], J[best_idx], best_idx
    
def plot_final_trajectories(trajectory_history, obstacles, drone_ids):
    """
    Plots the full path for all drones and draws 3D cylinders for obstacles.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    
    # 1. DRAW REAL CYLINDRICAL OBSTACLES
    # We iterate over the original list of objects to access .height and .radius
    for obs in obstacles:
        # Create cylinder mesh
        z_range = np.linspace(0, obs.height, 10)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z_range)
        
        # Parametric equations for a cylinder
        x_grid = obs.radius * np.cos(theta_grid) + obs.x
        y_grid = obs.radius * np.sin(theta_grid) + obs.y
        
        # Plot the surface
        ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.3, rstride=1, cstride=1)

    # 2. PLOT DRONE PATHS
    colors = plt.cm.get_cmap('tab10', len(drone_ids))
    for k in range(len(drone_ids)):
        path = np.array(trajectory_history[k])
        if path.size == 0: continue 
        
        colore_drone = colors(k)
        # Plot the continuous trajectory line
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                color=colore_drone, linewidth=2, label=f"Drone {drone_ids[k]}")
        
        # Start (Square) and Finish (Star)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=colore_drone, marker='s', s=100)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=colore_drone, marker='*', s=200)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Final Multi-UAV 3D Trajectories")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show(block=True)
    
def plot_dwa_results(pos_i, waypoints, mappa_ostacoli, drone_ids):
    """
    Visualizza in 3D la posizione attuale, gli ostacoli e il waypoint scelto per ogni drone.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Multi-UAV Trajectory - Current Step")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    
    # Plot degli ostacoli
    ax.scatter(mappa_ostacoli[:, 0], mappa_ostacoli[:, 1], mappa_ostacoli[:, 2], 
               c='gray', alpha=0.1, s=10, label="Ostacoli")

    colors = plt.cm.get_cmap('tab10', len(drone_ids))

    for k in range(len(drone_ids)):
        colore_drone = colors(k)
        
        # Posizione attuale (quadratino)
        curr_p = pos_i[k].ravel()
        ax.scatter(curr_p[0], curr_p[1], curr_p[2], 
                   color=colore_drone, marker='s', s=80)
        
        # Waypoint scelto (stella)
        next_p = waypoints[k].ravel()
        ax.scatter(next_p[0], next_p[1], next_p[2], 
                   color=colore_drone, marker='*', s=150, label=f"Drone {drone_ids[k]}")

        # Linea di movimento prevista per questo step
        ax.plot([curr_p[0], next_p[0]], 
                [curr_p[1], next_p[1]], 
                [curr_p[2], next_p[2]], 
                color=colore_drone, linestyle='-', linewidth=2)

    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc='upper right')
    plt.show(block=False)
    plt.pause(0.01) # Allows the plot to update live if in a loop

def plot_final_trajectories(trajectory_history, obstacles, drone_ids):
    """
    Plots the full path for all drones and draws 3D cylinders for obstacles.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    
    # 1. DRAW REAL CYLINDRICAL OBSTACLES
    # We iterate over the original list of objects to access .height and .radius
    for obs in obstacles:
        # Create cylinder mesh
        z_range = np.linspace(0, obs.height, 10)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z_range)
        
        # Parametric equations for a cylinder
        x_grid = obs.radius * np.cos(theta_grid) + obs.x
        y_grid = obs.radius * np.sin(theta_grid) + obs.y
        
        # Plot the surface
        ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.3, rstride=1, cstride=1)

    # 2. PLOT DRONE PATHS
    colors = plt.cm.get_cmap('tab10', len(drone_ids))
    for k in range(len(drone_ids)):
        path = np.array(trajectory_history[k])
        if path.size == 0: continue 
        
        colore_drone = colors(k)
        # Plot the continuous trajectory line
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                color=colore_drone, linewidth=2, label=f"Drone {drone_ids[k]}")
        
        # Start (Square) and Finish (Star)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=colore_drone, marker='s', s=100)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=colore_drone, marker='*', s=200)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Final Multi-UAV 3D Trajectories")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show(block=True)





        




