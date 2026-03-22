import numpy as np
import scipy as sci
import casadi as ca
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
    pos_nxt= pos + vel*dt + 0.5*acc*(dt**2)
    vel_nxt= vel + acc*dt
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
    a_sample_warm = np.random.normal(loc=a_previous, scale=sigma, size=(3, N_warm))
    
    # Una distribuzione normale ha code infinite, quindi alcuni campioni potrebbero 
    # finire fuori dalla finestra dinamica ammissibile. Li "tagliamo" ai limiti fisici.
    a_sample_warm = np.clip(a_sample_warm, a_dw_min, a_dw_max)

    #Campionamento Uniforme Globale
    # Genera numeri pseudo-casuali uniformi tra 0 e 1, dimensione (3, N_explore)
    random_base = np.random.rand(3, N_explore)
    
    # Scala questi numeri per coprire esattamente il volume tra a_dw_min e a_dw_max
    a_sample_explore = a_dw_min + random_base * (a_dw_max - a_dw_min)
    
    # Final matrix a_sample (3, N_tot)
    a_sample = np.hstack((a_sample_warm, a_sample_explore))
    
    return a_sample

def compute_obstacles_cost (p_i_final, kd_tree, safety_radius, N_tot):
    """
    p_i_final: array (3, N) dei waypoint finali campionati
    kd_tree: obstacles cartesian points converted to scipy.spatial.KDTree
    raggio_sicurezza: float, minimum tolered distance
    N_tot: int, number of acceleration samples
    """
    min_dist, _ = kd_tree.query(p_i_final.T)
    C_obstacle = np.zeros(N_tot)
    in_collision = min_dist <= safety_radius
    safe = min_dist > safety_radius
    C_obstacle[in_collision] = 1e6
    C_obstacle[safe] = 1.0 / (min_dist[safe] - safety_radius + 1e-6)

    return C_obstacle

class Drone:

    acc_lim= 3 # m/s^2
    vel_lim= 10 # m/s
    lim = []
    lim= list.append(vel_lim, acc_lim)

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
    def DWA(pos_i, vel_i, ref_j, a_prev, acc_lim, T_h, w1, w2, obs_tree, safe_rad):
        N_tot= 300
        a_vec= sample_acc(a_prev, -acc_lim, acc_lim, N_tot, ratio_warm=0.75, sigma=0.3)
        p_fin= drone_model(pos_i, vel_i, a_vec, T_h)
        dist= p_fin[:,:,np.newaxis] - ref_j[:, np.newaxis, :]
        sq_dist= np.sum(dist**2, axis=0)
        C_dist= np.sum(1.0 / (sq_dist+ 1e-6), axis=1)
        C_obs= compute_obstacles_cost(p_fin, obs_tree, safe_rad, N_tot)

        J= w1*C_dist + w2*C_obs

        best_idx= np.argmin(J)

        return pos_i[best_idx], a_vec[best_idx], J[best_idx], best_idx
    
def plot_dwa_results(p_i_t, p_i_final, J_min, J, mappa_ostacoli, delta_J_max=1.0):
    """
    Visualizza in 3D la posizione attuale, gli ostacoli e il fascio di waypoint futuri
    che soddisfano il criterio di costo.
    
    p_i_t: (3, 1) posizione attuale del drone
    p_i_final: (3, N) tutti i waypoint finali campionati
    J: (N,) l'array dei costi associati a ogni waypoint
    mappa_ostacoli: (M, 3) la nuvola di punti della mappa
    delta_J_max: float, la soglia massima di tolleranza rispetto al costo minimo
    """
    # 1. Setup della figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("DWA 3D - Valutazione Waypoint")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    
    # 2. Plot degli ostacoli (Mappa Nota)
    # Usiamo un'opacità (alpha) bassa e un colore neutro per non coprire le traiettorie
    ax.scatter(mappa_ostacoli[:, 0], mappa_ostacoli[:, 1], mappa_ostacoli[:, 2], 
               c='gray', alpha=0.1, s=10, label="Ostacoli")

    
    # Creiamo una maschera booleana: True per i sample che rientrano nel delta tollerato
    maschera_accettabili = J <= (J_min + delta_J_max)
    
    # Estraiamo solo i waypoint che passano il filtro
    p_i_final_accettabili = p_i_final[:, maschera_accettabili]
    J_accettabili = J[maschera_accettabili]

    # 4. Plot del fascio di waypoint accettabili (Colorati in base al costo)
    # Usiamo una colormap per far vedere la sfumatura da "molto buono" a "limite di tolleranza"
    scatter_bundle = ax.scatter(p_i_final_accettabili[0, :], 
                                p_i_final_accettabili[1, :], 
                                p_i_final_accettabili[2, :], 
                                c=J_accettabili, cmap='viridis', s=20, alpha=0.6, 
                                label=f"Campioni (Delta J < {delta_J_max})")
    plt.colorbar(scatter_bundle, ax=ax, label="Valore Funzione di Costo J")

    # 5. Plot della posizione attuale e del Waypoint Ottimo Scelto
    # p_i_t è (3, 1), lo "appiattiamo" con ravel() per matplotlib
    pos_attuale = p_i_t.ravel()
    ax.scatter(pos_attuale[0], pos_attuale[1], pos_attuale[2], 
               c='black', marker='s', s=100, label="Posizione Attuale (p_i_t)")
    
    best_waypoint = p_i_final[:, best_idx]
    ax.scatter(best_waypoint[0], best_waypoint[1], best_waypoint[2], 
               c='red', marker='*', s=200, label="Waypoint Scelto (Ottimo)")

    # 6. (Opzionale) Disegna una linea tra la posizione attuale e l'ottimo
    ax.plot([pos_attuale[0], best_waypoint[0]], 
            [pos_attuale[1], best_waypoint[1]], 
            [pos_attuale[2], best_waypoint[2]], 
            color='red', linestyle='--', linewidth=2)

    # Imposta un'inquadratura isometrica proporzionata (evita distorsioni visive)
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()





        




