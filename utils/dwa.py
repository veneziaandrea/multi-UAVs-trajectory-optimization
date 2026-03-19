import numpy as np
import scipy as sci
import casadi as ca
from scipy.spatial import KDTree

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

def sample_acc(a_previous, a_dw_min, a_dw_max, N_tot=300, ratio_warm=0.8, sigma=0.3):

    """
    Genera la matrice dei campioni di accelerazione (3, N_tot) usando una strategia ibrida.
    
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
    
    # ---------------------------------------------------------
    # 1. WARM START (Sfruttamento locale tramite Gaussiana)
    # ---------------------------------------------------------
    # Genera N_warm campioni per le 3 componenti (x, y, z) centrati su a_previous.
    # La dimensione risultante è (3, N_warm).
    a_sample_warm = np.random.normal(loc=a_previous, scale=sigma, size=(3, N_warm))
    
    # Una distribuzione normale ha code infinite, quindi alcuni campioni potrebbero 
    # finire fuori dalla finestra dinamica ammissibile. Li "tagliamo" ai limiti fisici.
    a_sample_warm = np.clip(a_sample_warm, a_dw_min, a_dw_max)
    
    # ---------------------------------------------------------
    # 2. ESPLORAZIONE (Campionamento Uniforme Globale)
    # ---------------------------------------------------------
    # Genera numeri pseudo-casuali uniformi tra 0 e 1, dimensione (3, N_explore)
    random_base = np.random.rand(3, N_explore)
    
    # Scala questi numeri per coprire esattamente il volume tra a_dw_min e a_dw_max
    a_sample_explore = a_dw_min + random_base * (a_dw_max - a_dw_min)
    
    # ---------------------------------------------------------
    # 3. UNIONE DEI SET
    # ---------------------------------------------------------
    # Affianca orizzontalmente le due matrici.
    # Il risultato è la matrice finale a_sample di dimensione (3, N_tot)
    a_sample = np.hstack((a_sample_warm, a_sample_explore))
    
    return a_sample

class Drone:

    acc_lim= 3 # m/s^2
    vel_lim= 10 # m/s
    lim = []
    lim= list.append(vel_lim, acc_lim)

    def __init__ (self, i, position, speed, lim):
        self.id = "d_{i}"
        self.pos= position
        self.speed= speed
        self.lim= lim
        self.a_prev= np.zeros(3)

    dt= 0.01 # Ts [s] (100 Hz)
    N= 50 # timesteps
    T_h= N*dt
    w1= int(0.7) # weight for distance cost function
    w2= 1-w1 # weight for obstacles cost function

    # Assuming drone class object with pos, vel, acc variables
    def DWA(pos_i, vel_i, ref_j, a_prev, acc_lim, T_h, w1, w2):

        a_vec= sample_acc(a_prev, -acc_lim, acc_lim, N_tot=300, ratio_warm=0.75, sigma=0.3)
        p_fin= drone_model(pos_i, vel_i, a_vec, T_h)
        dist= p_fin[:,:,np.newaxis] - ref_j[:, np.newaxis, :]
        sq_dist= np.sum(dist**2, axis=0)
        C_dist= np.sum(1.0 / (sq_dist+ 1e-6), axis=1)
        
        C_obs= ...

        J= w1*C_dist + w2*C_obs

        best_idx= np.argmin(J)

        return best_idx





        




