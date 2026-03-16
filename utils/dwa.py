import numpy as np
import scipy as sci
from ... import casadi as ca
import random

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
    res= list.append(pos_nxt, vel_nxt)
    return res

def sample_acc(distribution, a_prev, num_samples, variance):

    if distribution == "Normal":
        a_norm= np.random(a_prev, variance, num_samples)
    elif distribution == "Uniform":
        a_uni= np.random(..., ..., num_samples - 150) #spero sia una cosa simile; hp num samples tipo 200
    
    a_vec= np.cat(a_norm, a_uni)

    return a_vec

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

    dt= 0.01 # Ts [s] (100 Hz)
    N= 50 # timesteps
    T_h= N*dt
    # Assuming drone class object with pos, vel, acc variables + limitations as list
    def DWA(pos_i, vel_i, ref_j, acc_i, T_h, w1, w2):

        for _ in range(3):
            distrib= "Normal"
            a_norm= sample_acc(distrib, a_prev, N_norm)
            distrib= "Uniform"
            a_uni= sample_acc(distrib, a_prev, N_uni)
        
        a_vec= np.cat(a_norm, a_uni)

        p_fin= drone_model(pos_i, vel_i, a_vec, T_h)
        dist= p_fin[:,:,np.newaxis] - ref_j[:, np.newaxis, :]
        sq_dist= np.sum(dist**2, axis=0)
        C_dis= np.sum(1.0 / (sq_dist+ 1e-6), axis=1)
        
        C_obs= ...


        # choose k according to how many trajectories you want to generate
        k= 10
        # generate trajectories and save final positions and used control inputs
        pos_vec= np.array()
        vel_vec= np.array ()
        acc_vec= np.array()



        




