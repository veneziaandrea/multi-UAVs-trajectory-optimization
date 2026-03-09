import numpy as np
import scipy as sci
import casadi as ca

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

class Drone:

    acc_lim= 0.5 # m/s^2
    vel_lim= 4 # m/s
    lim = []
    lim= list.append(vel_lim, acc_lim)

    def __init__ (self, i, position, speed, lim):
        self.id = "d_{i}"
        self.pos= position
        self.speed= speed
        self.lim= lim

    # Assuming drone class object with pos, vel, acc variables + limitations as list
    def DWA(self):

        # unpack drone's state
        pos= self.pos
        vel= self.vel
        acc= self.acc
        v_lim= self.lim[1]
        a_lim= self.lim[2]

        # choose k according to how many trajectories you want to generate
        k= 10
        # generate trajectories and save final positions and used control inputs
        pos_list= []
        dt= 0.1 # simulation time [s]
        a_list= np.linspace(-a_lim, a_lim, k)
        for a_k in a_list:
            state_k= drone_model(pos, vel, a_k, dt)
            pos_k= state_k[1]
            vel_k= state_k[2]

            if vel_k <= v_lim:
                pos_list.append(pos_k)
            else:
                a_list.replace(a_k, NULL)
        




