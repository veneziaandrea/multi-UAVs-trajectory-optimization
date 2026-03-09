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
        u_list= []
        for i in range(0,k):
            
            ...





