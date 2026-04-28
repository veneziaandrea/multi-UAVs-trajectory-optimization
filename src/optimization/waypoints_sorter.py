import numpy as np
from scipy.spatial.distance import cdist

def sort_waypoints_tsp(start_pos, waypoints):
    if len(waypoints) <= 1: return waypoints
    
    # 1. Greedy Initial Path 
    ordered = []
    current_pos = start_pos[:2]
    remaining = waypoints.tolist()
    while remaining:
        dists = cdist([current_pos], [p[:2] for p in remaining])[0]
        next_idx = np.argmin(dists)
        current_pos = remaining[next_idx][:2]
        ordered.append(remaining.pop(next_idx))
    
    path = np.array(ordered)
    
    # 2. 2-Opt Refinement (migliora l'ottimalità)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue
                # Calcola il guadagno di distanza scambiando i segmenti
                old_dist = np.linalg.norm(path[i-1] - path[i]) + np.linalg.norm(path[j-1] - path[j])
                new_dist = np.linalg.norm(path[i-1] - path[j-1]) + np.linalg.norm(path[i] - path[j])
                
                if new_dist < old_dist:
                    path[i:j] = path[i:j][::-1] # Inverte il segmento
                    improved = True
    return path