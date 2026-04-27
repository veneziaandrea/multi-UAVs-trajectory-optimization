import numpy as np
from scipy.spatial.distance import cdist

def sort_waypoints_tsp(start_pos, waypoints):
    """Ordina i waypoint in una sequenza efficiente (Greedy TSP)."""
    if len(waypoints) <= 1: return waypoints
    
    ordered = []
    current_pos = start_pos[:2]
    remaining = waypoints.tolist()
    
    while remaining:
        # Trova il punto più vicino alla posizione attuale
        dists = cdist([current_pos], [p[:2] for p in remaining])[0]
        next_idx = np.argmin(dists)
        current_pos = remaining[next_idx][:2]
        ordered.append(remaining.pop(next_idx))
        
    return np.array(ordered)