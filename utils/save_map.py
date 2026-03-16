# save generated map and occupancy grid
import pickle 

def save_map(workspace, occupancy_grid):
    
    # Bundle them together
    data_to_save = {
        "map": workspace,
        "grid": occupancy_grid
    }

    # Save as a single file
    with open("navigation_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)