# save generated map and occupancy grid
import pickle
from pathlib import Path

def save_map(workspace, occupancy_grid, obstacles, drone_starts, centroids, filename):

    # Create a folder if it doesn't exist
    save_dir = Path("data")
    save_dir.mkdir(parents=True, exist_ok=True)
    # Define the full path
    file_path = save_dir / f"{filename}.pkl"

    # Bundle them together
    data_to_save = {
        "map": workspace,
        "grid": occupancy_grid,
        "obstacles": obstacles,
        "drone_starts": drone_starts, #np.arrays
        "centroids": centroids #np.arrays
    }

    # Save as a single file
    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)