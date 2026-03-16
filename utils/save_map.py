# save generated map and occupancy grid
import pickle
from pathlib import Path



def save_map(workspace, occupancy_grid, filename):

    # Create a 'maps' folder if it doesn't exist
    save_dir = Path("../data")
    save_dir.mkdir(parents=True, exist_ok=True)
    # Define the full path
    file_path = save_dir / f"{filename}.pkl"

    # Bundle them together
    data_to_save = {
        "map": workspace,
        "grid": occupancy_grid
    }

    # Save as a single file
    with open("navigation_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)