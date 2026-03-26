import numpy as np
import sys
from pathlib import Path


# Set the root directory and add the source directory to the Python path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, seed_everything
from environment.map_generation_v2 import Map3D
from utils.plot_initial_envronment import plot_initial_environment

def build_demo(config): 
    # Load environment configuration
    map_cfg = config["map"]
    uav_cfg = config["uav"]
    seed = config["seed"]

    # Generate the 3D map and get drone starting positions
    map3d, drone_positions = Map3D.generate_map3D(
        x_bounds=map_cfg["x_bounds"],
        y_bounds=map_cfg["y_bounds"],
        z_bounds=map_cfg["z_bounds"],
        num_obstacles=map_cfg["num_obstacles"],
        obstacle_radius_range=map_cfg["obstacle_radius_range"],
        obstacle_height_range=map_cfg["obstacle_height_range"],
        num_drones=uav_cfg["num_uavs"],                 
        spacing=uav_cfg["start_separation"],                       
        seed=seed,
    )
    print(f"Generated map with {len(map3d.obstacles)} obstacles and {len(drone_positions)} drone starting positions.")
    print("Drone starting positions:")
    for i, pos in enumerate(drone_positions):
        print(f"  Drone {i+1}: {pos}")

    return map3d, drone_positions


if __name__ == "__main__":
    # Load configuration
    config_path = ROOT / "configs" / "demo_parameters.json"
    config = load_config(config_path)

    # Set random seed for reproducibility
    seed_everything(config["seed"])

    # Build the demo environment and get initial drone positions
    Map3D, drone_positions = build_demo(config)
    print("Map bounds:", Map3D.x_bounds, Map3D.y_bounds, Map3D.z_bounds)

    # Visualize the generated map and drone starting positions
    plot_initial_environment(Map3D, drone_positions)









