# Multi-UAV Trajectory Optimization

Python project for multi-UAV trajectory optimization in cluttered 3D environments.
The repository combines map generation, waypoint sampling, Voronoi-based area partitioning, and Model Predictive Control (MPC) to coordinate a fleet of drones for coverage and obstacle avoidance tasks.

## Overview

The current pipeline implemented in the repository is:

1. Random 3D environment generation with cylindrical obstacles
2. Multi-UAV initial placement
3. Waypoint generation over the free space
4. Waypoint sanitization with obstacle safety margins
5. Voronoi partition of the workspace
6. Waypoint assignment to each drone
7. Waypoint ordering
8. Distributed MPC-based trajectory optimization
9. Visualization of trajectories, costs, and simulation playback

## Main Features

- Procedural 3D map generation with configurable bounds and obstacle sets
- Free-space waypoint generation through clustering
- Voronoi partitioning for decentralized workspace assignment
- Per-drone waypoint ordering before optimization
- MPC formulation for trajectory tracking, collision avoidance, and motion regularization
- Plotting utilities for environment setup, partitioning, kinematics, and simulation animation

## Repository Structure

```text
multi-UAVs-trajectory-optimization/
|-- main.py
|-- requirements.txt
|-- configs/
|   |-- demo_parameters.json
|   `-- optimization_params.json
`-- src/
    |-- config.py
    |-- environment/
    |   |-- map_generation.py
    |   `-- map_generation_v2.py
    |-- optimization/
    |   |-- mpc.py
    |   |-- optimization_plots.py
    |   `-- waypoints_sorter.py
    |-- partition/
    |   `-- voronoi.py
    `-- utils/
        |-- drones.py
        |-- geometry.py
        |-- kmeans.py
        |-- plot_initial_envronment.py
        `-- plot_voronoi.py
```

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `shapely`
- `matplotlib`
- `casadi`

## Installation

Clone the repository:

```bash
git clone https://github.com/veneziaandrea/multi-UAVs-trajectory-optimization.git
cd multi-UAVs-trajectory-optimization
```

Create and activate a virtual environment if you want an isolated setup:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install numpy scipy shapely matplotlib casadi
```

## Configuration

The simulation is driven by two JSON files in [`configs/`](./configs):

- `demo_parameters.json`: workspace size, obstacle generation, number of UAVs, altitude, and initial separation
- `optimization_params.json`: MPC horizon, timestep, iteration budget, safety distance, and cost weights

Default execution uses:

- `configs/demo_parameters.json` for environment generation
- `configs/optimization_params.json` for MPC settings

## Usage

Run the main simulation with:

```bash
python main.py
```

The script will:

- generate a random 3D map
- compute candidate waypoints in the free space
- build a Voronoi partition for the fleet
- assign and sort waypoints for each drone
- run the MPC loop
- open plots for partitioning, cost evolution, trajectories, kinematics, and animation

## Notes

- The project is currently research-oriented and still evolving.
- Some legacy files from earlier experiments are still present in the repository.
- The simulation is visualization-heavy, so running it in an environment with GUI support is recommended.

## Authors

- Andrea Venezia - [GitHub](https://github.com/veneziaandrea)
- Francesco Street - [GitHub](https://github.com/francescostreet)
- Francesco Urbano Sereno - [GitHub](https://github.com/FrancescoSereno)

## Contributions

Issues and pull requests are welcome.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
