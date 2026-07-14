# Multi-UAV Trajectory Optimization — Distributed-MPC Control for Multi-UAV Map Coverage

A Python framework for simulating the coverage of a 3D environment with a swarm of UAVs (quadrotors) via **Distributed Model Predictive Control (DMPC)**, developed for the *Numerical Optimization for Control* course at Politecnico di Milano.

The main pipeline generates a 3D map populated with cylindrical obstacles, builds a set of coverage waypoints from the cameras' field of view, assigns tasks to the drones through Voronoi partitioning, and runs a swarm simulation in which each drone locally solves a linear-quadratic optimization problem (QP), formulated with **CasADi** and solved with **OSQP**.

## Table of contents

- [Project goal](#project-goal)
- [Repository structure](#repository-structure)
- [Pipeline / current code workflow](#pipeline--current-code-workflow)
- [Dynamic model](#dynamic-model)
- [DMPC formulation](#dmpc-formulation)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the simulation](#running-the-simulation)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Results](#results)
- [Notes and limitations](#notes-and-limitations)
- [Authors](#authors)
- [References](#references)

## Project goal

The goal is to simulate a coverage mission in which multiple UAVs explore a known map, avoid static obstacles and other agents, maintain a reference altitude, and maximize visual coverage while minimizing control effort and energy consumption. The codebase also implements an **early switching** variant, which allows a drone to transition to the next waypoint before fully reaching the current one, improving trajectory smoothness and cornering speed.

Main assumptions:
- UAVs with hovering capability (quadrotors, not fixed-wing);
- the map and obstacle dimensions are known a priori;
- position estimation is initially ideal, later subject to Gaussian noise (20–30 cm) in the robustness analysis;
- inter-agent communication is available across the map.

## Repository structure

```
.
├── main.py                          # main entry point, runs the full simulation
├── configs/
│   ├── demo_parameters.json         # map size, obstacles, number of drones, seeds, scenario params
│   └── optimization_params.json     # MPC horizon, timestep, cost weights, constraints
├── src/
│   ├── environment/                 # 3D map and obstacle generation
│   ├── partition/                   # Voronoi partitioning and waypoint assignment
│   ├── optimization/                # MPC formulation, swarm simulation, metrics
│   └── utils/                       # clustering, waypoint sanitization, plotting, drone helpers
└── logs/                            # CSV outputs produced by the simulations, analysis scripts
```

- **`main.py`** — main entry point. Runs the full simulation for one or more configured seeds.
- **`configs/demo_parameters.json`** — map dimensions, number of obstacles, number of drones, seeds, and scene parameters.
- **`configs/optimization_params.json`** — MPC prediction horizon, timestep, cost weights, velocity/acceleration constraints, and safety distance.
- **`src/environment/`** — generation of the 3D map and cylindrical obstacles.
- **`src/partition/`** — Voronoi partitioning and waypoint-to-drone assignment.
- **`src/optimization/`** — MPC formulation, swarm simulation loop, and performance metrics.
- **`src/utils/`** — clustering utilities, waypoint sanitization, plotting, and drone support functions.
- **`logs/`** — CSV files produced by simulation runs, plus analysis scripts.

## Pipeline / current code workflow

The workflow executed by `main.py` is:

1. Load parameters from the JSON configuration files.
2. Generate the 3D map and the drones' initial positions.
3. Compute the required number of waypoints from the area covered by the camera FOV and the desired overlap factor.
4. Generate waypoints via **K-means** clustering and sanitize them to avoid positions too close to obstacles.
5. Build the **Voronoi** partition to assign waypoints to drones.
6. Run the **distributed MPC** simulation for each drone, optionally comparing the standard configuration against the early-switching variant.
7. Save metrics to CSV files in `logs/`.

This corresponds to the two-stage architecture described in the accompanying report:

```
Offline pre-processing                          Online optimization
──────────────────────                          ───────────────────
1. Environment representation        →           4. UAV dynamic model
   (workspace + cylindrical obstacles)               (double-integrator)
2. Waypoint generation (K-means)      →           5. Distributed MPC controller
   + sanitization (obstacle clearance)                (local QP per agent, CasADi + OSQP)
3. Task allocation (Voronoi)          →           6. Swarm evolution
4. Trajectory sequencing                              (executed trajectories,
   (Nearest Neighbor + 2-opt / TSP)                    state feedback)
```

1. **Environment representation** — a 3D workspace `[xmin,xmax]×[ymin,ymax]×[zmin,zmax]` populated with `No` cylindrical obstacles, projected onto the horizontal plane to simplify the pre-processing stage.
2. **Waypoint generation & sanitization** — the required number of waypoints is estimated from the camera FOV footprint and an overlap factor `γ`; centroids are generated with **K-means** and then pushed radially out of the obstacles' safety zones.
3. **Task allocation (Voronoi tessellation)** — waypoints are partitioned among the drones via Voronoi tessellation (K-means with `K = Nu` to generate the seeds), balancing workload and limiting overlap between drone-specific areas.
4. **Trajectory sequencing (TSP + 2-opt)** — for each drone, the visiting order of its assigned waypoints is optimized as a Traveling Salesman Problem, solved heuristically with **Nearest Neighbor** followed by **2-opt** local search.

## Dynamic model

Each UAV is modeled as a point mass with double-integrator dynamics in 3D:

- state: `x_k = [p_k; v_k] ∈ R^6` (position and velocity)
- input: `u_k = a_k ∈ R^3` (acceleration)
- zero-order-hold discretization with timestep `Δt`:
  - `p_{k+1} = p_k + v_k·Δt + ½·a_k·Δt²`
  - `v_{k+1} = v_k + a_k·Δt`
- bounds on maximum velocity and acceleration: `‖a‖ ≤ a_max`, `‖v‖ ≤ v_max`

## DMPC formulation

At every control cycle, each drone solves a **local QP** over a prediction horizon of `N = 50` steps (`dt = 50 ms`), applying only the first computed input (receding horizon). The cost function is a weighted sum of:

| Term | Description |
|---|---|
| `J_wp` | tracking of the next waypoint (target-focus strategy, only on the immediately next waypoint) |
| `J_eff` | jerk penalty for smooth trajectories |
| `J_batt` | velocity penalty (energy efficiency, natural deceleration near the target) |
| `J_z` | reference altitude maintenance |
| `J_slack` | penalty on slack variables used for obstacle avoidance |

Obstacle avoidance (against static obstacles and other agents) is kept convex through **linearization via separating hyperplanes** (first-order Taylor expansion around the previously predicted trajectory), using a **dynamic safety radius** and quadratic slack variables to preserve recursive feasibility. Both a QP and an NLP formulation were tested; the QP formulation was retained since it gave equivalent solution quality with orders-of-magnitude faster solve times, thanks to convexity.

An **early switching** technique is also implemented: it anticipates the validation of a waypoint by expanding its acceptance radius, improving cornering speed and overall smoothness at the cost of a marginal (~0.15%) reduction in coverage.

## Requirements

- Python 3.9+
- numpy
- scipy
- shapely
- casadi
- matplotlib
- pandas
- osqp (used internally by CasADi/the QP solver)

## Installation

```bash
git clone https://github.com/veneziaandrea/multi-UAVs-trajectory-optimization.git
cd multi-UAVs-trajectory-optimization

python -m venv venv
# on Windows
venv\Scripts\activate
# on Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Running the simulation

```bash
python main.py
```

During execution, the script reads parameters from `configs/demo_parameters.json` and `configs/optimization_params.json`, runs a new simulation for each seed listed in the configuration, and saves the results to `logs/`.

## Configuration

### Scenario parameters (`configs/demo_parameters.json`)

- map dimensions
- number of obstacles
- obstacle radius and height ranges
- number of drones
- simulation seeds
- name of the produced log file

### Optimization parameters (`configs/optimization_params.json`)

- MPC prediction horizon
- discretization timestep
- cost weights for waypoint tracking, control effort, battery/energy, altitude, and slack
- velocity, acceleration, and safety distance constraints
- coverage overlap factor

Reference values used in the report's simulations:

| Parameter | Value | Description |
|---|---|---|
| Workspace size | 40 × 40 × 20 m | Simulated environment |
| Number of obstacles | 40 | Randomly generated cylinders |
| Obstacle radius | 0.5–1.5 m | |
| Obstacle height | 4–18 m | |
| Number of UAVs | 5 | Swarm size |
| Maximum velocity | 5 m/s | |
| Maximum acceleration | 2 m/s² | |
| Camera FOV | 84° | |
| Overlap factor | 0.3 | Coverage redundancy factor (optimal) |
| Safety distance | 0.5 m | Minimum obstacle clearance |
| Prediction horizon | 50 | MPC steps |
| Sampling time | 0.05 s | |

## Outputs

The simulation saves metrics to CSV files in `logs/`. The codebase also includes plotting utilities for the initial environment, the Voronoi partition, and the simulation results, although some plotting functions are optionally enabled or commented out in `main.py`.

## Results

- Coverage up to **99.26%** under ideal conditions (no position noise).
- With **early switching**: higher average speed and reduced mission time, at the cost of a negligible coverage loss (< 0.5%).
- **Robustness analysis** with Gaussian position noise (0.2–0.3 m per component, 20–25 maps tested): **zero collisions** observed, ~99% success rate, with recursive feasibility recovered via an emergency braking maneuver whenever the solver failed to find a solution.

Full details, plots, and comparison tables are available in the accompanying report.

## Notes and limitations

- This project is intended as a research/simulation environment, not as an installable library.
- The MPC formulation relies on a 3D double-integrator dynamic model with physical and safety constraints.
- The code was developed experimentally: some components may be extended or modified for deeper studies on robustness, energy efficiency, or real-world deployment.
- The map and obstacle dimensions are assumed **perfectly known a priori**; extending this to SLAM-based estimates would require modeling spatial uncertainty in the safety constraints.
- The kinematic (double-integrator) model could be extended to full rigid-body dynamics (Newton-Euler formalism), including a control allocation stage mapping accelerations to individual rotor commands, with a corresponding real-time feasibility analysis for embedded flight controllers.
- The current architecture acts as a mid-level trajectory planner over a static, known map; deployment in unstructured environments would require pairing it with a low-level reactive planner for dynamic, unmapped obstacles.

## Authors

Project developed for the *Numerical Optimization for Control* course — Politecnico di Milano:

- Andrea Venezia - [GitHub](https://github.com/veneziaandrea)  
- Francesco Street - [GitHub](https://github.com/francescostreet)  
- Francesco Urbano Sereno - [GitHub](https://github.com/FrancescoSereno)


## References

The method is described in detail in the accompanying report, *"Distributed-MPC Control for Multi-UAV Map Coverage."*
