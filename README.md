# multi-UAV MPC based trajectory optimization
MPC based optimal control of a flying drones fleet for obstacle avoidance/surface monitoring 

--- 

# Pipeline
1. MAP GENERATION and GRID QUANTIZATION
2. INITIAL DRONE POSITIONS
3. VORONOI PARTITION (WIP)
4. LLOYD RELAXATION (init only)
5. STATIC CELL ASSIGNMENT
6. LOCAL WAYPOINT GENERATION
7. LOCAL PATH PLANNING
8. DISTRIBUTED MPC
9. (PYBULLET SIMULATION)

---

## Authors

- Andrea Venezia - [GitHub](https://github.com/veneziaandrea)  
- Francesco Street - [GitHub](https://github.com/francescostreet)  
- Francesco Urbano Sereno - [GitHub](https://github.com/FrancescoSereno)

---

## Structure
```
multi-UAVs-trajectory-optimization/
│
├── main.m                              % main script to run the simulation
├── sim_params.m                        % simulation parameters (Ts, limits, horizon, etc.)
├── drone_model.m                       % discrete-time 3D drone model
│
├── utils/                              % helper functions (plotting, obstacle generation)
│   ├── DWA.m                           % Dynamic Window Approach function
│   ├── FindIntesections.m              % function to minimize drones collision
│   └── generateDroneMap.m              % function to generate a 3D map with obstacles
│
├── mpc/                                % MPC scripts and functions
│   ├── pred_mats.m                     % function to generate prediction matrices for QP
│   ├── QP_mats.m                       % function to generate QP problem
│   └── MPC.m                           % MPC
│
├── results/                            % saved logs and figures
├── data/                               % static data (waypoints, obstacles, maps)
├── docs/                               % documentation and diagrams
│
├── .gitignore
└── README.md
```
--- 

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/veneziaandrea/multi-UAVs-trajectory-optimization.git
2. Run main.m

--- 

## Requirements
    - MATLAB R2020a or later

---

## Contributions

Pull requests and issues are welcome.

---

## License

MIT License


