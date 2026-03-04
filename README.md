# multi-UAV MPC based trajectory optimization
MPC based optimal control of a flying drones fleet for obstacle avoidance/surface monitoring 

--- 

# Pipeline
1. Modellization
2. Map generation
3. Voronoi Diagrams:
   3.1 fnc to compute map partitions
   3.2 fcn to discretize each assigned map partion to generate waypoints - DWA x each zone
   3.3 fcn to assign drone traj - cost fcn + iteration
4. MPC
5. Visualization/Simulation


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


