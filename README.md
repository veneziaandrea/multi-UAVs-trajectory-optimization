# multi-UAV MPC based trajectory optimization
MPC based optimal control of a flying drones fleet for obstacle avoidance/surface monitoring 

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
│   └── generateDroneMap.m              % function to generate a 3D map with obstacles
├── mpc/                                % MPC scripts and functions
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


