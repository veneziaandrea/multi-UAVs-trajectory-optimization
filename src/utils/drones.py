import numpy as np

class Drone:
    def __init__(self, drone_id, start_pos, waypoints, mpc_vars, horizon_n):
        self.id = drone_id

        self.mpc_vars = mpc_vars

        self.waypoints = waypoints
        # Convert input to numpy array
        p_pos = np.array(start_pos, dtype=float).flatten()

        # If the input is [x, y], add a default Z (e.g., 0.0 or 1.0 for takeoff)
        if p_pos.size == 2:
            p_pos = np.append(p_pos, 0.0) 

        # Now p_pos is guaranteed to be size 3, so reshape(3,1) will work
        self.state = {
            "p": p_pos,
            "v": np.zeros(3),
            "B": 100.0
        }

        self.last_traj = np.tile(self.state["p"].reshape(3, 1), (1, horizon_n + 1))
    
        
        # Telemetry history for plotting
        self.history_p = []
        self.history_predictions = []

    def drone_model(self, accel, dt):
        """
        Drone dynamics
        """
        # Update velocity: v = v + a*dt
        self.state["v"] += accel * dt
        # Update position: p = p + v*dt
        self.state["p"] += self.state["v"] * dt + 0.5*accel*dt**2
        
        # Optional: simple linear battery drop for simulation
        c_batt = 0.01
        self.state["B"] -= c_batt * np.linalg.norm(accel)**2 * dt
        self.state["B"] = max(0, self.state["B"])

    def check_waypoints(self, threshold=0.5):
        """
        Updates the 'seen' status of waypoints in the [N x 4] numpy array.
        Expected row format: [x, y, seen_flag]
        """
        # self.waypoints is a numpy array of shape (num_regions, 4)
        for i in range(self.waypoints.shape[0]):
            # Check if the 'seen_flag' (index 2) is 0
            if self.waypoints[i, 2] == 0:
                # Calculate distance between current position and the waypoint coords (indices 0,1)
                dist = np.linalg.norm(self.state["p"] - self.waypoints[i, :2])
                
                if dist < threshold:
                    # Update the seen_flag to 1 in the numpy array
                    self.waypoints[i, 2] = 1
                    # print(f"[Drone {self.id}] Waypoint reached at {self.waypoints[i, :2]}")

    def log_telemetry(self, trajectory):
        """
        Saves data for post-flight plotting.
        """
        self.history_p.append(self.state["p"].copy())
        self.history_predictions.append(trajectory.copy())