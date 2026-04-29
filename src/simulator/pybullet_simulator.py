import time

import numpy as np
import pybullet as p
import pybullet_data


class Simulator:
    def __init__(
        self,
        drones,
        map3d,
        dt,
        gui=True,
        real_time=True,
        drone_radius=0.1,
        drone_visual="sphere",
    ):
        self.dt = float(dt)
        self.gui = bool(gui)
        self.real_time = bool(real_time)
        self.drone_radius = float(drone_radius)
        self.drone_visual = str(drone_visual).lower()

        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)

        self.bodies = {}
        self.body_labels = {}
        self._last_collision_pairs = set()
        self._last_heading_by_drone = {}

        self._build_world(map3d)
        self._spawn_drones(drones)

        if self.gui:
            x_mid = 0.5 * (map3d.x_bounds[0] + map3d.x_bounds[1])
            y_mid = 0.5 * (map3d.y_bounds[0] + map3d.y_bounds[1])
            z_span = map3d.z_bounds[1] - map3d.z_bounds[0]
            p.resetDebugVisualizerCamera(
                cameraDistance=max(
                    map3d.x_bounds[1] - map3d.x_bounds[0],
                    map3d.y_bounds[1] - map3d.y_bounds[0],
                ),
                cameraYaw=45,
                cameraPitch=-40,
                cameraTargetPosition=[x_mid, y_mid, 0.35 * z_span],
                physicsClientId=self.client_id,
            )

    def _build_world(self, map3d):
        x_min, x_max = map3d.x_bounds
        y_min, y_max = map3d.y_bounds
        z_min, z_max = map3d.z_bounds

        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)

        floor_half_extents = [0.5 * x_size, 0.5 * y_size, 0.02]
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=floor_half_extents,
            physicsClientId=self.client_id,
        )
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=floor_half_extents,
            rgbaColor=[0.82, 0.86, 0.9, 1.0],
            physicsClientId=self.client_id,
        )
        floor_body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=[x_mid, y_mid, z_min - 0.02],
            physicsClientId=self.client_id,
        )
        self.body_labels[floor_body] = "floor"

        wall_thickness = 0.05
        wall_height = max(0.5 * z_size, 0.5)
        wall_z = z_min + wall_height
        wall_specs = [
            ([wall_thickness, 0.5 * y_size, wall_height], [x_min - wall_thickness, y_mid, wall_z], "x_min"),
            ([wall_thickness, 0.5 * y_size, wall_height], [x_max + wall_thickness, y_mid, wall_z], "x_max"),
            ([0.5 * x_size, wall_thickness, wall_height], [x_mid, y_min - wall_thickness, wall_z], "y_min"),
            ([0.5 * x_size, wall_thickness, wall_height], [x_mid, y_max + wall_thickness, wall_z], "y_max"),
        ]
        for half_extents, position, label in wall_specs:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.15, 0.2, 0.25, 0.12],
                physicsClientId=self.client_id,
            )
            wall_body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
                physicsClientId=self.client_id,
            )
            self.body_labels[wall_body] = f"boundary_{label}"

        boundary_points = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
        ]
        for idx in range(len(boundary_points)):
            p.addUserDebugLine(
                boundary_points[idx],
                boundary_points[(idx + 1) % len(boundary_points)],
                lineColorRGB=[0.1, 0.1, 0.1],
                lineWidth=3.0,
                physicsClientId=self.client_id,
            )

        for index, obstacle in enumerate(map3d.obstacles):
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=float(obstacle.radius),
                height=float(obstacle.height),
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=float(obstacle.radius),
                length=float(obstacle.height),
                rgbaColor=[0.74, 0.2, 0.18, 0.8],
                physicsClientId=self.client_id,
            )
            obstacle_body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[obstacle.x, obstacle.y, z_min + obstacle.height / 2.0],
                physicsClientId=self.client_id,
            )
            self.body_labels[obstacle_body] = f"obstacle_{index}"

    def _spawn_drones(self, drones):
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.drone_radius,
            physicsClientId=self.client_id,
        )
        for drone in drones:
            color = self._drone_color(drone.id)
            visual = self._create_drone_visual(color)
            body = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=drone.state["p"].tolist(),
                physicsClientId=self.client_id,
            )
            p.changeDynamics(
                body,
                -1,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self.client_id,
            )
            self.bodies[drone.id] = body
            self.body_labels[body] = f"drone_{drone.id}"
            self._last_heading_by_drone[drone.id] = 0.0

    def _create_drone_visual(self, color):
        if self.drone_visual != "quadrotor":
            return p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.drone_radius,
                rgbaColor=color,
                physicsClientId=self.client_id,
            )

        scale = max(self.drone_radius, 0.12)
        body_half_extents = [0.28 * scale, 0.1 * scale, 0.08 * scale]
        arm_half_extents = [0.78 * scale, 0.045 * scale, 0.03 * scale]
        rotor_radius = 0.2 * scale
        rotor_height = 0.03 * scale
        rotor_offset = 0.55 * scale
        arm_height = 0.02 * scale

        arm_quat_pos = p.getQuaternionFromEuler([0.0, 0.0, np.pi / 4.0])
        arm_quat_neg = p.getQuaternionFromEuler([0.0, 0.0, -np.pi / 4.0])

        shape_types = [
            p.GEOM_BOX,
            p.GEOM_BOX,
            p.GEOM_BOX,
            p.GEOM_CYLINDER,
            p.GEOM_CYLINDER,
            p.GEOM_CYLINDER,
            p.GEOM_CYLINDER,
        ]
        half_extents = [
            body_half_extents,
            arm_half_extents,
            arm_half_extents,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        radii = [0.0, 0.0, 0.0, rotor_radius, rotor_radius, rotor_radius, rotor_radius]
        lengths = [0.0, 0.0, 0.0, rotor_height, rotor_height, rotor_height, rotor_height]
        rgba_colors = [
            color,
            [0.15, 0.15, 0.18, 1.0],
            [0.15, 0.15, 0.18, 1.0],
            [0.08, 0.08, 0.08, 1.0],
            [0.08, 0.08, 0.08, 1.0],
            [0.08, 0.08, 0.08, 1.0],
            [0.08, 0.08, 0.08, 1.0],
        ]
        positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, arm_height],
            [0.0, 0.0, arm_height],
            [rotor_offset, rotor_offset, arm_height],
            [-rotor_offset, rotor_offset, arm_height],
            [-rotor_offset, -rotor_offset, arm_height],
            [rotor_offset, -rotor_offset, arm_height],
        ]
        orientations = [
            [0.0, 0.0, 0.0, 1.0],
            arm_quat_pos,
            arm_quat_neg,
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        return p.createVisualShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            radii=radii,
            lengths=lengths,
            rgbaColors=rgba_colors,
            visualFramePositions=positions,
            visualFrameOrientations=orientations,
            physicsClientId=self.client_id,
        )

    def _drone_color(self, drone_id):
        palette = [
            [0.16, 0.52, 0.96, 1.0],
            [0.14, 0.7, 0.35, 1.0],
            [0.94, 0.58, 0.14, 1.0],
            [0.62, 0.31, 0.89, 1.0],
            [0.9, 0.23, 0.49, 1.0],
        ]
        return palette[drone_id % len(palette)]

    def _drone_orientation(self, drone):
        if self.drone_visual != "quadrotor":
            return [0.0, 0.0, 0.0, 1.0]

        velocity_xy = np.asarray(drone.state["v"][:2], dtype=float)
        speed_xy = np.linalg.norm(velocity_xy)

        yaw = self._last_heading_by_drone.get(drone.id, 0.0)
        if speed_xy > 1e-4:
            yaw = float(np.arctan2(velocity_xy[1], velocity_xy[0]))
            self._last_heading_by_drone[drone.id] = yaw

        accel = np.asarray(drone.state.get("a", np.zeros(3)), dtype=float)
        pitch = float(np.clip(-0.08 * accel[0], -0.3, 0.3))
        roll = float(np.clip(0.08 * accel[1], -0.3, 0.3))

        return p.getQuaternionFromEuler([roll, pitch, yaw])

    def sync_drone_state(self, drone):
        body = self.bodies[drone.id]
        p.resetBasePositionAndOrientation(
            body,
            drone.state["p"].tolist(),
            self._drone_orientation(drone),
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            body,
            linearVelocity=drone.state["v"].tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def sync_all_drones(self, drones):
        for drone in drones:
            self.sync_drone_state(drone)

    def step(self):
        p.stepSimulation(physicsClientId=self.client_id)
        if self.real_time and self.gui:
            time.sleep(self.dt)

    def get_collisions(self, drones):
        p.performCollisionDetection(physicsClientId=self.client_id)
        collisions = {}
        current_pairs = set()

        for drone in drones:
            body = self.bodies[drone.id]
            contacts = p.getContactPoints(bodyA=body, physicsClientId=self.client_id)
            labels = set()

            for contact in contacts:
                other_body = contact[2]
                other_label = self.body_labels.get(other_body, f"body_{other_body}")
                if other_label == "floor":
                    continue
                if other_label.startswith("drone_") and other_label == f"drone_{drone.id}":
                    continue
                pair = (f"drone_{drone.id}", other_label)
                labels.add(other_label)
                current_pairs.add(pair)

            if labels:
                collisions[drone.id] = sorted(labels)

        self._last_collision_pairs = current_pairs
        return collisions

    def get_new_collisions(self, drones):
        previous_pairs = set(self._last_collision_pairs)
        collisions = self.get_collisions(drones)
        if not collisions:
            return {}

        new_collision_map = {}
        for drone_id, labels in collisions.items():
            fresh_labels = []
            for label in labels:
                pair = (f"drone_{drone_id}", label)
                if pair not in previous_pairs:
                    fresh_labels.append(label)
            if fresh_labels:
                new_collision_map[drone_id] = fresh_labels

        return new_collision_map

    def disconnect(self):
        if p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
