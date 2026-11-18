"""Outer-loop path tracking controller with feedforward velocity and body-frame pose correction."""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np

from mecanum_sim.config import RobotConfig
from mecanum_sim.trajectory import TrajectoryPoint


class PathTracker:
    """Computes body twist combining feedforward trajectory velocity and proportional body-frame pose trim."""

    def __init__(
        self,
        robot_config: RobotConfig,
        enable_pose_correction: bool = True,
    ):
        self.robot = robot_config
        self.enable_pose_correction = enable_pose_correction

        ctrl = self.robot.control
        self.kp_x = float(ctrl.pose_kp_x)
        self.kp_y = float(ctrl.pose_kp_y)
        self.kp_theta = float(ctrl.pose_kp_theta)

        self.last_ex_body = 0.0
        self.last_ey_body = 0.0
        self.last_etheta = 0.0

    def compute_twist(
        self,
        target_point: TrajectoryPoint,
        current_pose: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Calculates commanded chassis twist [vx_cmd, vy_cmd, wz_cmd] in body frame.

        Returns:
            (twist_cmd, (ex_body, ey_body, etheta))
        """
        x_cur, y_cur, theta_cur = current_pose

        if not self.enable_pose_correction:
            # Open-loop feedforward velocity
            twist_cmd = np.array([target_point.vx, target_point.vy, target_point.wz], dtype=float)
            return twist_cmd, (0.0, 0.0, 0.0)

        # 1. World frame position error
        dx_world = target_point.x - x_cur
        dy_world = target_point.y - y_cur

        # 2. Transform position error into robot body frame (REP-103)
        cos_th = math.cos(theta_cur)
        sin_th = math.sin(theta_cur)

        ex_body = cos_th * dx_world + sin_th * dy_world
        ey_body = -sin_th * dx_world + cos_th * dy_world

        # 3. Heading error wrapped to [-pi, +pi]
        dth = target_point.theta - theta_cur
        etheta = math.atan2(math.sin(dth), math.cos(dth))

        self.last_ex_body = ex_body
        self.last_ey_body = ey_body
        self.last_etheta = etheta

        # 4. Feedforward + Proportional body-frame correction
        vx_cmd = target_point.vx + self.kp_x * ex_body
        vy_cmd = target_point.vy + self.kp_y * ey_body
        wz_cmd = target_point.wz + self.kp_theta * etheta

        return np.array([vx_cmd, vy_cmd, wz_cmd], dtype=float), (ex_body, ey_body, etheta)

