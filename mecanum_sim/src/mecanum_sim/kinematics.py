"""Parametric kinematics, direction-preserving desaturation, and SE(2) dead-reckoned odometry."""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np


class MecanumKinematics:
    """Parametric forward and inverse kinematics for 4-wheel mecanum drivetrains.

    Wheel indexing convention:
      0: Front-Left (FL)  at (+Lx, +Ly), roller tilt +45 deg
      1: Front-Right (FR) at (+Lx, -Ly), roller tilt -45 deg
      2: Rear-Left (RL)   at (-Lx, +Ly), roller tilt -45 deg
      3: Rear-Right (RR)  at (-Lx, -Ly), roller tilt +45 deg
    """

    def __init__(self, wheel_radius: float, lx: float, ly: float):
        if wheel_radius <= 0:
            raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
        if lx <= 0 or ly <= 0:
            raise ValueError(f"lx and ly must be positive, got lx={lx}, ly={ly}")

        self.R = float(wheel_radius)
        self.lx = float(lx)
        self.ly = float(ly)
        self.L = self.lx + self.ly

        inv_R = 1.0 / self.R
        self.J_inv = inv_R * np.array([
            [1.0, -1.0, -self.L],
            [1.0,  1.0,  self.L],
            [1.0,  1.0, -self.L],
            [1.0, -1.0,  self.L],
        ], dtype=float)

        scale_fwd = self.R / 4.0
        self.J_fwd = scale_fwd * np.array([
            [1.0,  1.0,  1.0,  1.0],
            [-1.0, 1.0,  1.0, -1.0],
            [-1.0 / self.L, 1.0 / self.L, -1.0 / self.L, 1.0 / self.L],
        ], dtype=float)

    def inverse(self, twist: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Transforms chassis twist [vx, vy, wz] into desired wheel angular speeds [w_fl, w_fr, w_rl, w_rr] (rad/s)."""
        v = np.asarray(twist, dtype=float).reshape(3)
        return self.J_inv @ v

    def forward(self, wheel_speeds: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Transforms measured wheel angular speeds [w_fl, w_fr, w_rl, w_rr] (rad/s) into chassis twist [vx, vy, wz]."""
        w = np.asarray(wheel_speeds, dtype=float).reshape(4)
        return self.J_fwd @ w

    @staticmethod
    def desaturate_wheel_speeds(
        wheel_speeds: Union[np.ndarray, list, tuple],
        max_wheel_speed: float,
    ) -> np.ndarray:
        """Scales all wheel speeds uniformly to preserve the velocity vector heading during saturation."""
        w = np.asarray(wheel_speeds, dtype=float).copy()
        if max_wheel_speed <= 0:
            return w

        peak = float(np.max(np.abs(w)))
        if peak > max_wheel_speed:
            scale = max_wheel_speed / peak
            return w * scale
        return w


class WheelOdometryEstimator:
    """True dead-reckoned SE(2) odometry accumulator decoupled from simulation ground-truth."""

    def __init__(
        self,
        kinematics: MecanumKinematics,
        initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.kinematics = kinematics
        self.x = float(initial_pose[0])
        self.y = float(initial_pose[1])
        self.theta = float(initial_pose[2])

        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

    def reset(self, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.x = float(pose[0])
        self.y = float(pose[1])
        self.theta = float(pose[2])
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

    def update(
        self,
        wheel_speeds: Union[np.ndarray, list, tuple],
        dt: float,
    ) -> Tuple[float, float, float]:
        """Integrates measured wheel velocities across time step dt using midpoint SE(2) integration."""
        if dt <= 0:
            return self.pose

        twist = self.kinematics.forward(wheel_speeds)
        vx_enc, vy_enc, wz_enc = float(twist[0]), float(twist[1]), float(twist[2])

        self.last_vx = vx_enc
        self.last_vy = vy_enc
        self.last_wz = wz_enc

        # Midpoint planar SE(2) integration
        theta_mid = self.theta + 0.5 * wz_enc * dt
        cos_m = math.cos(theta_mid)
        sin_m = math.sin(theta_mid)

        self.x += (vx_enc * cos_m - vy_enc * sin_m) * dt
        self.y += (vx_enc * sin_m + vy_enc * cos_m) * dt
        new_theta = self.theta + wz_enc * dt
        self.theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

        return self.pose

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Returns dead-reckoned pose (x_odom, y_odom, theta_odom)."""
        return self.x, self.y, self.theta

    @property
    def twist(self) -> Tuple[float, float, float]:
        """Returns instantaneous estimated body twist (vx, vy, wz)."""
        return self.last_vx, self.last_vy, self.last_wz

