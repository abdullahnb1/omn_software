"""Velocity controllers, chassis acceleration rate limiter, and multi-wheel coordination."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from mecanum_sim.config import RobotConfig
from mecanum_sim.kinematics import MecanumKinematics


class ChassisRateLimiter:
    """Enforces max velocity and acceleration limits on commanded chassis twist [vx, vy, wz]."""

    def __init__(
        self,
        max_vx: float = 0.80,
        max_vy: float = 0.80,
        max_wz: float = 1.50,
        max_ax: float = 1.50,
        max_ay: float = 1.50,
        max_awz: float = 3.00,
    ):
        self.max_v = np.array([abs(max_vx), abs(max_vy), abs(max_wz)], dtype=float)
        self.max_a = np.array([abs(max_ax), abs(max_ay), abs(max_awz)], dtype=float)
        self.current_v = np.zeros(3, dtype=float)

    def reset(self, v_init: Optional[Union[np.ndarray, list, tuple]] = None) -> None:
        if v_init is None:
            self.current_v.fill(0.0)
        else:
            self.current_v = np.asarray(v_init, dtype=float).copy()

    def update(self, target_twist: Union[np.ndarray, list, tuple], dt: float) -> np.ndarray:
        """Applies velocity bounds and acceleration slew-rate limits across time step dt."""
        v_target = np.asarray(target_twist, dtype=float).reshape(3)
        # 1. Clamp to max velocity limits
        v_clamped = np.clip(v_target, -self.max_v, self.max_v)

        if dt <= 0:
            return self.current_v.copy()

        # 2. Acceleration slew rate limiting
        dv = v_clamped - self.current_v
        max_dv = self.max_a * dt
        dv_limited = np.clip(dv, -max_dv, max_dv)

        self.current_v += dv_limited
        return self.current_v.copy()


class WheelVelocityController:
    """Individual wheel velocity servo with angular acceleration slew limiter and anti-windup PI trim."""

    def __init__(
        self,
        max_wheel_speed: float = 35.0,
        max_wheel_alpha: float = 60.0,
        kp: float = 0.05,
        ki: float = 0.40,
        kd: float = 0.0,
    ):
        self.max_w = float(abs(max_wheel_speed))
        self.max_alpha = float(abs(max_wheel_alpha))
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.w_slew = 0.0
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self, initial_w: float = 0.0) -> None:
        self.w_slew = float(initial_w)
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, desired_w: float, measured_w: float, dt: float) -> float:
        """Computes slewed reference and anti-windup trim command in rad/s."""
        # 1. Acceleration slew rate limiting
        if dt > 0:
            dw = desired_w - self.w_slew
            max_dw = self.max_alpha * dt
            self.w_slew += float(np.clip(dw, -max_dw, max_dw))
        else:
            self.w_slew = desired_w

        # Slew output clamped to max wheel speed
        self.w_slew = float(np.clip(self.w_slew, -self.max_w, self.max_w))

        # 2. Velocity error
        error = self.w_slew - measured_w

        # 3. Derivative error with simple filtering
        d_error = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # 4. Proportional + Derivative trim
        pd_trim = self.kp * error + self.kd * d_error

        # 5. Tentative command without new integral
        tentative_cmd = self.w_slew + pd_trim + self.ki * self.integral

        # 6. Anti-windup conditional integration
        # Only integrate if not saturated, or if error pushes away from saturation
        is_saturated_pos = tentative_cmd >= self.max_w
        is_saturated_neg = tentative_cmd <= -self.max_w

        if not (is_saturated_pos and error > 0) and not (is_saturated_neg and error < 0):
            if dt > 0:
                self.integral += error * dt
                # Bound integral term
                max_int = self.max_w * 0.25
                self.integral = float(np.clip(self.integral, -max_int, max_int))

        # Final command
        cmd = self.w_slew + pd_trim + self.ki * self.integral
        return float(np.clip(cmd, -self.max_w, self.max_w))


# Alias WheelPID for backward/spec compatibility
WheelPID = WheelVelocityController


class MecanumController:
    """Two-layer controller coordinating chassis rate limiting, kinematics, and 4 wheel servos."""

    def __init__(self, robot_config: RobotConfig):
        self.robot = robot_config
        self.kinematics = MecanumKinematics(
            wheel_radius=self.robot.wheel_radius,
            lx=self.robot.lx,
            ly=self.robot.ly,
        )

        lim = self.robot.limits
        self.rate_limiter = ChassisRateLimiter(
            max_vx=lim.max_vx,
            max_vy=lim.max_vy,
            max_wz=lim.max_wz,
            max_ax=lim.max_ax,
            max_ay=lim.max_ay,
            max_awz=lim.max_awz,
        )

        ctrl = self.robot.control
        act = self.robot.actuator
        self.wheel_controllers = [
            WheelVelocityController(
                max_wheel_speed=act.max_speed,
                max_wheel_alpha=act.max_alpha,
                kp=ctrl.kp,
                ki=ctrl.ki,
                kd=ctrl.kd,
            )
            for _ in range(4)
        ]

    def reset(self) -> None:
        self.rate_limiter.reset()
        for wc in self.wheel_controllers:
            wc.reset()

    def update(
        self,
        commanded_twist: Union[np.ndarray, list, tuple],
        measured_wheel_speeds: Union[np.ndarray, list, tuple],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes chassis twist command into rate-limited, desaturated, slewed wheel commands.

        Returns:
            (actuator_commands_rad_s, desaturated_target_w_rad_s, rate_limited_twist)
        """
        # 1. Chassis velocity and acceleration limits
        twist_limited = self.rate_limiter.update(commanded_twist, dt)

        # 2. Inverse kinematics
        raw_w = self.kinematics.inverse(twist_limited)

        # 3. Direction-preserving wheel velocity desaturation
        desat_w = self.kinematics.desaturate_wheel_speeds(
            raw_w,
            max_wheel_speed=self.robot.actuator.max_speed,
        )

        # 4. Wheel-level slew rate limiting and velocity trim
        meas_w = np.asarray(measured_wheel_speeds, dtype=float).reshape(4)
        actuator_cmds = np.zeros(4, dtype=float)
        for i in range(4):
            actuator_cmds[i] = self.wheel_controllers[i].update(
                desired_w=desat_w[i],
                measured_w=meas_w[i],
                dt=dt,
            )

        return actuator_cmds, desat_w, twist_limited

