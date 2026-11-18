"""Cubic spline trajectory generation, 10 benchmark presets, and JSON waypoint serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    theta: float
    vx: float
    vy: float
    wz: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class CubicSplineTrajectory:
    """C^1-continuous cubic spline trajectory with smooth S-curve velocity profiling and settling phase."""

    def __init__(
        self,
        waypoints: List[Dict[str, float]],
        nominal_speed: float = 0.40,
        settle_time: float = 1.0,
        fixed_duration: Optional[float] = None,
    ):
        if len(waypoints) < 2:
            raise ValueError("Trajectory requires at least 2 waypoints.")

        self.waypoints = [dict(wp) for wp in waypoints]
        self.nominal_speed = max(0.05, float(nominal_speed))
        self.settle_time = max(0.5, float(settle_time))

        self.xs = np.array([wp["x"] for wp in self.waypoints], dtype=float)
        self.ys = np.array([wp["y"] for wp in self.waypoints], dtype=float)
        self.thetas = np.array([wp.get("theta", 0.0) for wp in self.waypoints], dtype=float)

        # Unwrap thetas for smooth angular interpolation
        self.thetas = np.unwrap(self.thetas)

        # Compute cumulative chord distances
        dx = np.diff(self.xs)
        dy = np.diff(self.ys)
        dth = np.diff(self.thetas)
        char_r = 0.20  # Equivalent radius for rotational distance

        ds = np.sqrt(dx**2 + dy**2) + np.abs(dth) * char_r
        # Avoid zero intervals for identical consecutive waypoints
        ds = np.maximum(ds, 1e-4)

        self.s_cumulative = np.insert(np.cumsum(ds), 0, 0.0)
        self.total_s = float(self.s_cumulative[-1])

        # Compute motion duration based on nominal speed
        self.motion_duration = max(2.0, self.total_s / self.nominal_speed)
        self.total_duration = self.motion_duration + self.settle_time

        # Fit cubic splines parameterized by arc length s
        # If waypoints count is 2, CubicSpline bc_type='natural' works
        self.cs_x = CubicSpline(self.s_cumulative, self.xs, bc_type="natural")
        self.cs_y = CubicSpline(self.s_cumulative, self.ys, bc_type="natural")
        self.cs_theta = CubicSpline(self.s_cumulative, self.thetas, bc_type="natural")

    def _s_curve(self, t: float) -> Tuple[float, float]:
        """Quintic polynomial time-scaling s(t) and s_dot(t) ensuring C^1 rest-to-rest motion."""
        if t <= 0.0:
            return 0.0, 0.0
        if t >= self.motion_duration:
            return self.total_s, 0.0

        tau = t / self.motion_duration
        # Quintic polynomial: 10*tau^3 - 15*tau^4 + 6*tau^5
        s_norm = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        # Derivative: (30*tau^2 - 60*tau^3 + 30*tau^4) / T
        s_norm_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / self.motion_duration

        s = s_norm * self.total_s
        s_dot = s_norm_dot * self.total_s
        return s, s_dot

    def evaluate(self, t: float) -> TrajectoryPoint:
        """Evaluates desired pose and feedforward twist at simulation time t."""
        s, s_dot = self._s_curve(t)

        x = float(self.cs_x(s))
        y = float(self.cs_y(s))
        theta = float(self.cs_theta(s))
        # Wrap theta to [-pi, +pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        # Velocity via chain rule: dX/dt = dX/ds * ds/dt
        vx = float(self.cs_x(s, 1) * s_dot)
        vy = float(self.cs_y(s, 1) * s_dot)
        wz = float(self.cs_theta(s, 1) * s_dot)

        return TrajectoryPoint(
            t=float(t),
            x=x,
            y=y,
            theta=theta,
            vx=vx,
            vy=vy,
            wz=wz,
        )


# ==============================================================================
# Waypoint JSON Serialization
# ==============================================================================

def save_waypoints_json(
    waypoints: List[Dict[str, float]],
    file_path: Union[str, Path],
    nominal_speed: float = 0.40,
) -> None:
    """Exports waypoints and nominal speed to a portable JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nominal_speed": float(nominal_speed),
        "num_waypoints": len(waypoints),
        "waypoints": [
            {
                "x": round(float(wp["x"]), 4),
                "y": round(float(wp["y"]), 4),
                "theta": round(float(wp.get("theta", 0.0)), 4),
            }
            for wp in waypoints
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_waypoints_json(file_path: Union[str, Path]) -> Tuple[List[Dict[str, float]], float]:
    """Loads waypoints and nominal speed from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Waypoints file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wps = data.get("waypoints", [])
    speed = float(data.get("nominal_speed", 0.40))
    return wps, speed


# ==============================================================================
# 10 Benchmark Preset Trajectories
# ==============================================================================

def get_preset_waypoints(preset_name: str) -> List[Dict[str, float]]:
    """Returns canonical waypoint sequence for any of the 10 benchmark trajectories."""
    name = preset_name.strip().lower()

    # Aliases
    alias_map = {
        "forward": "forward_reverse_stress",
        "forward_reverse": "forward_reverse_stress",
        "lateral": "pure_lateral_crabbing",
        "crabbing": "pure_lateral_crabbing",
        "diagonal": "diagonal_slalom",
        "slalom": "diagonal_slalom",
        "pirouette": "zero_radius_pirouette",
        "rotation": "zero_radius_pirouette",
        "square": "omni_square",
        "omni_square": "omni_square",
        "cloverleaf": "cloverleaf_spline",
        "cloverleaf_spline": "cloverleaf_spline",
        "figure8": "figure8_continuous",
        "figure8_continuous": "figure8_continuous",
        "doorway": "doorway_and_threshold",
        "doorway_and_threshold": "doorway_and_threshold",
        "liquid": "liquid_payload_gentle",
        "gentle": "liquid_payload_gentle",
        "liquid_payload_gentle": "liquid_payload_gentle",
        "corridor": "obstacle_avoidance_corridor",
        "obstacle": "obstacle_avoidance_corridor",
        "obstacle_avoidance_corridor": "obstacle_avoidance_corridor",
    }
    canonical = alias_map.get(name, name)

    if canonical == "forward_reverse_stress":
        return [
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.8, "y": 0.0, "theta": 0.0},
            {"x": 1.8, "y": 0.0, "theta": 0.0},
            {"x": 0.8, "y": 0.0, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "pure_lateral_crabbing":
        return [
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.0, "y": 0.6, "theta": 0.0},
            {"x": 0.0, "y": 1.2, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.0, "y": -1.2, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "diagonal_slalom":
        return [
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.8, "y": 0.8, "theta": 0.0},
            {"x": 1.6, "y": 0.0, "theta": 0.0},
            {"x": 2.4, "y": 0.8, "theta": 0.0},
            {"x": 3.2, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "zero_radius_pirouette":
        return [
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": math.pi / 2.0},
            {"x": 0.0, "y": 0.0, "theta": math.pi},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": -math.pi},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "omni_square":
        return [
            {"x": 0.0, "y": 0.0, "theta": 0.0},
            {"x": 1.2, "y": 0.0, "theta": 0.0},
            {"x": 1.2, "y": 1.2, "theta": 0.0},
            {"x": 0.0, "y": 1.2, "theta": 0.0},
            {"x": 0.0, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "cloverleaf_spline":
        wps = []
        n_pts = 16
        for i in range(n_pts + 1):
            phi = 2.0 * math.pi * i / n_pts
            # 4-leaf clover formula: r = 1.0 + 0.5 * cos(4 * phi)
            r = 0.80 * (1.0 + 0.40 * math.cos(4.0 * phi))
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            wps.append({"x": round(x, 4), "y": round(y, 4), "theta": 0.0})
        return wps

    elif canonical == "figure8_continuous":
        wps = []
        n_pts = 16
        for i in range(n_pts + 1):
            u = 2.0 * math.pi * i / n_pts
            x = 1.40 * math.sin(u)
            y = 0.90 * math.sin(u) * math.cos(u)
            wps.append({"x": round(x, 4), "y": round(y, 4), "theta": 0.0})
        return wps

    elif canonical == "doorway_and_threshold":
        # Straight transit from X=-1.8 through doorway at X=0 to X=+1.8
        return [
            {"x": -1.8, "y": 0.0, "theta": 0.0},
            {"x": -0.8, "y": 0.0, "theta": 0.0},
            {"x":  0.0, "y": 0.0, "theta": 0.0},
            {"x":  0.8, "y": 0.0, "theta": 0.0},
            {"x":  1.8, "y": 0.0, "theta": 0.0},
        ]

    elif canonical == "liquid_payload_gentle":
        return [
            {"x": 0.0, "y":  0.00, "theta": 0.0},
            {"x": 0.8, "y":  0.20, "theta": 0.0},
            {"x": 1.6, "y": -0.20, "theta": 0.0},
            {"x": 2.4, "y":  0.00, "theta": 0.0},
        ]

    elif canonical == "obstacle_avoidance_corridor":
        # Weaves through obstacles and doorway
        return [
            {"x": -2.0, "y": -1.0, "theta": 0.0},
            {"x": -1.2, "y": -0.3, "theta": 0.1},
            {"x":  0.0, "y":  0.0, "theta": 0.0},
            {"x":  1.0, "y":  0.4, "theta": -0.1},
            {"x":  2.0, "y":  1.0, "theta": 0.0},
        ]

    else:
        raise ValueError(
            f"Unknown trajectory preset: '{preset_name}'. "
            f"Available: forward_reverse_stress, pure_lateral_crabbing, diagonal_slalom, "
            f"zero_radius_pirouette, omni_square, cloverleaf_spline, figure8_continuous, "
            f"doorway_and_threshold, liquid_payload_gentle, obstacle_avoidance_corridor"
        )


def create_trajectory(
    preset_or_waypoints: Union[str, List[Dict[str, float]]],
    nominal_speed: float = 0.40,
    settle_time: float = 1.0,
    duration: Optional[float] = None,
) -> CubicSplineTrajectory:
    """Convenience factory creating a CubicSplineTrajectory from preset name or waypoints list."""
    if isinstance(preset_or_waypoints, str):
        wps = get_preset_waypoints(preset_or_waypoints)
    else:
        wps = preset_or_waypoints

    return CubicSplineTrajectory(
        waypoints=wps,
        nominal_speed=nominal_speed,
        settle_time=settle_time,
        fixed_duration=duration,
    )
