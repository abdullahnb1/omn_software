"""Type-safe configuration schemas, multi-robot loaders, and dynamic inertia calculations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np


@dataclass
class ChassisConfig:
    length: float
    width: float
    height: float
    base_thickness: float
    axle_height: float
    underbelly_clearance: float
    mass: float
    com: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.02])
    color: List[float] = field(default_factory=lambda: [0.2, 0.35, 0.65, 1.0])


@dataclass
class WheelsConfig:
    radius: float
    width: float
    lx: float
    ly: float
    assembly_mass: float
    num_rollers: int
    roller_radius: float
    roller_mass: float
    roller_color: List[float] = field(default_factory=lambda: [0.85, 0.25, 0.20, 1.0])
    damping: float = 0.005
    armature: float = 0.002
    roller_damping: float = 0.0001
    roller_armature: float = 0.00002

    @property
    def characteristic_length(self) -> float:
        return self.lx + self.ly

    @property
    def pitch_radius(self) -> float:
        return self.radius - self.roller_radius

    @property
    def total_width(self) -> float:
        return 2.0 * self.ly + self.width


@dataclass
class ActuatorConfig:
    type: str = "velocity"
    kv: float = 0.80
    max_torque: float = 2.5
    max_speed: float = 35.0
    max_alpha: float = 60.0


@dataclass
class LimitsConfig:
    nominal_speed: float = 0.40
    max_vx: float = 0.80
    max_vy: float = 0.80
    max_wz: float = 1.50
    max_ax: float = 1.50
    max_ay: float = 1.50
    max_awz: float = 3.00


@dataclass
class ControlConfig:
    kp: float = 0.05
    ki: float = 0.40
    kd: float = 0.0
    pose_kp_x: float = 2.5
    pose_kp_y: float = 2.5
    pose_kp_theta: float = 2.0


@dataclass
class PayloadConfig:
    max_mass: float = 20.0
    default_mass: float = 0.0
    deck_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.06])
    mass: float = 0.0
    offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class RobotConfig:
    name: str
    aliases: List[str]
    description: str
    chassis: ChassisConfig
    wheels: WheelsConfig
    actuator: ActuatorConfig
    limits: LimitsConfig
    control: ControlConfig
    payload: PayloadConfig

    @property
    def wheel_radius(self) -> float:
        return self.wheels.radius

    @property
    def lx(self) -> float:
        return self.wheels.lx

    @property
    def ly(self) -> float:
        return self.wheels.ly

    @property
    def characteristic_length(self) -> float:
        return self.wheels.characteristic_length

    @property
    def unladen_mass(self) -> float:
        return self.chassis.mass + 4.0 * self.wheels.assembly_mass

    @property
    def total_mass(self) -> float:
        return self.unladen_mass + self.payload.mass

    @property
    def nav2_footprint(self) -> List[List[float]]:
        """Dynamic 2D bounding footprint polygon in base_footprint coordinates."""
        half_l = self.chassis.length / 2.0
        half_w = max(self.chassis.width, self.wheels.total_width) / 2.0
        return [
            [round(half_l, 4), round(half_w, 4)],
            [round(half_l, 4), round(-half_w, 4)],
            [round(-half_l, 4), round(-half_w, 4)],
            [round(-half_l, 4), round(half_w, 4)],
        ]

    def compute_composite_inertia(
        self,
        payload_mass: Optional[float] = None,
        payload_offset: Optional[List[float]] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculates combined mass, CoM position, and diagonal moments of inertia.

        Applies the Parallel Axis Theorem for chassis, 4 wheels, and external payload.
        Returns:
            (total_mass_kg, com_vector_3d, diag_inertia_tensor_3d)
        """
        p_mass = self.payload.mass if payload_mass is None else float(payload_mass)
        p_off = list(self.payload.offset) if payload_offset is None else [float(x) for x in payload_offset]

        m_chassis = self.chassis.mass
        r_chassis = np.array(self.chassis.com, dtype=float)

        # Approximate chassis as a solid cuboid
        l_c, w_c, h_c = self.chassis.length, self.chassis.width, self.chassis.height
        i_chassis_com = np.array([
            (1.0 / 12.0) * m_chassis * (w_c**2 + h_c**2),
            (1.0 / 12.0) * m_chassis * (l_c**2 + h_c**2),
            (1.0 / 12.0) * m_chassis * (l_c**2 + w_c**2),
        ])

        # 4 wheel assemblies
        m_w = self.wheels.assembly_mass
        lx, ly = self.wheels.lx, self.wheels.ly
        # Wheel centers relative to base_link (z=0)
        wheel_positions = [
            np.array([+lx, +ly, 0.0]),
            np.array([+lx, -ly, 0.0]),
            np.array([-lx, +ly, 0.0]),
            np.array([-lx, -ly, 0.0]),
        ]
        # Approximate each wheel as cylinder of radius R and width w_w along Y
        r_w, ww = self.wheels.radius, self.wheels.width
        i_wheel_com = np.array([
            (1.0 / 12.0) * m_w * (3.0 * r_w**2 + ww**2),
            0.5 * m_w * r_w**2,
            (1.0 / 12.0) * m_w * (3.0 * r_w**2 + ww**2),
        ])

        # Payload
        r_payload = np.array(self.payload.deck_pos, dtype=float) + np.array(p_off, dtype=float)
        # Approximate payload as small cuboid
        i_payload_com = np.array([
            (1.0 / 12.0) * p_mass * (0.2**2 + 0.1**2) if p_mass > 0 else 0.0,
            (1.0 / 12.0) * p_mass * (0.2**2 + 0.1**2) if p_mass > 0 else 0.0,
            (1.0 / 12.0) * p_mass * (0.2**2 + 0.2**2) if p_mass > 0 else 0.0,
        ])

        total_mass = m_chassis + 4.0 * m_w + p_mass

        # Composite Center of Mass
        r_com_total = (
            m_chassis * r_chassis
            + sum(m_w * p for p in wheel_positions)
            + p_mass * r_payload
        ) / total_mass

        # Parallel Axis Theorem for each component around r_com_total
        def pat(i_local: np.ndarray, m: float, pos: np.ndarray) -> np.ndarray:
            d = pos - r_com_total
            return i_local + m * np.array([
                d[1]**2 + d[2]**2,
                d[0]**2 + d[2]**2,
                d[0]**2 + d[1]**2,
            ])

        i_total = pat(i_chassis_com, m_chassis, r_chassis)
        for p in wheel_positions:
            i_total += pat(i_wheel_com, m_w, p)
        if p_mass > 0:
            i_total += pat(i_payload_com, p_mass, r_payload)

        return float(total_mass), r_com_total, i_total


@dataclass
class SimulationConfig:
    physics_timestep: float = 0.002
    control_timestep: float = 0.010
    default_duration: float = 10.0
    floor_friction: float = 0.8
    impratio: float = 10.0
    solver: str = "Newton"
    cone: str = "elliptic"
    iterations: int = 60
    tolerance: float = 1e-8


@dataclass
class ArenaConfig:
    size_x: float = 8.0
    size_y: float = 8.0
    wall_height: float = 0.50
    wall_thickness: float = 0.08
    doorway_width: float = 0.80
    doorway_threshold_height: float = 0.008
    open_threshold_height: float = 0.005


@dataclass
class RenderingConfig:
    video_fps: int = 30
    pip_width: int = 240
    pip_height: int = 180
    pip_fps: int = 15


@dataclass
class SimConfig:
    simulation: SimulationConfig
    arena: ArenaConfig
    rendering: RenderingConfig
    robot: RobotConfig
    default_robot: str = "robot_1_compact"
    default_arena: str = "auto"


def _get_project_root() -> Path:
    """Find repository root containing configs/."""
    # Check current directory
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    # Check package file location
    pkg_root = Path(__file__).resolve().parent.parent.parent
    if (pkg_root / "configs").is_dir():
        return pkg_root
    return cwd


def list_available_robots(configs_dir: Optional[Path] = None) -> List[str]:
    """Scans configs/robots/ and returns list of robot profile keys."""
    root = configs_dir or (_get_project_root() / "configs" / "robots")
    if not root.is_dir():
        return ["robot_1_compact", "robot_2_heavy", "robot_3_custom"]
    robots = []
    for f in sorted(root.glob("*.yaml")):
        robots.append(f.stem)
    return robots


def _dict_to_robot_config(data: Dict[str, Any]) -> RobotConfig:
    """Converts parsed YAML dictionary to a RobotConfig instance."""
    chassis = ChassisConfig(**data["chassis"])
    wheels = WheelsConfig(**data["wheels"])
    actuator = ActuatorConfig(**data.get("actuator", {}))
    limits = LimitsConfig(**data.get("limits", {}))
    control = ControlConfig(**data.get("control", {}))
    payload = PayloadConfig(**data.get("payload", {}))
    return RobotConfig(
        name=data["name"],
        aliases=data.get("aliases", []),
        description=data.get("description", ""),
        chassis=chassis,
        wheels=wheels,
        actuator=actuator,
        limits=limits,
        control=control,
        payload=payload,
    )


def load_config(
    robot: str = "1",
    arena: Optional[str] = None,
    payload_mass: float = 0.0,
    payload_offset: Optional[List[float]] = None,
    floor_friction: Optional[float] = None,
    config_dir: Optional[Path] = None,
) -> SimConfig:
    """Loads default configuration and merges with chosen robot and runtime overrides."""
    root = config_dir or (_get_project_root() / "configs")
    default_file = root / "default_config.yaml"

    if default_file.exists():
        with open(default_file, "r", encoding="utf-8") as f:
            raw_default = yaml.safe_load(f) or {}
    else:
        raw_default = {}

    sim_cfg = SimulationConfig(**raw_default.get("simulation", {}))
    arena_cfg = ArenaConfig(**raw_default.get("arena", {}))
    rend_cfg = RenderingConfig(**raw_default.get("rendering", {}))

    if floor_friction is not None:
        sim_cfg.floor_friction = float(floor_friction)

    # Resolve robot key from aliases or filenames
    robot_target = str(robot).strip().lower()
    robot_files = list((root / "robots").glob("*.yaml")) if (root / "robots").is_dir() else []

    selected_data: Optional[Dict[str, Any]] = None
    for rf in robot_files:
        with open(rf, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
            if not d:
                continue
            name = d.get("name", "").lower()
            aliases = [str(a).lower() for a in d.get("aliases", [])]
            if robot_target == name or robot_target in aliases or robot_target == rf.stem.lower():
                selected_data = d
                break

    if selected_data is None:
        # Fallback heuristic mapping
        mapping = {
            "1": "robot_1_compact",
            "compact": "robot_1_compact",
            "robot_1_compact": "robot_1_compact",
            "2": "robot_2_heavy",
            "heavy": "robot_2_heavy",
            "robot_2_heavy": "robot_2_heavy",
            "3": "robot_3_custom",
            "custom": "robot_3_custom",
            "robot_3_custom": "robot_3_custom",
        }
        resolved = mapping.get(robot_target, "robot_1_compact")
        target_file = root / "robots" / f"{resolved}.yaml"
        if target_file.exists():
            with open(target_file, "r", encoding="utf-8") as f:
                selected_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown robot identifier: '{robot}'. Available: {list_available_robots(root / 'robots')}")

    robot_cfg = _dict_to_robot_config(selected_data)

    if payload_mass > 0.0:
        robot_cfg.payload.mass = float(payload_mass)
    if payload_offset is not None:
        robot_cfg.payload.offset = [float(x) for x in payload_offset]

    return SimConfig(
        simulation=sim_cfg,
        arena=arena_cfg,
        rendering=rend_cfg,
        robot=robot_cfg,
        default_robot=raw_default.get("default_robot", "robot_1_compact"),
        default_arena=arena or raw_default.get("default_arena", "auto"),
    )

