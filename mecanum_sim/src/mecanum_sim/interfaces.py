"""ROS 2 REP-103/REP-105 compliant dataclasses and actuator interfaces.

Zero-dependency pure-Python implementation providing full drop-in compatibility
with standard ROS 2 message types, ros2_control interfaces, and embedded serial bridges.
"""

from __future__ import annotations

import json
import struct
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ==============================================================================
# Pure Python ROS 2 Message Schemas
# ==============================================================================

@dataclass
class Header:
    stamp_sec: int = 0
    stamp_nanosec: int = 0
    frame_id: str = "odom"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=float)


@dataclass
class TwistCommand:
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_planar(cls, vx: float, vy: float, wz: float) -> TwistCommand:
        return cls(
            linear=Vector3(x=float(vx), y=float(vy), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=float(wz)),
        )


@dataclass
class OdometryState:
    header: Header = field(default_factory=Header)
    child_frame_id: str = "base_footprint"
    pose_position: Vector3 = field(default_factory=Vector3)
    pose_orientation: Quaternion = field(default_factory=Quaternion)
    twist_linear: Vector3 = field(default_factory=Vector3)
    twist_angular: Vector3 = field(default_factory=Vector3)
    pose_covariance: List[float] = field(default_factory=lambda: [0.0] * 36)
    twist_covariance: List[float] = field(default_factory=lambda: [0.0] * 36)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class JointStateMessage:
    header: Header = field(default_factory=Header)
    name: List[str] = field(default_factory=lambda: [
        "fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"
    ])
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    effort: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ImuStateMessage:
    header: Header = field(default_factory=Header)
    orientation: Quaternion = field(default_factory=Quaternion)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    linear_acceleration: Vector3 = field(default_factory=Vector3)
    orientation_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)
    angular_velocity_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)
    linear_acceleration_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class TransformMessage:
    header: Header = field(default_factory=Header)
    child_frame_id: str = "base_link"
    translation: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ClockMessage:
    sec: int = 0
    nanosec: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class DiagnosticStatusMessage:
    level: int = 0  # 0: OK, 1: WARN, 2: ERROR, 3: STALE
    name: str = "mecanum_drivetrain"
    message: str = "Nominal operation"
    hardware_id: str = "mecanum_amr"
    values: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ==============================================================================
# Abstract Actuator Interface & Implementations
# ==============================================================================

class ActuatorInterface(ABC):
    """Abstract velocity command and telemetry interface."""

    WHEEL_NAMES = [
        "fl_wheel_joint",
        "fr_wheel_joint",
        "rl_wheel_joint",
        "rr_wheel_joint",
    ]

    @abstractmethod
    def send_velocities(self, velocities: np.ndarray) -> None:
        """Sends target wheel angular velocities [w_fl, w_fr, w_rl, w_rr] in rad/s."""
        pass

    @abstractmethod
    def read_feedback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (positions_rad, velocities_rad_s, efforts_nm)."""
        pass

    def get_state_interfaces(self) -> Dict[str, Dict[str, float]]:
        """Emits ros2_control compatible state interfaces."""
        pos, vel, eff = self.read_feedback()
        return {
            name: {
                "position": float(pos[i]),
                "velocity": float(vel[i]),
                "effort": float(eff[i]),
            }
            for i, name in enumerate(self.WHEEL_NAMES)
        }

    def set_command_interfaces(self, commands: Dict[str, Dict[str, float]]) -> None:
        """Consumes ros2_control compatible velocity command interfaces."""
        v = np.zeros(4, dtype=float)
        for i, name in enumerate(self.WHEEL_NAMES):
            if name in commands and "velocity" in commands[name]:
                v[i] = float(commands[name]["velocity"])
        self.send_velocities(v)


class MuJoCoVelocityActuator(ActuatorInterface):
    """Dispatches wheel velocity commands directly to MuJoCo <velocity> servo actuators."""

    def __init__(
        self,
        model: Any,
        data: Any,
        actuator_indices: Optional[List[int]] = None,
        joint_indices: Optional[List[int]] = None,
    ):
        self.model = model
        self.data = data

        if actuator_indices is not None:
            self.actuator_indices = list(actuator_indices)
        else:
            # Query actuators by name if possible, or default to first 4
            self.actuator_indices = []
            for name in ["fl_actuator", "fr_actuator", "rl_actuator", "rr_actuator"]:
                try:
                    aid = model.actuator(name).id
                    self.actuator_indices.append(aid)
                except Exception:
                    pass
            if len(self.actuator_indices) != 4:
                self.actuator_indices = [0, 1, 2, 3]

        if joint_indices is not None:
            self.joint_indices = list(joint_indices)
        else:
            self.joint_indices = []
            for name in self.WHEEL_NAMES:
                try:
                    jid = model.joint(name).id
                    self.joint_indices.append(jid)
                except Exception:
                    pass
            if len(self.joint_indices) != 4:
                self.joint_indices = [0, 1, 2, 3]

        # Extract qpos and qvel addresses
        self.qpos_addrs = [int(self.model.jnt_qposadr[j]) for j in self.joint_indices]
        self.qvel_addrs = [int(self.model.jnt_dofadr[j]) for j in self.joint_indices]

    def send_velocities(self, velocities: np.ndarray) -> None:
        """Writes angular velocities in rad/s directly to data.ctrl."""
        for i, aid in enumerate(self.actuator_indices):
            self.data.ctrl[aid] = float(velocities[i])

    def read_feedback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reads wheel positions, velocities, and applied servo torques."""
        pos = np.array([self.data.qpos[adr] for adr in self.qpos_addrs], dtype=float)
        vel = np.array([self.data.qvel[adr] for adr in self.qvel_addrs], dtype=float)
        eff = np.array([self.data.actuator_force[aid] for aid in self.actuator_indices], dtype=float)
        return pos, vel, eff


class HardwareSerialBridge(ActuatorInterface):
    """Encapsulates 20-byte binary serial/CAN velocity frame packing with CRC-16."""

    FRAME_HEADER = 0xAA55
    FRAME_SIZE = 20  # 2 bytes header + 4 * 4 bytes float + 2 bytes CRC16

    def __init__(self) -> None:
        self._last_cmd = np.zeros(4, dtype=float)
        self._simulated_feedback_pos = np.zeros(4, dtype=float)
        self._simulated_feedback_vel = np.zeros(4, dtype=float)
        self._simulated_feedback_eff = np.zeros(4, dtype=float)

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """Computes standard CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @classmethod
    def pack_velocities(cls, w_fl: float, w_fr: float, w_rl: float, w_rr: float) -> bytes:
        """Packs 4 wheel velocities into a 20-byte binary frame with header and CRC16."""
        # Payload: 4 little-endian floats
        payload = struct.pack("<4f", float(w_fl), float(w_fr), float(w_rl), float(w_rr))
        header = struct.pack("<H", cls.FRAME_HEADER)
        data_to_checksum = header + payload
        crc = cls.compute_crc16(data_to_checksum)
        crc_bytes = struct.pack("<H", crc)
        return data_to_checksum + crc_bytes

    @classmethod
    def unpack_velocities(cls, frame: bytes) -> np.ndarray:
        """Validates header, size, and CRC16, returning unpacked np.ndarray([w_fl, w_fr, w_rl, w_rr])."""
        if len(frame) != cls.FRAME_SIZE:
            raise ValueError(f"Invalid frame size: {len(frame)} bytes, expected {cls.FRAME_SIZE}")

        header, = struct.unpack_from("<H", frame, 0)
        if header != cls.FRAME_HEADER:
            raise ValueError(f"Invalid frame header: 0x{header:04X}, expected 0x{cls.FRAME_HEADER:04X}")

        expected_crc, = struct.unpack_from("<H", frame, 18)
        actual_crc = cls.compute_crc16(frame[:18])
        if actual_crc != expected_crc:
            raise ValueError(f"CRC-16 mismatch: computed 0x{actual_crc:04X}, frame contained 0x{expected_crc:04X}")

        w_fl, w_fr, w_rl, w_rr = struct.unpack_from("<4f", frame, 2)
        return np.array([w_fl, w_fr, w_rl, w_rr], dtype=float)

    def send_velocities(self, velocities: np.ndarray) -> None:
        self._last_cmd = np.array(velocities, dtype=float)
        # Update mock feedback
        self._simulated_feedback_vel = self._last_cmd.copy()

    def read_feedback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._simulated_feedback_pos.copy(),
            self._simulated_feedback_vel.copy(),
            self._simulated_feedback_eff.copy(),
        )

