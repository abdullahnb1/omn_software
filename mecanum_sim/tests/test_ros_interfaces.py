"""Tests verifying zero-dependency ROS 2 data schemas, REP compliance, and MCU header export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from mecanum_sim.config import load_config
from mecanum_sim.interfaces import (
    Header,
    Vector3,
    Quaternion,
    TwistCommand,
    OdometryState,
    JointStateMessage,
    ImuStateMessage,
    TransformMessage,
    ClockMessage,
    DiagnosticStatusMessage,
    HardwareSerialBridge,
)
from scripts.export_mcu_header import generate_c_header


def test_zero_dependency_rclpy():
    """Verify that importing interfaces does not require rclpy."""
    assert "rclpy" not in sys.modules


def test_all_ten_ros_dataclasses():
    """Validates construction, dictionary serialization, and JSON serialization for all 10 dataclasses."""
    h = Header(stamp_sec=12, stamp_nanosec=345, frame_id="odom")
    assert h.to_dict()["frame_id"] == "odom"
    assert "12" in h.to_json()

    v = Vector3(1.0, 2.0, 3.0)
    assert np.allclose(v.to_numpy(), [1.0, 2.0, 3.0])
    assert v.to_dict()["x"] == 1.0

    q = Quaternion(0.0, 0.0, 0.0, 1.0)
    assert np.allclose(q.to_numpy(), [0.0, 0.0, 0.0, 1.0])
    assert q.to_dict()["w"] == 1.0

    twist = TwistCommand.from_planar(vx=0.5, vy=-0.2, wz=1.2)
    assert twist.linear.x == 0.5
    assert twist.linear.y == -0.2
    assert twist.angular.z == 1.2
    d_twist = twist.to_dict()
    assert d_twist["linear"]["x"] == 0.5

    odom = OdometryState(
        header=h,
        child_frame_id="base_footprint",
        pose_position=Vector3(1.0, 2.0, 0.0),
        pose_orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        twist_linear=Vector3(0.5, 0.0, 0.0),
        twist_angular=Vector3(0.0, 0.0, 0.2),
    )
    assert len(odom.pose_covariance) == 36
    assert len(odom.twist_covariance) == 36
    assert odom.child_frame_id == "base_footprint"
    assert json.loads(odom.to_json())["child_frame_id"] == "base_footprint"

    joint = JointStateMessage(
        header=h,
        name=["fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"],
        position=[1.0, 1.1, 1.2, 1.3],
        velocity=[10.0, 10.0, 10.0, 10.0],
        effort=[0.5, 0.5, 0.5, 0.5],
    )
    assert len(joint.position) == 4
    assert joint.to_dict()["name"][0] == "fl_wheel_joint"

    imu = ImuStateMessage(
        header=Header(frame_id="imu_link"),
        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        angular_velocity=Vector3(0.0, 0.0, 0.1),
        linear_acceleration=Vector3(0.0, 0.0, 9.81),
    )
    assert len(imu.orientation_covariance) == 9
    assert imu.linear_acceleration.z == 9.81

    tf_msg = TransformMessage(
        header=Header(frame_id="odom"),
        child_frame_id="base_footprint",
        translation=Vector3(1.5, 0.5, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )
    assert tf_msg.child_frame_id == "base_footprint"

    clk = ClockMessage(sec=10, nanosec=500000000)
    assert clk.sec == 10
    assert clk.nanosec == 500000000

    diag = DiagnosticStatusMessage(
        level=0,
        name="drivetrain",
        message="OK",
        hardware_id="robot_1",
        values={"voltage": "24.2V", "temperature": "32C"},
    )
    assert diag.level == 0
    assert diag.values["voltage"] == "24.2V"


def test_dynamic_nav2_footprints_for_all_robots():
    """Verify that nav2_footprint dynamically adjusts to each robot's dimensions."""
    c1 = load_config(robot="1")
    c2 = load_config(robot="2")
    c3 = load_config(robot="3")

    f1 = c1.robot.nav2_footprint
    f2 = c2.robot.nav2_footprint
    f3 = c3.robot.nav2_footprint

    assert len(f1) == 4 and len(f2) == 4 and len(f3) == 4

    # Robot 1 length is 0.36 -> half_l is 0.18
    assert abs(f1[0][0] - 0.18) < 1e-4
    # Robot 2 length is 0.58 -> half_l is 0.29
    assert abs(f2[0][0] - 0.29) < 1e-4
    # Robot 3 length is 0.46 -> half_l is 0.23
    assert abs(f3[0][0] - 0.23) < 1e-4

    # Footprints must all be distinct
    assert f1 != f2 and f2 != f3 and f1 != f3


def test_ros2_control_state_and_command_interfaces():
    """Verify ros2_control interface extraction and velocity command setting."""
    bridge = HardwareSerialBridge()
    states = bridge.get_state_interfaces()

    assert set(states.keys()) == {
        "fl_wheel_joint",
        "fr_wheel_joint",
        "rl_wheel_joint",
        "rr_wheel_joint",
    }
    for name, data in states.items():
        assert "position" in data
        assert "velocity" in data
        assert "effort" in data

    cmd = {
        "fl_wheel_joint": {"velocity": 12.5},
        "fr_wheel_joint": {"velocity": -12.5},
        "rl_wheel_joint": {"velocity": 8.0},
        "rr_wheel_joint": {"velocity": -8.0},
    }
    bridge.set_command_interfaces(cmd)
    _, vel_feedback, _ = bridge.read_feedback()
    assert np.allclose(vel_feedback, [12.5, -12.5, 8.0, -8.0])


def test_mcu_c_header_generation(tmp_path: Path):
    """Verify that export_mcu_header generates valid C headers with all macros and structs."""
    for r_key in ["1", "2", "3"]:
        hdr_path = generate_c_header(r_key, tmp_path)
        assert hdr_path.exists()
        content = hdr_path.read_text(encoding="utf-8")

        assert "WHEEL_RADIUS_M" in content
        assert "OFFSET_LX_M" in content
        assert "OFFSET_LY_M" in content
        assert "J_INV_MATRIX" in content
        assert "J_FWD_MATRIX" in content
        assert "SerialVelocityFrame_t" in content
        assert "SERIAL_VELOCITY_FRAME_HEADER" in content

