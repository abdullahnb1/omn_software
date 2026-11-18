"""Tests verifying velocity controllers, acceleration slew limits, rate limiters, and serial frames."""

from __future__ import annotations

import numpy as np
import pytest

from mecanum_sim.config import load_config
from mecanum_sim.controller import ChassisRateLimiter, WheelVelocityController, MecanumController
from mecanum_sim.interfaces import HardwareSerialBridge


def test_wheel_velocity_controller_slew_and_saturation():
    """Verifies that WheelVelocityController respects max_wheel_alpha slew rate and max_wheel_speed."""
    max_w = 30.0
    max_alpha = 50.0  # rad/s^2
    ctrl = WheelVelocityController(
        max_wheel_speed=max_w,
        max_wheel_alpha=max_alpha,
        kp=0.05,
        ki=0.40,
        kd=0.0,
    )

    dt = 0.02  # 50 Hz -> max step delta = 50.0 * 0.02 = 1.0 rad/s
    cmd = ctrl.update(desired_w=20.0, measured_w=0.0, dt=dt)
    # After one dt from rest, w_slew should be at most 1.0 rad/s
    assert ctrl.w_slew <= 1.0 + 1e-6
    # Command should not exceed slew + small PD trim
    assert cmd <= 2.0

    # Step for 1 second (50 steps) where motor tracks w_slew
    meas_w = 0.0
    for _ in range(50):
        cmd = ctrl.update(desired_w=20.0, measured_w=meas_w, dt=dt)
        meas_w = ctrl.w_slew  # Motor tracks slewed reference
    # A few additional steps at steady state
    for _ in range(20):
        cmd = ctrl.update(desired_w=20.0, measured_w=20.0, dt=dt)
    assert np.isclose(ctrl.w_slew, 20.0, atol=1e-3)
    assert np.isclose(cmd, 20.0, atol=0.25)

    # Now request excessive speed 100.0 rad/s
    for _ in range(100):
        cmd = ctrl.update(desired_w=100.0, measured_w=30.0, dt=dt)
    assert ctrl.w_slew <= max_w + 1e-6
    assert cmd <= max_w + 1e-6


def test_wheel_velocity_controller_anti_windup():
    """Verifies that integral anti-windup prevents integrator runaway during saturation."""
    max_w = 25.0
    ctrl = WheelVelocityController(
        max_wheel_speed=max_w,
        max_wheel_alpha=100.0,
        kp=0.1,
        ki=2.0,
    )
    dt = 0.01

    # Request 35 rad/s (exceeding 25 rad/s) while measured speed is stalled at 0
    for _ in range(500):  # 5 seconds of complete stall
        cmd = ctrl.update(desired_w=35.0, measured_w=0.0, dt=dt)

    assert cmd <= max_w + 1e-6
    # Integral term should be capped and not have grown to thousands
    assert ctrl.integral <= max_w * 0.5


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_chassis_rate_limiter_bounds(robot_key: str):
    """Verifies that ChassisRateLimiter enforces both max velocity and acceleration for each robot."""
    cfg = load_config(robot=robot_key)
    lim = cfg.robot.limits
    rate_limiter = ChassisRateLimiter(
        max_vx=lim.max_vx,
        max_vy=lim.max_vy,
        max_wz=lim.max_wz,
        max_ax=lim.max_ax,
        max_ay=lim.max_ay,
        max_awz=lim.max_awz,
    )

    dt = 0.01
    # Step request from zero to well above max limits
    step_cmd = [lim.max_vx * 2.0, lim.max_vy * 2.0, lim.max_wz * 2.0]
    out1 = rate_limiter.update(step_cmd, dt)

    # After 1 step of dt, change must not exceed max_accel * dt
    assert out1[0] <= lim.max_ax * dt + 1e-9
    assert out1[1] <= lim.max_ay * dt + 1e-9
    assert out1[2] <= lim.max_awz * dt + 1e-9

    # After many steps, output must saturate at max_v
    for _ in range(500):
        out_steady = rate_limiter.update(step_cmd, dt)
    assert np.isclose(out_steady[0], lim.max_vx)
    assert np.isclose(out_steady[1], lim.max_vy)
    assert np.isclose(out_steady[2], lim.max_wz)


def test_mecanum_controller_coordination():
    """Verifies complete two-layer pipeline in MecanumController."""
    cfg = load_config(robot="1")
    ctrl = MecanumController(cfg.robot)

    # Command twist [0.4, 0.2, 0.5]
    cmd_w, desat_w, lim_twist = ctrl.update(
        commanded_twist=[0.4, 0.2, 0.5],
        measured_wheel_speeds=[0.0, 0.0, 0.0, 0.0],
        dt=0.01,
    )

    assert len(cmd_w) == 4
    assert len(desat_w) == 4
    assert len(lim_twist) == 3
    assert np.all(np.abs(cmd_w) <= cfg.robot.actuator.max_speed)


def test_hardware_serial_bridge_packet_packing_and_corruption():
    """Verifies 20-byte binary velocity frame packing, CRC16 validation, and corruption detection."""
    bridge = HardwareSerialBridge()

    w_fl, w_fr, w_rl, w_rr = 12.34, -5.67, 8.90, -14.21
    frame = bridge.pack_velocities(w_fl, w_fr, w_rl, w_rr)

    assert len(frame) == 20
    assert frame[:2] == b"\x55\xAA"  # 0xAA55 little-endian is 0x55, 0xAA

    # Unpack and verify
    unpacked = bridge.unpack_velocities(frame)
    assert np.allclose(unpacked, [w_fl, w_fr, w_rl, w_rr], atol=1e-5)

    # Test corrupted payload byte
    corrupted_payload = bytearray(frame)
    corrupted_payload[6] ^= 0xFF
    with pytest.raises(ValueError, match="CRC-16 mismatch"):
        bridge.unpack_velocities(bytes(corrupted_payload))

    # Test corrupted header
    corrupted_header = bytearray(frame)
    corrupted_header[0] = 0x00
    with pytest.raises(ValueError, match="Invalid frame header"):
        bridge.unpack_velocities(bytes(corrupted_header))

    # Test invalid length
    with pytest.raises(ValueError, match="Invalid frame size"):
        bridge.unpack_velocities(frame[:15])
