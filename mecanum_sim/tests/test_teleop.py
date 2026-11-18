"""Tests verifying headless teleoperation controller, driving modes, and key state transitions."""

from __future__ import annotations

import numpy as np
import pytest

from mecanum_sim.config import load_config
from mecanum_sim.viewer_gui import FreeTeleopController


def test_teleop_initial_state_and_mode_toggle():
    """Verifies default CONTINUOUS mode and edge-triggered 3-mode toggle: CONTINUOUS -> DIRECT -> CRUISE -> CONTINUOUS."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)

    assert teleop.mode == "CONTINUOUS"
    assert teleop.cmd_vx == 0.0
    assert teleop.cmd_vy == 0.0
    assert teleop.cmd_wz == 0.0

    # 1. CONTINUOUS -> DIRECT
    teleop.handle_key_down("m")
    assert teleop.mode == "DIRECT"

    # 2. DIRECT -> CRUISE
    teleop.handle_key_down("m")
    assert teleop.mode == "CRUISE"

    # 3. CRUISE -> CONTINUOUS
    teleop.handle_key_down("m")
    assert teleop.mode == "CONTINUOUS"


def test_direct_drive_acceleration_and_release_decay():
    """Verifies that in Direct mode holding keys accelerates, and releasing decays smoothly to zero."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)
    teleop.mode = "DIRECT"
    dt = 0.05

    # Hold 'up' key for 1 second (20 steps)
    teleop.handle_key_down("up")
    for _ in range(20):
        twist = teleop.update(dt)

    assert twist[0] > 0.4
    assert twist[1] == 0.0
    assert twist[2] == 0.0

    # Release 'up' key
    teleop.handle_key_up("up")
    # Step until decayed
    for _ in range(30):
        twist = teleop.update(dt)

    assert twist[0] < 0.05
    for _ in range(20):
        twist = teleop.update(dt)
    assert twist[0] == 0.0


def test_cruise_mode_latching():
    """Verifies that in Cruise mode, releasing keys maintains/latches the current setpoint."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)
    teleop.mode = "CRUISE"
    dt = 0.05

    teleop.handle_key_down("up")
    for _ in range(10):
        teleop.update(dt)
    v_latched = teleop.cmd_vx
    assert v_latched > 0.2

    # Release 'up' key: in CRUISE, velocity must remain latched
    teleop.handle_key_up("up")
    for _ in range(20):
        twist = teleop.update(dt)

    assert np.isclose(twist[0], v_latched)


def test_emergency_brake_and_camera_cycle():
    """Verifies instant velocity zeroing via emergency brake keys ('b', 'x') and camera cycling."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)

    teleop.cmd_vx = 0.6
    teleop.cmd_vy = -0.4
    teleop.cmd_wz = 1.0

    # Press 'B' for brake
    teleop.handle_key_down("b")
    assert teleop.cmd_vx == 0.0
    assert teleop.cmd_vy == 0.0
    assert teleop.cmd_wz == 0.0

    # Camera cycling
    cam0 = teleop.active_camera_idx
    teleop.handle_key_down("v")
    assert teleop.active_camera_idx == (cam0 + 1) % 3


def test_keypad_controls_and_wasd_rejection():
    """Verifies that Keypad keys 8456 and 79 drive the robot, while WASD and QE are ignored."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)
    dt = 0.05

    # 1. WASD and QE must be rejected (no motion)
    for key in ["w", "a", "s", "d", "q", "e"]:
        teleop.handle_key_down(key)
        teleop.handle_viewer_key(key)
        twist = teleop.update(dt)
        assert np.allclose(twist, [0.0, 0.0, 0.0]), f"Key '{key}' should not cause motion!"

    # 2. Keypad 8 (forward)
    teleop.handle_key_down("8")
    for _ in range(10):
        twist = teleop.update(dt)
    assert twist[0] > 0.2
    assert twist[1] == 0.0
    teleop.handle_key_up("8")

    # 3. Keypad 4 (strafe left)
    teleop.handle_key_down("4")
    for _ in range(10):
        twist = teleop.update(dt)
    assert twist[1] > 0.2
    teleop.handle_key_up("4")

    # 4. Keypad 7 (rotate CCW)
    teleop.handle_key_down("7")
    for _ in range(10):
        twist = teleop.update(dt)
    assert twist[2] > 0.2
    teleop.handle_key_up("7")


def test_continuous_mode_hold_and_release():
    """Verifies CONTINUOUS mode press-and-hold behavior and release decay."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)
    assert teleop.mode == "CONTINUOUS"

    dt = 0.05
    # Hold 'up'
    teleop.handle_key_down("up")
    for _ in range(20):
        twist = teleop.update(dt)
    assert twist[0] > 0.4

    # Release 'up'
    teleop.handle_key_up("up")
    for _ in range(30):
        twist = teleop.update(dt)
    assert twist[0] < 0.05


def test_dynamic_speed_scaling_keys():
    """Verifies that '+' and '-' keys dynamically step speed ceiling between 10% and 200%."""
    cfg = load_config(robot="1")
    teleop = FreeTeleopController(cfg.robot)

    assert teleop.speed_scale == 1.0
    initial_max_vx = teleop.max_vx

    # 1. Step up with '+'
    teleop.handle_viewer_key("+")
    assert np.isclose(teleop.speed_scale, 1.1)

    teleop.handle_key_down("+")
    assert np.isclose(teleop.speed_scale, 1.2)

    # 2. Step up to max clamp (2.0)
    for _ in range(15):
        teleop.handle_key_down("+")
    assert np.isclose(teleop.speed_scale, 2.0)

    # Further '+' must not exceed 2.0
    teleop.handle_key_down("+")
    assert np.isclose(teleop.speed_scale, 2.0)

    # 3. Step down to min clamp (0.10)
    for _ in range(25):
        teleop.handle_key_down("-")
    assert np.isclose(teleop.speed_scale, 0.10)

    # Further '-' must not drop below 0.10
    teleop.handle_viewer_key("-")
    assert np.isclose(teleop.speed_scale, 0.10)

    # 4. Moving with low speed scale clamps commanded velocity to scaled max
    dt = 0.05
    teleop.handle_key_down("up")
    for _ in range(30):
        twist = teleop.update(dt)
    # At 10% speed scale, max vx is 0.10 * initial_max_vx
    assert np.isclose(twist[0], 0.10 * initial_max_vx, atol=1e-3)

    # 5. Stepping up while moving immediately increases available velocity ceiling
    for _ in range(5):
        teleop.handle_viewer_key("+")
    assert np.isclose(teleop.speed_scale, 0.60)
    for _ in range(30):
        twist = teleop.update(dt)
    assert np.isclose(twist[0], 0.60 * initial_max_vx, atol=1e-3)


