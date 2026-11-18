"""Tests for parametric kinematics, desaturation, and SE(2) dead-reckoned odometry."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mecanum_sim.config import load_config
from mecanum_sim.kinematics import MecanumKinematics, WheelOdometryEstimator


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_kinematics_roundtrip_all_robots(robot_key: str):
    """Asserts that J_fwd @ J_inv is the 3x3 identity matrix for all 3 robots."""
    cfg = load_config(robot=robot_key)
    kin = MecanumKinematics(
        wheel_radius=cfg.robot.wheel_radius,
        lx=cfg.robot.lx,
        ly=cfg.robot.ly,
    )
    product = kin.J_fwd @ kin.J_inv
    identity = np.eye(3)
    assert np.allclose(product, identity, atol=1e-12)

    # Test round trip on arbitrary 3D twists
    test_twists = [
        [0.4, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 1.2],
        [0.3, -0.2, 0.5],
        [-0.5, 0.4, -1.0],
    ]
    for twist in test_twists:
        w = kin.inverse(twist)
        twist_rec = kin.forward(w)
        assert np.allclose(twist, twist_rec, atol=1e-12)


def test_wheel_speed_sign_patterns():
    """Verifies standard holonomic sign patterns for canonical motions."""
    kin = MecanumKinematics(wheel_radius=0.05, lx=0.12, ly=0.12)

    # 1. Pure forward (+X): FL>0, FR>0, RL>0, RR>0 (++++)
    w_fwd = kin.inverse([0.3, 0.0, 0.0])
    assert np.all(w_fwd > 0)
    assert np.allclose(w_fwd, [6.0, 6.0, 6.0, 6.0])

    # 2. Pure reverse (-X): FL<0, FR<0, RL<0, RR<0 (----)
    w_rev = kin.inverse([-0.3, 0.0, 0.0])
    assert np.all(w_rev < 0)
    assert np.allclose(w_rev, [-6.0, -6.0, -6.0, -6.0])

    # 3. Pure lateral left (+Y): FL<0, FR>0, RL>0, RR<0 (-++-)
    w_left = kin.inverse([0.0, 0.3, 0.0])
    assert w_left[0] < 0 and w_left[1] > 0 and w_left[2] > 0 and w_left[3] < 0
    assert np.allclose(w_left, [-6.0, 6.0, 6.0, -6.0])

    # 4. Pure lateral right (-Y): FL>0, FR<0, RL<0, RR>0 (+--+)
    w_right = kin.inverse([0.0, -0.3, 0.0])
    assert w_right[0] > 0 and w_right[1] < 0 and w_right[2] < 0 and w_right[3] > 0
    assert np.allclose(w_right, [6.0, -6.0, -6.0, 6.0])

    # 5. Pure CCW Yaw (+wz): FL<0, FR>0, RL<0, RR>0 (-+-+)
    w_ccw = kin.inverse([0.0, 0.0, 1.0])
    assert w_ccw[0] < 0 and w_ccw[1] > 0 and w_ccw[2] < 0 and w_ccw[3] > 0
    assert np.allclose(w_ccw, [-4.8, 4.8, -4.8, 4.8])

    # 6. Pure CW Yaw (-wz): FL>0, FR<0, RL>0, RR<0 (+-+-)
    w_cw = kin.inverse([0.0, 0.0, -1.0])
    assert w_cw[0] > 0 and w_cw[1] < 0 and w_cw[2] > 0 and w_cw[3] < 0
    assert np.allclose(w_cw, [4.8, -4.8, 4.8, -4.8])

    # 7. 45 deg Forward-Left diagonal: FL=0, FR>0, RL>0, RR=0
    w_diag_fl = kin.inverse([0.3, 0.3, 0.0])
    assert abs(w_diag_fl[0]) < 1e-12 and abs(w_diag_fl[3]) < 1e-12
    assert w_diag_fl[1] > 0 and w_diag_fl[2] > 0

    # 8. 45 deg Forward-Right diagonal: FL>0, FR=0, RL=0, RR>0
    w_diag_fr = kin.inverse([0.3, -0.3, 0.0])
    assert abs(w_diag_fr[1]) < 1e-12 and abs(w_diag_fr[2]) < 1e-12
    assert w_diag_fr[0] > 0 and w_diag_fr[3] > 0


def test_direction_preserving_wheel_velocity_desaturation():
    """Verifies that desaturate_wheel_speeds uniformly scales wheel velocities to preserve heading."""
    kin = MecanumKinematics(wheel_radius=0.05, lx=0.12, ly=0.12)
    max_w = 30.0

    # Command requesting large twist exceeding wheel limits
    requested_twist = np.array([1.2, 0.9, 2.5])
    raw_w = kin.inverse(requested_twist)
    assert np.max(np.abs(raw_w)) > max_w

    desat_w = kin.desaturate_wheel_speeds(raw_w, max_wheel_speed=max_w)
    assert np.max(np.abs(desat_w)) <= max_w + 1e-12
    assert np.isclose(np.max(np.abs(desat_w)), max_w)

    # Forward project desaturated wheel speeds
    resulting_twist = kin.forward(desat_w)

    # Check that ratios between vx, vy, and wz are strictly preserved
    scale = resulting_twist[0] / requested_twist[0]
    assert scale < 1.0
    assert np.allclose(resulting_twist, requested_twist * scale, atol=1e-12)


def test_wheel_odometry_estimator_integration():
    """Verifies SE(2) dead-reckoned odometry integration on synthetic motions."""
    kin = MecanumKinematics(wheel_radius=0.05, lx=0.12, ly=0.12)
    odom = WheelOdometryEstimator(kin, initial_pose=(0.0, 0.0, 0.0))

    dt = 0.01  # 100 Hz
    steps = 100  # 1.0 second total

    # 1. Move forward at 0.5 m/s for 1 second
    w_fwd = kin.inverse([0.5, 0.0, 0.0])
    for _ in range(steps):
        odom.update(w_fwd, dt)
    x, y, th = odom.pose
    assert np.isclose(x, 0.5, atol=1e-3)
    assert np.isclose(y, 0.0, atol=1e-3)
    assert np.isclose(th, 0.0, atol=1e-3)

    # 2. Pure rotation CCW at pi/2 rad/s for 1 second
    w_rot = kin.inverse([0.0, 0.0, math.pi / 2.0])
    for _ in range(steps):
        odom.update(w_rot, dt)
    x, y, th = odom.pose
    assert np.isclose(x, 0.5, atol=1e-3)
    assert np.isclose(y, 0.0, atol=1e-3)
    assert np.isclose(th, math.pi / 2.0, atol=1e-3)

    # 3. Now move forward in body frame at 0.5 m/s for 1 second (should move along world +Y!)
    for _ in range(steps):
        odom.update(w_fwd, dt)
    x, y, th = odom.pose
    assert np.isclose(x, 0.5, atol=1e-3)
    assert np.isclose(y, 0.5, atol=1e-3)
    assert np.isclose(th, math.pi / 2.0, atol=1e-3)

    # 4. Reset
    odom.reset((1.0, 2.0, 0.5))
    assert odom.pose == (1.0, 2.0, 0.5)

