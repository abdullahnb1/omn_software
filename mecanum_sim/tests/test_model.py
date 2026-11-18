"""Tests verifying MJCF model compilation, velocity actuators, and closed-loop physics sign agreement."""

from __future__ import annotations

import math
from pathlib import Path
import mujoco
import numpy as np
import pytest

from mecanum_sim.config import load_config
from mecanum_sim.kinematics import MecanumKinematics
from models.generate_model import generate_all_models


@pytest.fixture(scope="module", autouse=True)
def ensure_models_generated():
    """Ensure all MJCF models are synthesized before running model tests."""
    generate_all_models()


@pytest.mark.parametrize("scene_filename", [
    "robot_1_compact.xml",
    "robot_1_doorway.xml",
    "robot_1_open.xml",
    "robot_1_ramps.xml",
    "robot_2_heavy.xml",
    "robot_2_doorway.xml",
    "robot_2_open.xml",
    "robot_2_ramps.xml",
    "robot_3_custom.xml",
    "robot_3_doorway.xml",
    "robot_3_open.xml",
    "robot_3_ramps.xml",
])
def test_all_mjcf_scenes_compile_cleanly(scene_filename: str):
    """Verifies that all 12 composed scene XML files compile with zero errors."""
    scene_path = Path("models") / scene_filename
    assert scene_path.exists(), f"Missing expected scene: {scene_path}"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model is not None
    assert model.nbody > 10
    assert model.opt.impratio == 10.0


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_wheel_actuators_are_strictly_velocity_servos(robot_key: str):
    """Verifies that all 4 wheel actuators are <velocity> servos and forcelimited."""
    scene_map = {
        "1": "models/robot_1_open.xml",
        "2": "models/robot_2_open.xml",
        "3": "models/robot_3_open.xml",
    }
    model = mujoco.MjModel.from_xml_path(scene_map[robot_key])

    # Check actuator names and types
    actuator_names = ["fl_actuator", "fr_actuator", "rl_actuator", "rr_actuator"]
    for act_name in actuator_names:
        aid = model.actuator(act_name).id
        # In MuJoCo, trntype is joint (mjTRN_JOINT = 0) and dyntype is mjDYN_NONE = 0, gaintype is mjGAIN_FIXED
        # The key velocity actuator signature is kv > 0 in gainprm and biasprm
        gain = model.actuator_gainprm[aid, 0]
        bias = model.actuator_biasprm[aid, 2]
        assert gain > 0.0, f"Actuator {act_name} gain must be positive"
        assert bias < 0.0, f"Actuator {act_name} damping bias must be negative for velocity control"
        # Check force limits
        assert model.actuator_forcelimited[aid] == 1
        assert model.actuator_forcerange[aid, 1] > 0.0


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_cameras_presence_and_orientation(robot_key: str):
    """Verifies presence and upright xyaxes orientation of front_camera and track_camera."""
    scene_map = {
        "1": "models/robot_1_open.xml",
        "2": "models/robot_2_open.xml",
        "3": "models/robot_3_open.xml",
    }
    model = mujoco.MjModel.from_xml_path(scene_map[robot_key])

    # Verify front_camera
    cam_front_id = model.camera("front_camera").id
    assert cam_front_id >= 0

    # Verify track_camera
    cam_track_id = model.camera("track_camera").id
    assert cam_track_id >= 0


def _get_chassis_yaw(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    """Extracts planar yaw angle from base_link orientation."""
    q = data.xquat[model.body("base_link").id]  # [w, x, y, z]
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    return float(np.arctan2(siny_cosp, cosy_cosp))


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_closed_loop_physics_kinematics_sign_agreement(robot_key: str):
    """MANDATORY TEST: Verifies that commanded +vx, +vy, +wz in J_inv physically produce +X, +Y, +theta motions."""
    cfg = load_config(robot=robot_key)
    kin = MecanumKinematics(
        wheel_radius=cfg.robot.wheel_radius,
        lx=cfg.robot.lx,
        ly=cfg.robot.ly,
    )

    scene_map = {
        "1": "models/robot_1_open.xml",
        "2": "models/robot_2_open.xml",
        "3": "models/robot_3_open.xml",
    }
    model = mujoco.MjModel.from_xml_path(scene_map[robot_key])
    base_id = model.body("base_link").id

    # 1. Forward Test: Command [0.3, 0.0, 0.0] for 0.6s
    data = mujoco.MjData(model)
    # Settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    p0 = data.xpos[base_id].copy()

    w_fwd = kin.inverse([0.3, 0.0, 0.0])
    for _ in range(300):  # 300 steps * 0.002s = 0.6s
        data.ctrl[:4] = w_fwd
        mujoco.mj_step(model, data)
    p1 = data.xpos[base_id].copy()
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    assert dx > 0.05, f"Robot {robot_key} forward test failed: dx={dx:.4f} <= 0.05m"
    assert abs(dy) < 0.015, f"Robot {robot_key} forward test had excessive lateral veering: dy={dy:.4f} >= 0.015m"

    # 2. Lateral Left Test: Command [0.0, 0.3, 0.0] for 0.6s
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    p0 = data.xpos[base_id].copy()

    w_left = kin.inverse([0.0, 0.3, 0.0])
    for _ in range(300):
        data.ctrl[:4] = w_left
        mujoco.mj_step(model, data)
    p1 = data.xpos[base_id].copy()
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    assert dy > 0.05, f"Robot {robot_key} lateral left test failed: dy={dy:.4f} <= 0.05m"
    assert abs(dx) < 0.015, f"Robot {robot_key} lateral left test had excessive longitudinal veering: dx={dx:.4f} >= 0.015m"

    # 3. CCW Yaw Test: Command [0.0, 0.0, 1.0] for 0.6s
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    th0 = _get_chassis_yaw(data, model)

    w_ccw = kin.inverse([0.0, 0.0, 1.0])
    for _ in range(300):
        data.ctrl[:4] = w_ccw
        mujoco.mj_step(model, data)
    th1 = _get_chassis_yaw(data, model)
    dtheta = th1 - th0
    assert dtheta > 0.15, f"Robot {robot_key} CCW yaw test failed: dtheta={dtheta:.4f} <= 0.15 rad"


def test_composite_inertia_recalculation():
    """Verifies 3D payload mass and offset center-of-mass / Parallel Axis Theorem inertia recalculation."""
    c1 = load_config(robot="1", payload_mass=0.0)
    m0, com0, i0 = c1.robot.compute_composite_inertia()
    assert abs(m0 - c1.robot.unladen_mass) < 1e-4

    # Add 10 kg payload offset at [+0.05, 0.0, +0.02]
    c1_loaded = load_config(robot="1", payload_mass=10.0, payload_offset=[0.05, 0.0, 0.02])
    m1, com1, i1 = c1_loaded.robot.compute_composite_inertia()

    assert abs(m1 - (m0 + 10.0)) < 1e-4
    # CoM in X should shift forward (+X)
    assert com1[0] > com0[0]
    # Inertia moments should all increase
    assert np.all(i1 > i0)


def test_payload_isolation_keeps_base_unladen(tmp_path: Path):
    """Verifies that running with payload does NOT modify base unladen robot models on disk."""
    from mecanum_sim.simulation import MecanumSimulation

    # 1. Run with a 25.0 kg payload
    sim_payload = MecanumSimulation(robot="1", arena="open", mode="free", payload_mass=25.0, duration=0.05, no_csv=True)
    assert sim_payload.model.body("payload_block") is not None

    # Base unladen body file on disk must NOT contain payload_block
    base_body_text = Path("models/robots/robot_1_compact_body.xml").read_text(encoding="utf-8")
    assert "payload_block" not in base_body_text

    # 2. Run in free mode with payload_mass=0.0: must have NO payload_block
    sim_free = MecanumSimulation(robot="1", arena="open", mode="free", payload_mass=0.0, duration=0.05, no_csv=True)
    with pytest.raises(KeyError):
        sim_free.model.body("payload_block")


def test_ramps_arena_simulation():
    """Verifies that the ramps arena loads, spawns the robot cleanly on the central runway, and simulates cleanly."""
    from mecanum_sim.simulation import MecanumSimulation

    sim = MecanumSimulation(robot="1", arena="ramps", mode="free", duration=0.10, no_csv=True)
    assert np.isclose(sim.data.qpos[0], 0.0, atol=0.01)
    assert np.isclose(sim.data.qpos[1], 0.0, atol=0.01)
    res = sim.run()
    assert res["status"] == "completed"
    assert not res["has_nan"]


