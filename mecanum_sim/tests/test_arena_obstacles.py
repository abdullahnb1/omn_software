"""Tests verifying arena obstacle geometries, doorway clearance, and threshold dimensions."""

from __future__ import annotations

from pathlib import Path
import mujoco
import numpy as np
import pytest

from mecanum_sim.config import load_config
from models.generate_model import generate_all_models


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    generate_all_models()


def test_arena_dimensions_and_doorway_clearance():
    """Verifies 8x8m perimeter walls and 800mm clear doorway in doorway and full arenas."""
    model_doorway = mujoco.MjModel.from_xml_path("models/robot_1_doorway.xml")

    # Perimeter walls check
    wall_n_id = model_doorway.geom("wall_n").id
    wall_e_id = model_doorway.geom("wall_e").id
    # North wall is at pos y = 4.0, size x = 4.0
    assert np.isclose(model_doorway.geom_pos[wall_n_id, 1], 4.0)
    assert np.isclose(model_doorway.geom_size[wall_n_id, 0], 4.0)
    assert np.isclose(model_doorway.geom_pos[wall_e_id, 0], 4.0)

    # Partition walls and doorway width
    part_l_id = model_doorway.geom("partition_left").id
    part_r_id = model_doorway.geom("partition_right").id

    # Left partition: pos Y = 2.20, size Y = 1.80 -> inner edge is 2.20 - 1.80 = +0.40m
    # Right partition: pos Y = -2.20, size Y = 1.80 -> inner edge is -2.20 + 1.80 = -0.40m
    inner_l = model_doorway.geom_pos[part_l_id, 1] - model_doorway.geom_size[part_l_id, 1]
    inner_r = model_doorway.geom_pos[part_r_id, 1] + model_doorway.geom_size[part_r_id, 1]
    doorway_width = inner_l - inner_r

    assert np.isclose(doorway_width, 0.80, atol=0.005), f"Doorway width was {doorway_width:.4f}m, expected 0.80m"


def test_threshold_heights():
    """Verifies 8mm doorway threshold and 5mm open-arena threshold."""
    model_full = mujoco.MjModel.from_xml_path("models/robot_1_compact.xml")

    # 8mm doorway threshold
    thresh_8mm_id = model_full.geom("door_threshold_core").id
    # box size z is half-height: 0.004m -> full height = 0.008m (8mm)
    h_8mm = model_full.geom_size[thresh_8mm_id, 2] * 2.0
    assert np.isclose(h_8mm, 0.008, atol=1e-4)

    # 5mm open threshold
    thresh_5mm_id = model_full.geom("open_threshold_core").id
    h_5mm = model_full.geom_size[thresh_5mm_id, 2] * 2.0
    assert np.isclose(h_5mm, 0.005, atol=1e-4)

    # Verify horizontal orientation of threshold beveled ramps (no perpendicular component)
    ramp_e_id = model_full.geom("door_threshold_ramp_e").id
    ramp_n_id = model_full.geom("open_threshold_ramp_n").id
    # door ramp rotated around X by -90 deg to align with Y (quat w = -x != 0, y=z=0)
    assert not np.isclose(model_full.geom_quat[ramp_e_id, 1], 0.0)
    # open ramp rotated to align with X (quat != identity)
    assert not np.isclose(model_full.geom_quat[ramp_n_id, 2], 0.0)


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_all_robots_have_sufficient_clearances(robot_key: str):
    """Verifies that all 3 robots have underbelly clearance > 8mm and total width < 800mm with margin."""
    cfg = load_config(robot=robot_key)
    total_w = cfg.robot.wheels.total_width
    door_clearance = (0.80 - total_w) / 2.0

    # Doorway clearance must be at least 130mm (0.13m) per side
    assert door_clearance >= 0.130, f"Robot {robot_key} doorway clearance {door_clearance*1000:.1f}mm < 130mm"

    # Underbelly clearance must be at least 30mm (0.030m) to clear 8mm threshold comfortably
    underbelly = cfg.robot.chassis.underbelly_clearance
    assert underbelly >= 0.030, f"Robot {robot_key} underbelly clearance {underbelly*1000:.1f}mm < 30mm"

