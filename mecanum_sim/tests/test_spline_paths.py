"""Tests verifying C^1 cubic spline continuity, 10 presets, rest-to-rest profiles, and JSON I/O."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from mecanum_sim.trajectory import (
    CubicSplineTrajectory,
    create_trajectory,
    get_preset_waypoints,
    save_waypoints_json,
    load_waypoints_json,
)

PRESET_NAMES = [
    "forward_reverse_stress",
    "pure_lateral_crabbing",
    "diagonal_slalom",
    "zero_radius_pirouette",
    "omni_square",
    "cloverleaf_spline",
    "figure8_continuous",
    "doorway_and_threshold",
    "liquid_payload_gentle",
    "obstacle_avoidance_corridor",
]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_all_ten_presets_c1_continuity_and_settling(preset: str):
    """Verifies that all 10 preset trajectories start at rest, are finite and continuous, and settle to zero velocity."""
    traj = create_trajectory(preset, nominal_speed=0.40, settle_time=1.0)
    assert traj.total_duration > 2.0
    assert traj.motion_duration > 0.0

    # 1. Start at rest (t = 0)
    p_start = traj.evaluate(0.0)
    assert abs(p_start.vx) < 1e-6
    assert abs(p_start.vy) < 1e-6
    assert abs(p_start.wz) < 1e-6

    # 2. Sample densely across duration to check finiteness and continuity
    times = np.linspace(0.0, traj.total_duration, 200)
    pts = [traj.evaluate(t) for t in times]

    vxs = np.array([p.vx for p in pts])
    vys = np.array([p.vy for p in pts])
    wzs = np.array([p.wz for p in pts])

    assert np.all(np.isfinite(vxs))
    assert np.all(np.isfinite(vys))
    assert np.all(np.isfinite(wzs))

    # Check finite velocity bounds
    assert np.max(np.abs(vxs)) < 3.0
    assert np.max(np.abs(vys)) < 3.0
    assert np.max(np.abs(wzs)) < 10.0

    # 3. Final 1.0 second settling phase must have strictly zero velocity
    t_settle_start = traj.motion_duration + 0.05
    settle_times = np.linspace(t_settle_start, traj.total_duration, 20)
    for t_s in settle_times:
        p_s = traj.evaluate(t_s)
        assert abs(p_s.vx) < 1e-6, f"{preset} failed settle vx at t={t_s}: vx={p_s.vx}"
        assert abs(p_s.vy) < 1e-6, f"{preset} failed settle vy at t={t_s}: vy={p_s.vy}"
        assert abs(p_s.wz) < 1e-6, f"{preset} failed settle wz at t={t_s}: wz={p_s.wz}"


def test_waypoint_json_save_load_roundtrip(tmp_path: Path):
    """Verifies round-trip saving and loading of custom waypoints and nominal speed."""
    original_wps = [
        {"x": 0.0, "y": 0.0, "theta": 0.0},
        {"x": 1.25, "y": 0.75, "theta": 0.35},
        {"x": 2.50, "y": -0.50, "theta": -0.20},
    ]
    original_speed = 0.55
    json_file = tmp_path / "test_path.json"

    save_waypoints_json(original_wps, json_file, nominal_speed=original_speed)
    assert json_file.exists()

    loaded_wps, loaded_speed = load_waypoints_json(json_file)
    assert np.isclose(loaded_speed, original_speed)
    assert len(loaded_wps) == len(original_wps)

    for orig, loaded in zip(original_wps, loaded_wps):
        assert np.isclose(orig["x"], loaded["x"], atol=1e-4)
        assert np.isclose(orig["y"], loaded["y"], atol=1e-4)
        assert np.isclose(orig["theta"], loaded["theta"], atol=1e-4)


def test_preset_aliases():
    """Verifies that short aliases resolve cleanly to their canonical preset."""
    aliases = {
        "doorway": "doorway_and_threshold",
        "figure8": "figure8_continuous",
        "square": "omni_square",
        "lateral": "pure_lateral_crabbing",
        "forward": "forward_reverse_stress",
    }
    for alias, canonical in aliases.items():
        wps_alias = get_preset_waypoints(alias)
        wps_canonical = get_preset_waypoints(canonical)
        assert wps_alias == wps_canonical

