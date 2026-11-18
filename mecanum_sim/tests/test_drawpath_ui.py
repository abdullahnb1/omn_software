"""Tests verifying WaypointPlannerCore, robot footprint scaling, JSON export/import, and non-interactive execution."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from mecanum_sim.drawpath_ui import WaypointPlannerCore, run_draw_mode


def test_waypoint_planner_core_lifecycle(tmp_path: Path):
    """Verifies adding, deleting, clearing, and fitting cubic splines to waypoints."""
    planner = WaypointPlannerCore(robot="1", nominal_speed=0.45)
    assert len(planner.waypoints) == 1
    assert planner.waypoints[0] == {"x": 0.0, "y": 0.0, "theta": 0.0}

    # Add waypoints
    planner.add_waypoint(1.0, 0.5, 0.2)
    planner.add_waypoint(2.0, -0.5, -0.2)
    assert len(planner.waypoints) == 3

    # Generate spline
    spline = planner.generate_spline()
    assert spline is not None
    assert spline.total_duration > 2.0

    # Remove last waypoint
    removed = planner.remove_last_waypoint()
    assert removed is not None
    assert removed["x"] == 2.0
    assert len(planner.waypoints) == 2

    # Save to JSON and reload
    json_path = tmp_path / "planner_test.json"
    planner.save_to_json(json_path)
    assert json_path.exists()

    planner2 = WaypointPlannerCore(robot="2")
    planner2.load_from_json(json_path)
    assert len(planner2.waypoints) == 2
    assert np.isclose(planner2.nominal_speed, 0.45)


def test_robot_selector_footprint_scaling():
    """Verifies that selecting different robots scales the footprint polygon accordingly."""
    planner = WaypointPlannerCore(robot="1")
    poly1 = planner.get_robot_footprint_polygon(0.0, 0.0, 0.0)

    # Switch to Robot 2
    planner.select_robot("2")
    poly2 = planner.get_robot_footprint_polygon(0.0, 0.0, 0.0)

    # Switch to Robot 3
    planner.select_robot("3")
    poly3 = planner.get_robot_footprint_polygon(0.0, 0.0, 0.0)

    # Check that lengths match robot chassis lengths
    # Robot 1 length 0.36 -> half 0.18 -> poly max x is 0.18
    assert np.isclose(np.max(poly1[:, 0]), 0.18)
    # Robot 2 length 0.58 -> half 0.29 -> poly max x is 0.29
    assert np.isclose(np.max(poly2[:, 0]), 0.29)
    # Robot 3 length 0.46 -> half 0.23 -> poly max x is 0.23
    assert np.isclose(np.max(poly3[:, 0]), 0.23)


def test_draw_mode_non_interactive_execution(tmp_path: Path):
    """Verifies that run_draw_mode executes headlessly when non_interactive=True."""
    res = run_draw_mode(
        robot="1",
        preset="doorway_and_threshold",
        speed=0.40,
        non_interactive=True,
        output_dir=str(tmp_path),
    )
    assert res["status"] == "completed"
    assert not res["has_nan"]
    assert res["num_steps"] > 50


def test_waypoint_planner_core_arena_and_payload():
    """Verifies arena and payload setting on WaypointPlannerCore."""
    core = WaypointPlannerCore(robot="1", nominal_speed=0.35, arena="open", payload_mass=15.0)
    assert core.arena == "open"
    assert np.isclose(core.payload_mass, 15.0)

    core.set_arena("doorway")
    assert core.arena == "doorway"

    core.set_payload(25.5)
    assert np.isclose(core.payload_mass, 25.5)


def test_draw_mode_with_custom_waypoints(tmp_path: Path):
    """Verifies that custom drawn/saved waypoints execute instead of hardcoded default."""
    from mecanum_sim.trajectory import save_waypoints_json
    wps = [
        {"x": 0.0, "y": 0.0, "theta": 0.0},
        {"x": 1.0, "y": 0.5, "theta": 0.1},
        {"x": 2.0, "y": 0.0, "theta": 0.0},
    ]
    wp_file = tmp_path / "custom_drawn.json"
    save_waypoints_json(wps, wp_file, nominal_speed=0.40)

    res = run_draw_mode(
        robot="1",
        waypoints_file=str(wp_file),
        arena="doorway",
        payload_mass=10.0,
        speed=0.40,
        non_interactive=True,
        output_dir=str(tmp_path),
    )
    assert res["status"] == "completed"
    assert not res["has_nan"]
    assert res["num_steps"] > 20
    assert "drawn_path" in str(res["csv_path"])


def test_draw_mode_ramps_arena_and_high_speed(tmp_path: Path):
    """Verifies non-interactive execution with the ramps arena and increased speed limit."""
    res = run_draw_mode(
        robot="1",
        preset="forward_reverse_stress",
        arena="ramps",
        speed=1.80,
        non_interactive=True,
        output_dir=str(tmp_path),
    )
    assert res["status"] == "completed"
    assert not res["has_nan"]
    assert res["num_steps"] > 20



