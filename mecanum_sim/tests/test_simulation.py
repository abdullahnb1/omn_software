"""Tests verifying multi-rate simulation, all 3 robots, offscreen video, CSV export, and odometry."""

from __future__ import annotations

from pathlib import Path
import mujoco
import numpy as np
import pytest

from mecanum_sim.simulation import MecanumSimulation


@pytest.mark.parametrize("robot_key", ["1", "2", "3"])
def test_simulation_run_all_three_robots(robot_key: str, tmp_path: Path):
    """Runs 2.0s headless path simulation for each robot, verifying stability and telemetry."""
    sim = MecanumSimulation(
        robot=robot_key,
        arena="open",
        mode="path",
        preset="forward_reverse_stress",
        duration=2.0,
        output_dir=str(tmp_path),
        gui=False,
    )
    result = sim.run()

    assert result["status"] == "completed"
    assert not result["has_nan"]
    assert result["num_steps"] >= 150
    assert result["csv_path"] is not None

    csv_file = Path(result["csv_path"])
    assert csv_file.exists()
    assert csv_file.stat().st_size > 1000

    # Verify metrics
    metrics = result["metrics"]
    assert "rmse_tracking_m" in metrics
    assert "final_odom_drift_m" in metrics
    assert metrics["rmse_tracking_m"] < 0.50


def test_offscreen_camera_rendering_and_video(tmp_path: Path):
    """Verifies that offscreen front_camera rendering and GIF generation succeed headlessly."""
    video_path = tmp_path / "test_anim.gif"
    sim = MecanumSimulation(
        robot="1",
        arena="open",
        mode="path",
        preset="pure_lateral_crabbing",
        duration=1.0,
        record_video=str(video_path),
        camera="front_camera",
        output_dir=str(tmp_path),
        gui=False,
    )
    res = sim.run()

    assert res["status"] == "completed"
    assert video_path.exists()
    assert video_path.stat().st_size > 5000


def test_ros2_adapter_step_and_messages():
    """Verifies ROS 2 interface adapters in simulation."""
    from mecanum_sim.interfaces import TwistCommand

    sim = MecanumSimulation(robot="1", arena="open", mode="free", gui=False)
    # Settle
    for _ in range(20):
        sim.step()

    twist_cmd = TwistCommand.from_planar(vx=0.3, vy=0.0, wz=0.0)
    snap = sim.step_with_twist(twist_cmd)
    assert "cmd_vx" in snap

    # Test ROS messages extraction
    odom_msg = sim.get_ros_odometry()
    assert odom_msg.child_frame_id == "base_footprint"

    joint_msg = sim.get_ros_joint_states()
    assert len(joint_msg.position) == 4

    imu_msg = sim.get_ros_imu()
    assert imu_msg.header.frame_id == "imu_link"

    tfs = sim.get_ros_transforms()
    assert len(tfs) == 2

