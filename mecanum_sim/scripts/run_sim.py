#!/usr/bin/env python3
"""Master canonical CLI entry point for mecanum_sim platform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src and project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from mecanum_sim.config import load_config, list_available_robots
from mecanum_sim.drawpath_ui import run_draw_mode
from mecanum_sim.simulation import MecanumSimulation
try:
    from scripts.export_mcu_header import generate_c_header
    from scripts.generate_plots import generate_single_robot_dashboard, generate_cross_robot_comparison
except ModuleNotFoundError:
    from export_mcu_header import generate_c_header
    from generate_plots import generate_single_robot_dashboard, generate_cross_robot_comparison


def run_single_robot(args: argparse.Namespace, robot_key: str) -> Optional[Path]:
    """Runs simulation for a single robot profile and generates telemetry and plots."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_mcu_header:
        generate_c_header(robot_key, out_dir)

    if args.mode == "draw":
        res = run_draw_mode(
            robot=robot_key,
            preset=args.preset,
            waypoints_file=args.waypoints_file,
            speed=args.speed or 0.40,
            arena=args.arena,
            payload_mass=args.payload_mass,
            payload_offset=args.payload_offset,
            floor_friction=args.floor_friction,
            pose_correction=not args.no_pose_correction,
            no_front_cam=args.no_front_cam,
            record_video=args.record_video,
            camera=args.camera,
            no_csv=args.no_csv,
            non_interactive=args.non_interactive,
            output_dir=str(out_dir),
        )
        csv_path = Path(res["csv_path"]) if res.get("csv_path") else None
    else:
        sim = MecanumSimulation(
            robot=robot_key,
            arena=args.arena,
            mode=args.mode,
            preset=args.preset,
            waypoints_file=args.waypoints_file,
            speed=args.speed,
            duration=args.duration,
            payload_mass=args.payload_mass,
            payload_offset=args.payload_offset,
            floor_friction=args.floor_friction,
            pose_correction=not args.no_pose_correction,
            gui=args.gui,
            record_video=args.record_video,
            camera=args.camera,
            output_dir=str(out_dir),
            no_csv=args.no_csv,
            non_interactive=args.non_interactive,
        )
        res = sim.run()
        csv_path = Path(res["csv_path"]) if res.get("csv_path") else None

    # Automatically generate single-robot plots
    if csv_path and csv_path.exists() and not args.no_plot:
        generate_single_robot_dashboard(csv_path, out_dir)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mecanum_sim: Model-agnostic 4-wheel mecanum AMR simulation platform in MuJoCo."
    )
    parser.add_argument(
        "--robot",
        default="1",
        help="Robot identifier ('1', '2', '3', 'all', 'compact', 'heavy', 'custom'). Default: '1'.",
    )
    parser.add_argument(
        "--arena",
        default="auto",
        choices=["auto", "open", "doorway", "full", "ramps"],
        help="Arena environment. Default: 'auto'.",
    )
    parser.add_argument(
        "--mode",
        default="free",
        choices=["free", "path", "draw"],
        help="Operation mode ('free', 'path', 'draw'). Default: 'free'.",
    )
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D viewer.")
    parser.add_argument("--no-front-cam", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preset", default=None, help="Predefined trajectory preset name.")
    parser.add_argument("--waypoints-file", default=None, help="Path to JSON waypoints file.")
    parser.add_argument("--speed", type=float, default=None, help="Nominal cruising speed in m/s.")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration override in seconds.")
    parser.add_argument("--payload-mass", type=float, default=0.0, help="Payload mass in kg (default: 0.0).")
    parser.add_argument(
        "--payload-offset",
        type=float,
        nargs=3,
        default=None,
        help="Payload offset x y z relative to deck in meters.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.8, help="Floor friction coefficient (default: 0.8).")
    parser.add_argument("--no-pose-correction", action="store_true", help="Disable outer-loop pose correction.")
    parser.add_argument("--record-video", default=None, help="Path to save offscreen rendered animation (.gif/.mp4).")
    parser.add_argument(
        "--camera",
        default="track_camera",
        choices=["track_camera", "front_camera"],
        help="Camera for offscreen video. Default: 'track_camera'.",
    )
    parser.add_argument("--export-mcu-header", action="store_true", help="Export C/C++ firmware configuration header.")
    parser.add_argument("--output-dir", default="results", help="Directory for logs and plots. Default: 'results'.")
    parser.add_argument("--no-plot", action="store_true", help="Suppress automatic plot generation.")
    parser.add_argument("--no-csv", action="store_true", help="Suppress telemetry CSV export.")
    parser.add_argument("--non-interactive", action="store_true", help="Run draw mode headlessly.")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_robot = str(args.robot).lower()
    if target_robot == "all":
        # Execute all 3 robots sequentially
        robots = ["1", "2", "3"]
        csv_paths: List[Path] = []
        for r_key in robots:
            p = run_single_robot(args, r_key)
            if p and p.exists():
                csv_paths.append(p)

        # Generate comparative overlay dashboard
        if len(csv_paths) >= 2 and not args.no_plot:
            scenario = args.preset or args.mode
            generate_cross_robot_comparison(csv_paths, scenario, out_dir)
    else:
        run_single_robot(args, target_robot)


if __name__ == "__main__":
    main()

