#!/usr/bin/env python3
"""Automated batch benchmark & cross-robot comparison runner across all 10 presets and all 3 robots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add src and project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from mecanum_sim.simulation import MecanumSimulation
from mecanum_sim.trajectory import get_preset_waypoints
try:
    from scripts.generate_plots import generate_single_robot_dashboard, generate_cross_robot_comparison
except ModuleNotFoundError:
    from generate_plots import generate_single_robot_dashboard, generate_cross_robot_comparison

ALL_PRESETS = [
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


def run_all(duration: float = 4.0, output_dir: str = "results/batch", no_plot: bool = False) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robots = ["1", "2", "3"]

    print("=" * 70)
    print("   mecanum_sim: Running Automated Multi-Robot Benchmark Suite")
    print(f"   Duration per scenario: {duration:.1f}s | Output: {out_dir}")
    print("=" * 70)

    total_runs = len(ALL_PRESETS) * len(robots)
    completed = 0
    diverged = 0

    for preset in ALL_PRESETS:
        print(f"\n>>> Running Scenario: {preset} <<<")
        csv_paths: List[Path] = []

        for r_key in robots:
            sim = MecanumSimulation(
                robot=r_key,
                arena="auto",
                mode="path",
                preset=preset,
                duration=duration,
                output_dir=str(out_dir),
                gui=False,
            )
            res = sim.run()

            if res["has_nan"] or res["status"] != "completed":
                print(f"  [-] Robot {r_key} FAILED/DIVERGED!")
                diverged += 1
            else:
                completed += 1
                rmse = res["metrics"].get("rmse_tracking_m", 0.0)
                drift = res["metrics"].get("final_odom_drift_m", 0.0)
                print(f"  [+] Robot {r_key} ({sim.robot.name}) PASSED: RMSE={rmse:.4f}m, Drift={drift:.4f}m")

            if res.get("csv_path"):
                p = Path(res["csv_path"])
                if p.exists():
                    csv_paths.append(p)
                    if not no_plot:
                        generate_single_robot_dashboard(p, out_dir)

        if len(csv_paths) >= 2 and not no_plot:
            generate_cross_robot_comparison(csv_paths, preset, out_dir)

    print("\n" + "=" * 70)
    print(f"BATCH SUMMARY: Completed {completed}/{total_runs} runs successfully. (Diverged: {diverged})")
    print("=" * 70)
    if diverged > 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmark presets across all 3 robots in batch.")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration per scenario in seconds (default: 4.0).")
    parser.add_argument("--output-dir", default="results/batch", help="Output directory for logs and plots.")
    parser.add_argument("--no-plot", action="store_true", help="Suppress plot generation.")
    args = parser.parse_args()

    run_all(duration=args.duration, output_dir=args.output_dir, no_plot=args.no_plot)


if __name__ == "__main__":
    main()

