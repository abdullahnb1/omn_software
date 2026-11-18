#!/usr/bin/env python3
"""Generates single-robot telemetry dashboards and cross-robot comparative visualization plots."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_telemetry_csv(csv_path: Path) -> pd.DataFrame:
    """Reads telemetry CSV log into pandas DataFrame."""
    return pd.read_csv(csv_path)


def generate_single_robot_dashboard(csv_path: Path, output_dir: Optional[Path] = None) -> List[Path]:
    """Generates 6-panel summary dashboard, tracking plot, and top-down arena trajectory plot."""
    df = load_telemetry_csv(csv_path)
    out_dir = output_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem.replace("_telemetry", "")
    created_plots: List[Path] = []

    # --------------------------------------------------------------------------
    # 1. 6-Panel Summary Dashboard
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Telemetry Dashboard: {stem}", fontsize=14, fontweight="bold")

    times = df["time"].to_numpy()

    # Panel 1: Commanded vs Measured Body Velocities
    ax = axes[0, 0]
    ax.plot(times, df["cmd_limited_vx"], "r--", label="Cmd vx (m/s)")
    ax.plot(times, df["meas_vx_body"], "r-", label="Actual vx (m/s)")
    ax.plot(times, df["cmd_limited_vy"], "b--", label="Cmd vy (m/s)")
    ax.plot(times, df["meas_vy_body"], "b-", label="Actual vy (m/s)")
    ax.plot(times, df["cmd_limited_wz"], "g--", label="Cmd wz (rad/s)")
    ax.plot(times, df["meas_wz_body"], "g-", label="Actual wz (rad/s)")
    ax.set_title("Chassis Body Velocities")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: 2D Trajectory with Ground Truth vs Odometry
    ax = axes[0, 1]
    ax.plot(df["ref_x"], df["ref_y"], "k:", label="Reference Path")
    ax.plot(df["pos_x"], df["pos_y"], "b-", label="True Trajectory")
    ax.plot(df["odom_x"], df["odom_y"], "r--", label="Dead-Reckoned Odom")
    ax.plot(df["pos_x"].iloc[0], df["pos_y"].iloc[0], "go", markersize=6, label="Start")
    ax.plot(df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], "ks", markersize=6, label="End")
    ax.set_title("2D Floor Trajectory & Odometry Drift")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Commanded vs Measured Wheel Speeds
    ax = axes[1, 0]
    ax.plot(times, df["ctrl_w_fl"], "r--", label="Cmd FL (rad/s)")
    ax.plot(times, df["meas_w_fl"], "r-", label="Meas FL (rad/s)")
    ax.plot(times, df["ctrl_w_fr"], "b--", label="Cmd FR (rad/s)")
    ax.plot(times, df["meas_w_fr"], "b-", label="Meas FR (rad/s)")
    ax.set_title("Front Wheel Velocities (FL / FR)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 4: Rear Wheel Velocities (RL / RR)
    ax = axes[1, 1]
    ax.plot(times, df["ctrl_w_rl"], "m--", label="Cmd RL (rad/s)")
    ax.plot(times, df["meas_w_rl"], "m-", label="Meas RL (rad/s)")
    ax.plot(times, df["ctrl_w_rr"], "c--", label="Cmd RR (rad/s)")
    ax.plot(times, df["meas_w_rr"], "c-", label="Meas RR (rad/s)")
    ax.set_title("Rear Wheel Velocities (RL / RR)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 5: Motor Torques
    ax = axes[2, 0]
    ax.plot(times, df["torque_fl"], label="Torque FL (N·m)")
    ax.plot(times, df["torque_fr"], label="Torque FR (N·m)")
    ax.plot(times, df["torque_rl"], label="Torque RL (N·m)")
    ax.plot(times, df["torque_rr"], label="Torque RR (N·m)")
    ax.set_title("Actuator Physical Applied Torque")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 6: Wheel Slip Magnitude & Odometry Drift
    ax = axes[2, 1]
    ax.plot(times, df["slip_mag"], "r-", label="Slip Velocity ||v_slip|| (m/s)")
    ax2 = ax.twinx()
    ax2.plot(times, df["odom_drift"], "m--", label="Odometry Drift ||e_odom|| (m)")
    ax.set_title("Instantaneous Wheel Slip & Dead-Reckoning Drift")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slip Velocity (m/s)", color="r")
    ax2.set_ylabel("Drift (m)", color="m")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    dash_path = out_dir / f"{stem}_dashboard.png"
    plt.savefig(dash_path, dpi=120)
    plt.close()
    created_plots.append(dash_path)

    # --------------------------------------------------------------------------
    # 2. Tracking Error Breakdown Plot
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Pose & Velocity Tracking Analysis: {stem}", fontsize=13, fontweight="bold")

    e_x = df["ref_x"] - df["pos_x"]
    e_y = df["ref_y"] - df["pos_y"]
    e_xy = np.sqrt(e_x**2 + e_y**2)
    e_th = np.arctan2(np.sin(df["ref_yaw"] - df["yaw"]), np.cos(df["ref_yaw"] - df["yaw"]))

    ax = axes[0]
    ax.plot(times, e_xy, "r-", label="Planar Distance Error ||e_xy|| (m)")
    ax.plot(times, e_x, "b:", label="e_x (m)")
    ax.plot(times, e_y, "g:", label="e_y (m)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Position Tracking Error")

    ax = axes[1]
    ax.plot(times, np.degrees(e_th), "m-", label="Heading Error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw Error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Heading Tracking Error")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    track_path = out_dir / f"{stem}_tracking.png"
    plt.savefig(track_path, dpi=120)
    plt.close()
    created_plots.append(track_path)

    # --------------------------------------------------------------------------
    # 3. Top-Down 2D Trajectory with Arena Layout Overlay
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Arena Top-Down Trajectory: {stem}", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m) [East / Forward]")
    ax.set_ylabel("Y (m) [North / Left]")

    # Perimeter walls
    ax.plot([-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4], "k-", linewidth=2.5, label="Arena Perimeter")
    # Partition wall with 800mm doorway at X=0
    ax.plot([0, 0], [0.4, 4.0], "gray", linewidth=3.5)
    ax.plot([0, 0], [-4.0, -0.4], "gray", linewidth=3.5)
    # 8mm doorway threshold
    ax.plot([0, 0], [-0.4, 0.4], "orange", linewidth=4.0, label="8mm Doorway Threshold")
    # 5mm threshold
    ax.plot([-1.5, -1.5], [1.5, 2.5], color="darkorange", linewidth=3.0, linestyle=":", label="5mm Threshold")

    # Pillars
    for py in [2.0, -2.0]:
        circle = plt.Circle((2.2, py), 0.15, color="teal", alpha=0.6)
        ax.add_patch(circle)

    # Crates
    crates = [(-2.5, 2.0), (-2.5, -2.0), (2.5, 0.85), (2.5, -0.85)]
    for cx, cy in crates:
        rect = plt.Rectangle((cx - 0.15, cy - 0.15), 0.30, 0.30, color="saddlebrown", alpha=0.5)
        ax.add_patch(rect)

    # Robot trajectory
    ax.plot(df["ref_x"], df["ref_y"], "k:", linewidth=1.5, label="Reference Path")
    ax.plot(df["pos_x"], df["pos_y"], "b-", linewidth=2.0, label="Actual Trajectory")
    ax.plot(df["odom_x"], df["odom_y"], "r--", linewidth=1.2, label="Dead-Reckoned Odometry")
    ax.plot(df["pos_x"].iloc[0], df["pos_y"].iloc[0], "go", markersize=8, label="Start")
    ax.plot(df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], "ks", markersize=8, label="End")

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    traj_path = out_dir / f"{stem}_trajectory.png"
    plt.savefig(traj_path, dpi=120)
    plt.close()
    created_plots.append(traj_path)

    print(f"Generated single-robot plots in: {out_dir}")
    return created_plots


def generate_cross_robot_comparison(
    csv_paths: List[Path],
    scenario_name: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generates cross-robot overlay dashboard and summary comparison table for all 3 robots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_records: List[Dict[str, Any]] = []

    # Map colors and labels for robots
    robot_styles = {
        "robot_1_compact": {"color": "royalblue", "label": "Robot 1 (Compact)"},
        "robot_2_heavy": {"color": "dimgray", "label": "Robot 2 (Heavy)"},
        "robot_3_custom": {"color": "teal", "label": "Robot 3 (Custom)"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Cross-Robot Benchmark Comparison: {scenario_name}", fontsize=14, fontweight="bold")

    for p in sorted(csv_paths):
        df = load_telemetry_csv(p)
        r_name = df["robot_name"].iloc[0]
        style = robot_styles.get(r_name, {"color": "black", "label": r_name})
        color = style["color"]
        label = style["label"]

        times = df["time"].to_numpy()
        e_xy = np.sqrt((df["ref_x"] - df["pos_x"])**2 + (df["ref_y"] - df["pos_y"])**2)

        # 1. Subplot (0, 0): 2D Path Comparison
        ax = axes[0, 0]
        ax.plot(df["pos_x"], df["pos_y"], color=color, linewidth=2.0, label=f"{label} Path")
        ax.plot(df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], marker="s", color=color, markersize=5)

        # 2. Subplot (0, 1): Position Tracking Error Norm ||e_xy||
        ax = axes[0, 1]
        ax.plot(times, e_xy, color=color, linewidth=1.8, label=f"{label} ||e_xy||")

        # 3. Subplot (1, 0): Wheel Slip Magnitude
        ax = axes[1, 0]
        ax.plot(times, df["slip_mag"], color=color, linewidth=1.5, label=f"{label} Slip")

        # 4. Subplot (1, 1): Normalized Motor Effort (RMS torque)
        ax = axes[1, 1]
        rms_t = np.sqrt((df["torque_fl"]**2 + df["torque_fr"]**2 + df["torque_rl"]**2 + df["torque_rr"]**2) / 4.0)
        ax.plot(times, rms_t, color=color, linewidth=1.5, label=f"{label} RMS Torque")

        # Summary Metrics
        summary_records.append({
            "robot_name": r_name,
            "scenario": scenario_name,
            "rmse_tracking_m": round(float(np.sqrt(np.mean(e_xy**2))), 4),
            "max_tracking_err_m": round(float(np.max(e_xy)), 4),
            "final_odom_drift_m": round(float(df["odom_drift"].iloc[-1]), 4),
            "mean_slip_m_s": round(float(np.mean(df["slip_mag"])), 4),
            "max_slip_m_s": round(float(np.max(df["slip_mag"])), 4),
            "peak_torque_nm": round(float(max(df["torque_fl"].abs().max(), df["torque_fr"].abs().max(), df["torque_rl"].abs().max(), df["torque_rr"].abs().max())), 3),
        })

    # Reference path on 2D plot
    if len(csv_paths) > 0:
        first_df = load_telemetry_csv(csv_paths[0])
        axes[0, 0].plot(first_df["ref_x"], first_df["ref_y"], "k:", linewidth=1.5, label="Reference Path")

    # Format Subplots
    axes[0, 0].set_title("2D Path Comparison")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].set_title("Position Tracking Error Norm ||e_xy||")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Tracking Error (m)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].set_title("Wheel Slip Velocity Magnitude ||v_slip||")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Slip (m/s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].set_title("Average RMS Wheel Torque")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Torque (N·m)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    cmp_plot_path = output_dir / f"comparison_{scenario_name}.png"
    plt.savefig(cmp_plot_path, dpi=120)
    plt.close()

    # Save summary CSV
    cmp_csv_path = output_dir / f"comparison_{scenario_name}_summary.csv"
    if summary_records:
        keys = list(summary_records[0].keys())
        with open(cmp_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_records)

    print(f"Generated cross-robot comparison: {cmp_plot_path} and {cmp_csv_path}")
    return cmp_plot_path, cmp_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate telemetry plots from CSV logs.")
    parser.add_argument("csv_files", nargs="+", help="One or more telemetry CSV files to plot.")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots.")
    parser.add_argument("--scenario", default="benchmark", help="Scenario name for cross-robot comparison.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.csv_files]
    out_dir = Path(args.output_dir) if args.output_dir else paths[0].parent

    for p in paths:
        generate_single_robot_dashboard(p, out_dir)

    if len(paths) >= 2:
        generate_cross_robot_comparison(paths, args.scenario, out_dir)


if __name__ == "__main__":
    main()

