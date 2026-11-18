"""Analytical performance evaluation, wheel slip computation, and telemetry metrics."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def compute_run_metrics(telemetry_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive trajectory tracking and physical efficiency metrics from telemetry logs."""
    if not telemetry_records:
        return {}

    times = np.array([r["time"] for r in telemetry_records])
    pos_x = np.array([r["pos_x"] for r in telemetry_records])
    pos_y = np.array([r["pos_y"] for r in telemetry_records])
    ref_x = np.array([r.get("ref_x", r["pos_x"]) for r in telemetry_records])
    ref_y = np.array([r.get("ref_y", r["pos_y"]) for r in telemetry_records])

    odom_x = np.array([r.get("odom_x", r["pos_x"]) for r in telemetry_records])
    odom_y = np.array([r.get("odom_y", r["pos_y"]) for r in telemetry_records])

    # Position tracking error
    pos_err = np.sqrt((pos_x - ref_x)**2 + (pos_y - ref_y)**2)
    rmse_tracking = float(np.sqrt(np.mean(pos_err**2)))
    max_tracking_err = float(np.max(pos_err))

    # Odometry drift
    odom_drift = np.sqrt((pos_x - odom_x)**2 + (pos_y - odom_y)**2)
    final_odom_drift = float(odom_drift[-1])
    max_odom_drift = float(np.max(odom_drift))

    # Slip velocity
    slip_mag = np.array([r.get("slip_mag", 0.0) for r in telemetry_records])
    mean_slip = float(np.mean(slip_mag))
    max_slip = float(np.max(slip_mag))

    # Motor effort and energy proxy
    t_fl = np.array([r.get("torque_fl", 0.0) for r in telemetry_records])
    t_fr = np.array([r.get("torque_fr", 0.0) for r in telemetry_records])
    t_rl = np.array([r.get("torque_rl", 0.0) for r in telemetry_records])
    t_rr = np.array([r.get("torque_rr", 0.0) for r in telemetry_records])

    w_fl = np.array([r.get("meas_w_fl", 0.0) for r in telemetry_records])
    w_fr = np.array([r.get("meas_w_fr", 0.0) for r in telemetry_records])
    w_rl = np.array([r.get("meas_w_rl", 0.0) for r in telemetry_records])
    w_rr = np.array([r.get("meas_w_rr", 0.0) for r in telemetry_records])

    total_power = np.abs(t_fl * w_fl) + np.abs(t_fr * w_fr) + np.abs(t_rl * w_rl) + np.abs(t_rr * w_rr)
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
    energy_joules = float(np.sum(total_power) * dt)
    peak_torque = float(max(np.max(np.abs(t_fl)), np.max(np.abs(t_fr)), np.max(np.abs(t_rl)), np.max(np.abs(t_rr))))

    return {
        "rmse_tracking_m": round(rmse_tracking, 4),
        "max_tracking_err_m": round(max_tracking_err, 4),
        "final_odom_drift_m": round(final_odom_drift, 4),
        "max_odom_drift_m": round(max_odom_drift, 4),
        "mean_slip_m_s": round(mean_slip, 4),
        "max_slip_m_s": round(max_slip, 4),
        "peak_torque_nm": round(peak_torque, 3),
        "energy_joules": round(energy_joules, 2),
        "duration_sec": round(float(times[-1] - times[0]), 2),
    }

