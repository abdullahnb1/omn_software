#!/usr/bin/env python3
"""Parametric multi-robot and multi-arena MJCF generator with modular <include> composition."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mecanum_sim.config import load_config, RobotConfig, SimConfig


def generate_robot_body_xml(
    robot_cfg: RobotConfig,
    payload_mass: float = 0.0,
    payload_offset: Optional[List[float]] = None,
) -> str:
    """Generates the standalone robot body/wheel/actuator MJCF XML submodel wrapped in <mujocoinclude>."""
    r = robot_cfg
    c = r.chassis
    w = r.wheels
    act = r.actuator

    p_mass = payload_mass if payload_mass > 0 else r.payload.mass
    p_offset = payload_offset if payload_offset is not None else r.payload.offset

    # Chassis dimensions
    l_c, w_c, h_c = c.length, c.width, c.height
    r_w, w_w = w.radius, w.width
    lx, ly = w.lx, w.ly
    r_roller = w.roller_radius
    rp = w.pitch_radius
    n_rollers = w.num_rollers

    # Self-collision prevention formula from Section 3.1 item 4
    w_col_half = min(ly - w_w / 2.0 - 0.012, 0.42 * w_c)
    l_col_half = l_c / 2.0 - 0.010
    h_col_half = h_c / 2.0 - 0.005

    # Ground clearance and chassis vertical placement
    # Axle is at base_link z=0, base_link will be at world z=r_w
    # Chassis bottom is at -r_w + underbelly_clearance relative to axle
    # Center of chassis collision box:
    z_chassis_center = -r_w + c.underbelly_clearance + h_col_half

    chassis_rgba = " ".join(f"{x:.4f}" for x in c.color)
    roller_rgba = " ".join(f"{x:.4f}" for x in w.roller_color)

    # Wheel definitions: name, pos, alpha (FL=+45, FR=-45, RL=-45, RR=+45)
    wheel_defs = [
        ("fl", [+lx, +ly, 0.0], +np.radians(45.0)),
        ("fr", [+lx, -ly, 0.0], -np.radians(45.0)),
        ("rl", [-lx, +ly, 0.0], -np.radians(45.0)),
        ("rr", [-lx, -ly, 0.0], +np.radians(45.0)),
    ]

    lines: List[str] = [
        '<mujocoinclude>',
        '  <default>',
        '    <default class="visual_only">',
        '      <geom contype="0" conaffinity="0" group="1"/>',
        '    </default>',
        '    <default class="roller_geom">',
        '      <geom contype="2" conaffinity="1" condim="3" solref="0.004 1" solimp="0.9 0.95 0.001" friction="0.8 0.005 0.0001"/>',
        '    </default>',
        '  </default>',
        '',
        '  <worldbody>',
        f'    <body name="base_link" pos="0 0 {r_w:.4f}">',
        '      <freejoint name="root_joint"/>',
        f'      <site name="chassis_center" pos="0 0 0" size="0.01" rgba="0 1 0 0.5"/>',
        f'      <site name="imu_site" pos="0 0 {c.com[2]:.4f}" size="0.01" rgba="1 0 0 0.5"/>',
        '',
        '      <!-- Cameras (REP-103 Upright Orientation: xyaxes="0 -1 0 0 0 1") -->',
        f'      <camera name="front_camera" pos="{l_c/2.0 + 0.01:.4f} 0 {h_c + 0.02:.4f}" xyaxes="0 -1 0 0 0 1" fovy="75"/>',
        '      <camera name="track_camera" mode="trackcom" pos="-1.30 -0.60 0.95" xyaxes="0.42 -0.91 0 0.52 0.24 0.82" fovy="60"/>',
        '',
        '      <!-- Chassis Structural Geoms -->',
        f'      <geom name="chassis_col_geom" type="box" pos="0 0 {z_chassis_center:.4f}" size="{l_col_half:.4f} {w_col_half:.4f} {h_col_half:.4f}" mass="{c.mass:.3f}" contype="1" conaffinity="1" rgba="{chassis_rgba}"/>',
        f'      <geom name="chassis_top_plate" class="visual_only" type="box" pos="0 0 {z_chassis_center + h_col_half:.4f}" size="{l_c/2.0:.4f} {w_c/2.0:.4f} 0.003" rgba="0.15 0.15 0.18 1"/>',
    ]

    # Optional Payload Box
    if p_mass > 0.0:
        p_x = r.payload.deck_pos[0] + p_offset[0]
        p_y = r.payload.deck_pos[1] + p_offset[1]
        p_z = r.payload.deck_pos[2] + p_offset[2] + 0.05
        lines.extend([
            '      <!-- Configurable Payload -->',
            f'      <body name="payload_block" pos="{p_x:.4f} {p_y:.4f} {p_z:.4f}">',
            f'        <geom name="payload_geom" type="box" size="0.10 0.08 0.05" mass="{p_mass:.3f}" contype="1" conaffinity="1" rgba="0.9 0.7 0.1 0.9"/>',
            '      </body>',
        ])

    # Generate 4 Mecanum Wheels
    for w_name, w_pos, alpha in wheel_defs:
        lines.extend([
            f'',
            f'      <!-- Wheel: {w_name.upper()} -->',
            f'      <body name="{w_name}_wheel" pos="{w_pos[0]:.4f} {w_pos[1]:.4f} {w_pos[2]:.4f}">',
            f'        <joint name="{w_name}_wheel_joint" type="hinge" axis="0 1 0" damping="{w.damping:.5f}" armature="{w.armature:.5f}"/>',
            f'        <!-- Wheel Hub Visual Cylinder -->',
            f'        <geom name="{w_name}_hub" class="visual_only" type="cylinder" size="{r_w * 0.72:.4f} {w_w * 0.42:.4f}" zaxis="0 1 0" mass="{w.assembly_mass * 0.4:.3f}" rgba="0.30 0.32 0.35 1"/>',
        ])

        # Generate N passive crowned rollers per wheel
        m_central = (w.assembly_mass * 0.6 / n_rollers) * 0.50
        m_shoulder = (w.assembly_mass * 0.6 / n_rollers) * 0.25

        for i in range(n_rollers):
            theta_i = 2.0 * math.pi * i / n_rollers
            x_i = rp * math.sin(theta_i)
            y_i = 0.0
            z_i = rp * math.cos(theta_i)

            # Axis unit vector in wheel frame
            ax = math.sin(alpha) * math.cos(theta_i)
            ay = math.cos(alpha)
            az = -math.sin(alpha) * math.sin(theta_i)
            a_vec = np.array([ax, ay, az], dtype=float)

            # 3-capsule crowned profile points
            c_p1 = -0.18 * w_w * a_vec
            c_p2 = +0.18 * w_w * a_vec
            s1_p1 = -0.40 * w_w * a_vec
            s1_p2 = -0.15 * w_w * a_vec
            s2_p1 = +0.15 * w_w * a_vec
            s2_p2 = +0.40 * w_w * a_vec

            r_shoulder = 0.88 * r_roller

            lines.extend([
                f'        <body name="{w_name}_r{i}" pos="{x_i:.6f} {y_i:.6f} {z_i:.6f}">',
                f'          <joint name="{w_name}_r{i}_j" type="hinge" axis="{ax:.6f} {ay:.6f} {az:.6f}" damping="{w.roller_damping:.6f}" armature="{w.roller_armature:.6f}"/>',
                f'          <geom class="roller_geom" type="capsule" fromto="{c_p1[0]:.6f} {c_p1[1]:.6f} {c_p1[2]:.6f} {c_p2[0]:.6f} {c_p2[1]:.6f} {c_p2[2]:.6f}" size="{r_roller:.6f}" mass="{m_central:.5f}" rgba="{roller_rgba}"/>',
                f'          <geom class="roller_geom" type="capsule" fromto="{s1_p1[0]:.6f} {s1_p1[1]:.6f} {s1_p1[2]:.6f} {s1_p2[0]:.6f} {s1_p2[1]:.6f} {s1_p2[2]:.6f}" size="{r_shoulder:.6f}" mass="{m_shoulder:.5f}" rgba="{roller_rgba}"/>',
                f'          <geom class="roller_geom" type="capsule" fromto="{s2_p1[0]:.6f} {s2_p1[1]:.6f} {s2_p1[2]:.6f} {s2_p2[0]:.6f} {s2_p2[1]:.6f} {s2_p2[2]:.6f}" size="{r_shoulder:.6f}" mass="{m_shoulder:.5f}" rgba="{roller_rgba}"/>',
                f'        </body>',
            ])

        lines.append('      </body>')

    lines.extend([
        '    </body>',
        '  </worldbody>',
        '',
        '  <actuator>',
        f'    <velocity name="fl_actuator" joint="fl_wheel_joint" kv="{act.kv:.4f}" ctrllimited="true" ctrlrange="-{act.max_speed:.2f} {act.max_speed:.2f}" forcelimited="true" forcerange="-{act.max_torque:.2f} {act.max_torque:.2f}"/>',
        f'    <velocity name="fr_actuator" joint="fr_wheel_joint" kv="{act.kv:.4f}" ctrllimited="true" ctrlrange="-{act.max_speed:.2f} {act.max_speed:.2f}" forcelimited="true" forcerange="-{act.max_torque:.2f} {act.max_torque:.2f}"/>',
        f'    <velocity name="rl_actuator" joint="rl_wheel_joint" kv="{act.kv:.4f}" ctrllimited="true" ctrlrange="-{act.max_speed:.2f} {act.max_speed:.2f}" forcelimited="true" forcerange="-{act.max_torque:.2f} {act.max_torque:.2f}"/>',
        f'    <velocity name="rr_actuator" joint="rr_wheel_joint" kv="{act.kv:.4f}" ctrllimited="true" ctrlrange="-{act.max_speed:.2f} {act.max_speed:.2f}" forcelimited="true" forcerange="-{act.max_torque:.2f} {act.max_torque:.2f}"/>',
        '  </actuator>',
        '',
        '  <sensor>',
        '    <actuatorfrc name="fl_torque" actuator="fl_actuator"/>',
        '    <actuatorfrc name="fr_torque" actuator="fr_actuator"/>',
        '    <actuatorfrc name="rl_torque" actuator="rl_actuator"/>',
        '    <actuatorfrc name="rr_torque" actuator="rr_actuator"/>',
        '    <jointpos name="fl_pos" joint="fl_wheel_joint"/>',
        '    <jointpos name="fr_pos" joint="fr_wheel_joint"/>',
        '    <jointpos name="rl_pos" joint="rl_wheel_joint"/>',
        '    <jointpos name="rr_pos" joint="rr_wheel_joint"/>',
        '    <jointvel name="fl_vel" joint="fl_wheel_joint"/>',
        '    <jointvel name="fr_vel" joint="fr_wheel_joint"/>',
        '    <jointvel name="rl_vel" joint="rl_wheel_joint"/>',
        '    <jointvel name="rr_vel" joint="rr_wheel_joint"/>',
        '    <accelerometer name="imu_accel" site="imu_site"/>',
        '    <gyro name="imu_gyro" site="imu_site"/>',
        '  </sensor>',
        '</mujocoinclude>',
    ])

    return "\n".join(lines) + "\n"


def generate_arena_open_xml(floor_friction: float = 0.8) -> str:
    """Generates the open 8x8m unobstructed arena wrapped in <mujocoinclude>."""
    return f"""<mujocoinclude>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4.0 4.0 0.1" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" material="grid_mat"/>

    <!-- Perimeter Boundary Walls (8x8m) -->
    <geom name="wall_n" type="box" pos="0 4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_s" type="box" pos="0 -4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_e" type="box" pos="4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_w" type="box" pos="-4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
  </worldbody>
</mujocoinclude>
"""


def generate_arena_doorway_xml(floor_friction: float = 0.8) -> str:
    """Generates the 8x8m arena with an 800mm doorway at X=0 and an 8mm beveled threshold."""
    return f"""<mujocoinclude>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4.0 4.0 0.1" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" material="grid_mat"/>

    <!-- Perimeter Boundary Walls (8x8m) -->
    <geom name="wall_n" type="box" pos="0 4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_s" type="box" pos="0 -4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_e" type="box" pos="4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_w" type="box" pos="-4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>

    <!-- Interior Partition Wall at X=0 with 800mm Doorway (Y in [-0.40, +0.40]) -->
    <geom name="partition_left" type="box" pos="0 2.20 0.25" size="0.04 1.80 0.25" contype="1" conaffinity="1" rgba="0.6 0.65 0.7 0.8"/>
    <geom name="partition_right" type="box" pos="0 -2.20 0.25" size="0.04 1.80 0.25" contype="1" conaffinity="1" rgba="0.6 0.65 0.7 0.8"/>

    <!-- 8mm Beveled Doorway Ground Threshold (Z=0.008m total height) -->
    <geom name="door_threshold_core" type="box" pos="0 0 0.004" size="0.015 0.40 0.004" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
    <geom name="door_threshold_ramp_e" type="cylinder" zaxis="0 1 0" pos="0.015 0 0.004" size="0.004 0.40" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
    <geom name="door_threshold_ramp_w" type="cylinder" zaxis="0 1 0" pos="-0.015 0 0.004" size="0.004 0.40" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
  </worldbody>
</mujocoinclude>
"""


def generate_arena_full_xml(floor_friction: float = 0.8) -> str:
    """Generates the full arena including doorway, 8mm & 5mm thresholds, pillars, crates, and floor cables."""
    return f"""<mujocoinclude>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4.0 4.0 0.1" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" material="grid_mat"/>

    <!-- Perimeter Boundary Walls (8x8m) -->
    <geom name="wall_n" type="box" pos="0 4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_s" type="box" pos="0 -4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_e" type="box" pos="4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_w" type="box" pos="-4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>

    <!-- Interior Partition Wall with 800mm Doorway -->
    <geom name="partition_left" type="box" pos="0 2.20 0.25" size="0.04 1.80 0.25" contype="1" conaffinity="1" rgba="0.6 0.65 0.7 0.8"/>
    <geom name="partition_right" type="box" pos="0 -2.20 0.25" size="0.04 1.80 0.25" contype="1" conaffinity="1" rgba="0.6 0.65 0.7 0.8"/>

    <!-- 8mm Doorway Threshold -->
    <geom name="door_threshold_core" type="box" pos="0 0 0.004" size="0.015 0.40 0.004" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
    <geom name="door_threshold_ramp_e" type="cylinder" zaxis="0 1 0" pos="0.015 0 0.004" size="0.004 0.40" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
    <geom name="door_threshold_ramp_w" type="cylinder" zaxis="0 1 0" pos="-0.015 0 0.004" size="0.004 0.40" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>

    <!-- 5mm Open-Arena Threshold at (-1.5, 2.0) -->
    <geom name="open_threshold_core" type="box" pos="-1.5 2.0 0.0025" size="0.50 0.015 0.0025" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.85 0.5 0.15 1"/>
    <geom name="open_threshold_ramp_n" type="cylinder" zaxis="1 0 0" pos="-1.5 2.015 0.0025" size="0.0025 0.50" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.85 0.5 0.15 1"/>
    <geom name="open_threshold_ramp_s" type="cylinder" zaxis="1 0 0" pos="-1.5 1.985 0.0025" size="0.0025 0.50" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.85 0.5 0.15 1"/>

    <!-- Cylindrical Pillars (radius 0.15m, height 1.0m) -->
    <geom name="pillar_left" type="cylinder" pos="2.2 2.0 0.50" size="0.15 0.50" contype="1" conaffinity="1" rgba="0.4 0.45 0.5 1"/>
    <geom name="pillar_right" type="cylinder" pos="2.2 -2.0 0.50" size="0.15 0.50" contype="1" conaffinity="1" rgba="0.4 0.45 0.5 1"/>

    <!-- Modular Crates -->
    <geom name="crate_1" type="box" pos="-2.5 2.0 0.20" size="0.25 0.25 0.20" contype="1" conaffinity="1" rgba="0.75 0.65 0.5 1"/>
    <geom name="crate_2" type="box" pos="-2.5 -2.0 0.20" size="0.25 0.25 0.20" contype="1" conaffinity="1" rgba="0.75 0.65 0.5 1"/>
    <geom name="crate_3" type="box" pos="2.5 0.85 0.20" size="0.20 0.20 0.20" contype="1" conaffinity="1" rgba="0.75 0.65 0.5 1"/>
    <geom name="crate_4" type="box" pos="2.5 -0.85 0.20" size="0.20 0.20 0.20" contype="1" conaffinity="1" rgba="0.75 0.65 0.5 1"/>

    <!-- Low-Profile Floor Cable Protectors (5mm radius) -->
    <geom name="cable_1" type="capsule" fromto="-0.8 1.0 0.005 -0.8 2.5 0.005" size="0.005" contype="1" conaffinity="1" condim="3" rgba="0.2 0.2 0.2 1"/>
    <geom name="cable_2" type="capsule" fromto="-0.8 -2.5 0.005 -0.8 -1.0 0.005" size="0.005" contype="1" conaffinity="1" condim="3" rgba="0.2 0.2 0.2 1"/>
    <geom name="cable_3" type="capsule" fromto="1.2 1.5 0.005 1.2 3.0 0.005" size="0.005" contype="1" conaffinity="1" condim="3" rgba="0.2 0.2 0.2 1"/>
    <geom name="cable_4" type="capsule" fromto="1.2 -3.0 0.005 1.2 -1.5 0.005" size="0.005" contype="1" conaffinity="1" condim="3" rgba="0.2 0.2 0.2 1"/>
  </worldbody>
</mujocoinclude>
"""


def generate_arena_ramps_xml(floor_friction: float = 0.8) -> str:
    """Generates arena_ramps.xml with open central spawn at (0,0), North ramp bridge, and South obstacle lane."""
    return f"""<mujocoinclude>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4.0 4.0 0.1" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" material="grid_mat"/>

    <!-- Perimeter Boundary Walls (8x8m) -->
    <geom name="wall_n" type="box" pos="0 4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_s" type="box" pos="0 -4.0 0.25" size="4.0 0.04 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_e" type="box" pos="4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>
    <geom name="wall_w" type="box" pos="-4.0 0 0.25" size="0.04 4.0 0.25" contype="1" conaffinity="1" rgba="0.5 0.55 0.6 0.6"/>

    <!-- ======================================================== -->
    <!-- NORTH OVERPASS BRIDGE & DUAL RAMPS (Y in [1.3, 2.7])     -->
    <!-- Center Y = 2.0m, Width = 1.4m (half-width = 0.70m)       -->
    <!-- ======================================================== -->
    <!-- West Climb Ramp: length 1.6m (X in [-2.4, -0.8]), rises 0.0 to 0.08m (2.86 deg) -->
    <geom name="ramp_climb" type="box" pos="-1.6 2.0 0.04" size="0.80 0.70 0.006" euler="0 -0.05 0" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.35 0.40 0.48 1"/>
    <!-- Climb Ramp Curbs / Safety Edges -->
    <geom name="ramp_climb_curb_l" type="box" pos="-1.6 2.72 0.065" size="0.80 0.02 0.03" euler="0 -0.05 0" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>
    <geom name="ramp_climb_curb_r" type="box" pos="-1.6 1.28 0.065" size="0.80 0.02 0.03" euler="0 -0.05 0" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>

    <!-- Elevated Bridge Plateau: X in [-0.8, +0.8], height 0.08m -->
    <geom name="bridge_plateau" type="box" pos="0 2.0 0.04" size="0.80 0.70 0.04" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.25 0.32 0.40 1"/>
    <!-- Plateau Side Guardrails -->
    <geom name="bridge_rail_l" type="box" pos="0 2.72 0.11" size="0.80 0.02 0.04" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>
    <geom name="bridge_rail_r" type="box" pos="0 1.28 0.11" size="0.80 0.02 0.04" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>

    <!-- East Descent Ramp: length 1.6m (X in [+0.8, +2.4]), descends 0.08 to 0.0m -->
    <geom name="ramp_descent" type="box" pos="1.6 2.0 0.04" size="0.80 0.70 0.006" euler="0 0.05 0" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.35 0.40 0.48 1"/>
    <!-- Descent Ramp Curbs -->
    <geom name="ramp_desc_curb_l" type="box" pos="1.6 2.72 0.065" size="0.80 0.02 0.03" euler="0 0.05 0" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>
    <geom name="ramp_desc_curb_r" type="box" pos="1.6 1.28 0.065" size="0.80 0.02 0.03" euler="0 0.05 0" contype="1" conaffinity="1" rgba="0.95 0.65 0.15 1"/>

    <!-- ======================================================== -->
    <!-- SOUTH OBSTACLE & SLALOM LANE (Y in [-2.7, -1.3])         -->
    <!-- Center Y = -2.0m, Width = 1.4m (half-width = 0.70m)      -->
    <!-- ======================================================== -->
    <!-- Speed Table / Hump at X = -2.4, Y = -2.0 (Height 12mm) -->
    <geom name="speed_hump_w" type="box" pos="-2.4 -2.0 0.006" size="0.35 0.60 0.006" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>
    <geom name="speed_hump_e" type="box" pos="2.4 -2.0 0.006" size="0.35 0.60 0.006" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>

    <!-- Slalom Agility Pylons at X = -1.0, 0.0, 1.0, Y = -2.0 -->
    <geom name="pylon_1" type="cylinder" pos="-1.0 -2.0 0.20" size="0.12 0.20" contype="1" conaffinity="1" rgba="0.15 0.70 0.65 0.9"/>
    <geom name="pylon_2" type="cylinder" pos="0.0 -2.0 0.20" size="0.12 0.20" contype="1" conaffinity="1" rgba="0.15 0.70 0.65 0.9"/>
    <geom name="pylon_3" type="cylinder" pos="1.0 -2.0 0.20" size="0.12 0.20" contype="1" conaffinity="1" rgba="0.15 0.70 0.65 0.9"/>

    <!-- Transverse Rumble Strips (Height 6mm, spacing 0.25m, at Y = -2.0) -->
    <geom name="rumble_1" type="box" pos="-0.50 -2.0 0.003" size="0.02 0.60 0.003" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.85 0.3 0.2 1"/>
    <geom name="rumble_2" type="box" pos="-0.25 -2.0 0.003" size="0.02 0.60 0.003" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.85 0.3 0.2 1"/>

    <!-- Lateral Banked Camber Wedge (Roll Tilt 3.4 deg) at X in [0.3, 1.7] -->
    <geom name="banked_wedge" type="box" pos="1.0 -2.0 0.015" size="0.70 0.50 0.006" euler="0.06 0 0" contype="1" conaffinity="1" condim="3" friction="{floor_friction:.3f} 0.005 0.0001" rgba="0.45 0.35 0.55 1"/>
  </worldbody>
</mujocoinclude>
"""


def generate_composed_scene_xml(
    robot_name: str,
    arena_type: str,
    timestep: float = 0.002,
    body_filename: Optional[str] = None,
) -> str:
    """Generates the composed root MJCF scene file linking robot body and arena via <include>."""
    body_inc = body_filename or f"robots/{robot_name}_body.xml"
    return f"""<mujoco model="{robot_name}_{arena_type}">
  <compiler angle="radian"/>
  <option timestep="{timestep:.4f}" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" cone="elliptic" impratio="10" iterations="60" tolerance="1e-8"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.20 0.22 0.25" rgb2="0.30 0.33 0.36"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <include file="scenes/arena_{arena_type}.xml"/>
  <include file="{body_inc}"/>
</mujoco>
"""


def generate_all_models(
    models_dir: Optional[Path] = None,
    robots: Optional[List[str]] = None,
    payload_mass: float = 0.0,
    payload_offset: Optional[List[float]] = None,
    floor_friction: float = 0.8,
) -> None:
    """Generates all standalone and composed XML files for the specified robots and arenas."""
    out_dir = models_dir or (Path(__file__).resolve().parent)
    robots_dir = out_dir / "robots"
    scenes_dir = out_dir / "scenes"
    robots_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    target_robots = robots or ["1", "2", "3"]

    # 1. Generate standalone arena scene XMLs
    (scenes_dir / "arena_open.xml").write_text(generate_arena_open_xml(floor_friction), encoding="utf-8")
    (scenes_dir / "arena_doorway.xml").write_text(generate_arena_doorway_xml(floor_friction), encoding="utf-8")
    (scenes_dir / "arena_full.xml").write_text(generate_arena_full_xml(floor_friction), encoding="utf-8")
    (scenes_dir / "arena_ramps.xml").write_text(generate_arena_ramps_xml(floor_friction), encoding="utf-8")
    print(f"Generated arena XMLs in {scenes_dir}")

    # 2. For each robot, generate body XML and composed scenes
    for r_key in target_robots:
        cfg = load_config(robot=r_key, payload_mass=payload_mass, payload_offset=payload_offset)
        r_name = cfg.robot.name

        short_names = {
            "robot_1_compact": "robot_1",
            "robot_2_heavy": "robot_2",
            "robot_3_custom": "robot_3",
        }
        prefix = short_names.get(r_name, r_name)

        if payload_mass > 0.0 or payload_offset is not None:
            # Generate dedicated payload body and scenes without modifying unladen base
            body_xml = generate_robot_body_xml(cfg.robot, payload_mass, payload_offset)
            body_path = robots_dir / f"{r_name}_payload_body.xml"
            body_path.write_text(body_xml, encoding="utf-8")
            body_rel = f"robots/{r_name}_payload_body.xml"

            for a_type in ["open", "doorway", "full", "ramps"]:
                (out_dir / f"{prefix}_{a_type}_payload.xml").write_text(
                    generate_composed_scene_xml(f"{r_name}_payload", a_type, cfg.simulation.physics_timestep, body_filename=body_rel),
                    encoding="utf-8",
                )
        else:
            # Clean unladen base model (payload_mass = 0.0)
            body_xml = generate_robot_body_xml(cfg.robot, 0.0, None)
            body_path = robots_dir / f"{r_name}_body.xml"
            body_path.write_text(body_xml, encoding="utf-8")
            print(f"Generated robot body XML: {body_path}")

            (out_dir / f"{prefix}_open.xml").write_text(
                generate_composed_scene_xml(r_name, "open", cfg.simulation.physics_timestep),
                encoding="utf-8",
            )
            (out_dir / f"{prefix}_doorway.xml").write_text(
                generate_composed_scene_xml(r_name, "doorway", cfg.simulation.physics_timestep),
                encoding="utf-8",
            )
            (out_dir / f"{prefix}_ramps.xml").write_text(
                generate_composed_scene_xml(r_name, "ramps", cfg.simulation.physics_timestep),
                encoding="utf-8",
            )
            (out_dir / f"{r_name}.xml").write_text(
                generate_composed_scene_xml(r_name, "full", cfg.simulation.physics_timestep),
                encoding="utf-8",
            )

    print("All MJCF models generated successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parametric MuJoCo MJCF robot and arena models.")
    parser.add_argument(
        "--robot",
        default="all",
        help="Robot identifier ('1', '2', '3', 'all', or name). Default: 'all'.",
    )
    parser.add_argument(
        "--arena",
        default="all",
        choices=["open", "doorway", "full", "all"],
        help="Arena variant to generate. Default: 'all'.",
    )
    parser.add_argument("--payload-mass", type=float, default=0.0, help="Payload mass in kg (default: 0.0)")
    parser.add_argument(
        "--payload-offset",
        type=float,
        nargs=3,
        default=None,
        help="Payload offset [x, y, z] in meters relative to payload deck.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.8, help="Floor friction coefficient (default: 0.8)")
    parser.add_argument("--output-dir", default=None, help="Directory for generated XML files.")
    args = parser.parse_args()

    models_dir = Path(args.output_dir) if args.output_dir else None
    target_r = ["1", "2", "3"] if str(args.robot).lower() == "all" else [args.robot]

    generate_all_models(
        models_dir=models_dir,
        robots=target_r,
        payload_mass=args.payload_mass,
        payload_offset=args.payload_offset,
        floor_friction=args.floor_friction,
    )


if __name__ == "__main__":
    main()

