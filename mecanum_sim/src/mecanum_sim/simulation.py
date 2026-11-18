"""Multi-rate simulation engine, offscreen video recorder, CSV telemetry logger, and ROS 2 adapters."""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
try:
    import mujoco.viewer
except Exception:
    pass
import numpy as np
from PIL import Image

from mecanum_sim.config import load_config, RobotConfig, SimConfig
from mecanum_sim.controller import MecanumController
from mecanum_sim.interfaces import (
    ClockMessage,
    Header,
    ImuStateMessage,
    JointStateMessage,
    MuJoCoVelocityActuator,
    OdometryState,
    Quaternion,
    TransformMessage,
    TwistCommand,
    Vector3,
)
from mecanum_sim.kinematics import MecanumKinematics, WheelOdometryEstimator
from mecanum_sim.metrics import compute_run_metrics
from mecanum_sim.path_tracker import PathTracker
from mecanum_sim.trajectory import (
    CubicSplineTrajectory,
    create_trajectory,
    load_waypoints_json,
)
from mecanum_sim.viewer_gui import FreeTeleopController


class MecanumSimulation:
    """Core simulation orchestrator running 500 Hz physics and 100 Hz velocity control."""

    def __init__(
        self,
        robot: str = "1",
        arena: str = "auto",
        mode: str = "free",
        preset: Optional[str] = None,
        waypoints_file: Optional[str] = None,
        waypoints: Optional[List[Dict[str, float]]] = None,
        speed: Optional[float] = None,
        duration: Optional[float] = None,
        payload_mass: float = 0.0,
        payload_offset: Optional[List[float]] = None,
        floor_friction: float = 0.8,
        pose_correction: bool = True,
        gui: bool = False,
        no_front_cam: bool = False,
        record_video: Optional[str] = None,
        camera: str = "track_camera",
        output_dir: str = "results",
        no_csv: bool = False,
        non_interactive: bool = False,
    ):
        self.mode = mode.lower()
        self.waypoints = waypoints
        if self.waypoints is not None and len(self.waypoints) > 0:
            self.preset = preset
        else:
            self.preset = preset or ("doorway_and_threshold" if self.mode == "path" else None)
        self.waypoints_file = waypoints_file
        self.user_speed = speed
        self.user_duration = duration
        self.gui = gui
        self.no_front_cam = no_front_cam
        self.record_video_path = record_video
        self.record_camera = camera
        self.output_dir = Path(output_dir)
        self.no_csv = no_csv
        self.non_interactive = non_interactive

        # 1. Resolve arena variant
        chosen_arena = self._resolve_arena(arena, self.mode, self.preset)

        # 2. Load configuration
        self.cfg = load_config(
            robot=robot,
            arena=chosen_arena,
            payload_mass=payload_mass,
            payload_offset=payload_offset,
            floor_friction=floor_friction,
        )
        self.robot = self.cfg.robot

        # 3. Load or generate MuJoCo model
        self.model, self.model_path = self._load_model(chosen_arena, floor_friction, payload_mass, payload_offset)
        self.data = mujoco.MjData(self.model)

        # 4. Resolve bodies, joints, sensors
        self.base_id = self.model.body("base_link").id

        # 5. Controller, Kinematics, Odometry, Actuator
        self.kinematics = MecanumKinematics(
            wheel_radius=self.robot.wheel_radius,
            lx=self.robot.lx,
            ly=self.robot.ly,
        )
        self.odometry = WheelOdometryEstimator(self.kinematics)
        self.controller = MecanumController(self.robot)
        self.path_tracker = PathTracker(self.robot, enable_pose_correction=pose_correction)
        self.actuator = MuJoCoVelocityActuator(self.model, self.data)
        self.teleop = FreeTeleopController(self.robot)

        # 6. Trajectory setup if in path mode
        self.trajectory: Optional[CubicSplineTrajectory] = None
        self._setup_trajectory()

        # 7. Multi-rate timing parameters
        self.dt_physics = float(self.model.opt.timestep)  # e.g. 0.002 (500 Hz)
        self.dt_control = float(self.cfg.simulation.control_timestep)  # e.g. 0.010 (100 Hz)
        self.substeps = max(1, int(round(self.dt_control / self.dt_physics)))

        # 8. Telemetry state
        self.sim_time = 0.0
        self.telemetry_records: List[Dict[str, Any]] = []
        self.has_nan = False

    def _resolve_arena(self, arena: str, mode: str, preset: Optional[str]) -> str:
        """Auto-pairs scenario with optimal arena environment if arena is 'auto'."""
        a = arena.strip().lower()
        if a != "auto":
            return a

        if preset:
            p = preset.lower()
            if "doorway" in p:
                return "doorway"
            if "obstacle" in p or "corridor" in p:
                return "full"

        if mode in ["free", "draw"] or (mode == "path" and self.waypoints is not None):
            return "full"

        # Default open for wide-field curves (figure8, cloverleaf, square, etc.)
        return "open"

    def _load_model(
        self,
        arena: str,
        friction: float,
        payload_mass: float,
        payload_offset: Optional[List[float]],
    ) -> Tuple[mujoco.MjModel, Path]:
        """Locates and loads compiled MJCF model matching robot and arena."""
        from models.generate_model import generate_all_models

        models_dir = Path("models")
        # Map robot profile and arena to scene filename
        short_names = {
            "robot_1_compact": "robot_1",
            "robot_2_heavy": "robot_2",
            "robot_3_custom": "robot_3",
        }
        prefix = short_names.get(self.robot.name, self.robot.name)

        # Dynamic payload: write to dedicated _payload files to NEVER pollute base unladen models
        if payload_mass > 0.0 or payload_offset is not None:
            scene_path = models_dir / f"{prefix}_{arena}_payload.xml"
            if not scene_path.exists():
                generate_all_models(
                    models_dir=models_dir,
                    robots=[self.robot.name],
                    payload_mass=payload_mass,
                    payload_offset=payload_offset,
                    floor_friction=friction,
                )
            else:
                # Ensure payload params match current run
                generate_all_models(
                    models_dir=models_dir,
                    robots=[self.robot.name],
                    payload_mass=payload_mass,
                    payload_offset=payload_offset,
                    floor_friction=friction,
                )
        else:
            if arena == "open":
                scene_path = models_dir / f"{prefix}_open.xml"
            elif arena == "doorway":
                scene_path = models_dir / f"{prefix}_doorway.xml"
            elif arena == "ramps":
                scene_path = models_dir / f"{prefix}_ramps.xml"
            else:
                scene_path = models_dir / f"{self.robot.name}.xml"

            if not scene_path.exists():
                generate_all_models(
                    models_dir=models_dir,
                    robots=[self.robot.name],
                    payload_mass=0.0,
                    floor_friction=friction,
                )

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        return model, scene_path

    def _setup_trajectory(self) -> None:
        """Initializes CubicSplineTrajectory for path mode or loaded waypoints."""
        nominal_spd = self.user_speed or self.robot.limits.nominal_speed

        if self.waypoints is not None and len(self.waypoints) > 0:
            self.trajectory = create_trajectory(self.waypoints, nominal_speed=nominal_spd, duration=self.user_duration)
        elif self.waypoints_file:
            wps, json_speed = load_waypoints_json(self.waypoints_file)
            spd = self.user_speed or json_speed
            self.trajectory = create_trajectory(wps, nominal_speed=spd, duration=self.user_duration)
        elif self.preset:
            self.trajectory = create_trajectory(self.preset, nominal_speed=nominal_spd, duration=self.user_duration)

    def _get_chassis_state(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """Reads ground-truth world position, orientation quaternion, yaw, world vel, and body vel."""
        pos = self.data.xpos[self.base_id].copy()
        quat = self.data.xquat[self.base_id].copy()  # [qw, qx, qy, qz]

        # Planar yaw
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(math.atan2(siny_cosp, cosy_cosp))

        # Velocities: root joint is freejoint (qvel: 3 linear, 3 angular)
        qvel = self.data.qvel[:6].copy()
        v_world = qvel[:3]
        w_world = qvel[3:]

        # Transform linear velocity into robot body frame
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx_body = float(cos_y * v_world[0] + sin_y * v_world[1])
        vy_body = float(-sin_y * v_world[0] + cos_y * v_world[1])
        wz_body = float(w_world[2])
        v_body = np.array([vx_body, vy_body, wz_body], dtype=float)

        return pos, quat, yaw, v_world, v_body

    def step(self, commanded_twist: Optional[Union[np.ndarray, list, tuple]] = None) -> Dict[str, Any]:
        """Executes one control step (100 Hz) consisting of 5 physics sub-steps."""
        # 1. Ground truth chassis state
        pos, quat, yaw, v_world, v_body = self._get_chassis_state()

        # 2. Actuator feedback
        pos_w, vel_w, eff_w = self.actuator.read_feedback()

        # 3. Update SE(2) dead-reckoned odometry
        self.odometry.update(vel_w, self.dt_control)
        odom_x, odom_y, odom_yaw = self.odometry.pose

        # 4. Resolve commanded twist
        ref_x, ref_y, ref_yaw = pos[0], pos[1], yaw
        if commanded_twist is not None:
            cmd_raw = np.asarray(commanded_twist, dtype=float).reshape(3)
        elif self.mode == "path" and self.trajectory is not None:
            pt = self.trajectory.evaluate(self.sim_time)
            ref_x, ref_y, ref_yaw = pt.x, pt.y, pt.theta
            cmd_raw, _ = self.path_tracker.compute_twist(pt, (odom_x, odom_y, odom_yaw))
        else:
            # Free mode teleoperation
            cmd_raw = self.teleop.update(self.dt_control)

        # 5. Inner-loop velocity control and direction-preserving desaturation
        actuator_cmds, desat_w, twist_lim = self.controller.update(cmd_raw, vel_w, self.dt_control)

        # 6. Dispatch velocity commands to MuJoCo <velocity> servos
        self.actuator.send_velocities(actuator_cmds)

        # 7. Slip and drift metrics
        enc_twist = self.kinematics.forward(vel_w)
        slip_vx = v_body[0] - enc_twist[0]
        slip_vy = v_body[1] - enc_twist[1]
        slip_mag = float(math.sqrt(slip_vx**2 + slip_vy**2))
        odom_drift = float(math.sqrt((pos[0] - odom_x)**2 + (pos[1] - odom_y)**2))

        # Check for NaNs
        if np.isnan(pos).any() or np.isnan(vel_w).any():
            self.has_nan = True

        # 8. Record telemetry snapshot
        snapshot = {
            "time": round(self.sim_time, 4),
            "robot_name": self.robot.name,
            "cmd_vx": round(float(cmd_raw[0]), 4),
            "cmd_vy": round(float(cmd_raw[1]), 4),
            "cmd_wz": round(float(cmd_raw[2]), 4),
            "cmd_limited_vx": round(float(twist_lim[0]), 4),
            "cmd_limited_vy": round(float(twist_lim[1]), 4),
            "cmd_limited_wz": round(float(twist_lim[2]), 4),
            "desired_w_fl": round(float(desat_w[0]), 4),
            "desired_w_fr": round(float(desat_w[1]), 4),
            "desired_w_rl": round(float(desat_w[2]), 4),
            "desired_w_rr": round(float(desat_w[3]), 4),
            "ctrl_w_fl": round(float(actuator_cmds[0]), 4),
            "ctrl_w_fr": round(float(actuator_cmds[1]), 4),
            "ctrl_w_rl": round(float(actuator_cmds[2]), 4),
            "ctrl_w_rr": round(float(actuator_cmds[3]), 4),
            "meas_w_fl": round(float(vel_w[0]), 4),
            "meas_w_fr": round(float(vel_w[1]), 4),
            "meas_w_rl": round(float(vel_w[2]), 4),
            "meas_w_rr": round(float(vel_w[3]), 4),
            "err_w_fl": round(float(desat_w[0] - vel_w[0]), 4),
            "err_w_fr": round(float(desat_w[1] - vel_w[1]), 4),
            "err_w_rl": round(float(desat_w[2] - vel_w[2]), 4),
            "err_w_rr": round(float(desat_w[3] - vel_w[3]), 4),
            "torque_fl": round(float(eff_w[0]), 4),
            "torque_fr": round(float(eff_w[1]), 4),
            "torque_rl": round(float(eff_w[2]), 4),
            "torque_rr": round(float(eff_w[3]), 4),
            "pos_x": round(float(pos[0]), 4),
            "pos_y": round(float(pos[1]), 4),
            "pos_z": round(float(pos[2]), 4),
            "qw": round(float(quat[0]), 4),
            "qx": round(float(quat[1]), 4),
            "qy": round(float(quat[2]), 4),
            "qz": round(float(quat[3]), 4),
            "yaw": round(float(yaw), 4),
            "odom_x": round(float(odom_x), 4),
            "odom_y": round(float(odom_y), 4),
            "odom_yaw": round(float(odom_yaw), 4),
            "ref_x": round(float(ref_x), 4),
            "ref_y": round(float(ref_y), 4),
            "ref_yaw": round(float(ref_yaw), 4),
            "meas_vx_body": round(float(v_body[0]), 4),
            "meas_vy_body": round(float(v_body[1]), 4),
            "meas_wz_body": round(float(v_body[2]), 4),
            "slip_vx": round(float(slip_vx), 4),
            "slip_vy": round(float(slip_vy), 4),
            "slip_mag": round(slip_mag, 4),
            "odom_drift": round(odom_drift, 4),
        }
        self.telemetry_records.append(snapshot)

        # 9. Step physical simulation substeps
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.sim_time += self.dt_control
        return snapshot

    def _on_viewer_key(self, keycode: int) -> None:
        """Callback for keyboard events from MuJoCo GLFW viewer window."""
        GLFW_MAP = {
            265: "up",
            264: "down",
            263: "left",
            262: "right",
            266: "prior",
            267: "next",
            259: "backspace",
            32: "space",
            # Keypad / Numpad in GLFW
            328: "8",
            325: "5",
            324: "4",
            326: "6",
            327: "7",
            329: "9",
            334: "+",  # Keypad +
            333: "-",  # Keypad -
            # Standard ASCII numbers & symbols
            56: "8",
            53: "5",
            52: "4",
            54: "6",
            55: "7",
            57: "9",
            91: "[",
            93: "]",
            44: ",",
            46: ".",
            61: "+",  # Main keyboard '=' / '+'
            45: "-",  # Main keyboard '-'
            43: "+",
        }
        if keycode in GLFW_MAP:
            key_name = GLFW_MAP[keycode]
        else:
            try:
                key_name = chr(keycode).lower()
            except (ValueError, OverflowError):
                return

        self.teleop.handle_viewer_key(key_name)
        if self.teleop.reset_requested:
            self.reset_robot()
            self.teleop.reset_requested = False

    def _update_viewer_camera(self, viewer: Any) -> None:
        """Updates MuJoCo viewer camera mode according to teleop selection."""
        cam_idx = self.teleop.active_camera_idx
        try:
            if cam_idx == 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            elif cam_idx == 1:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track_camera")
                if cam_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
            elif cam_idx == 2:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
                if cam_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
        except Exception:
            pass

    def reset_robot(self) -> None:
        """Resets robot position, orientation, actuators, odometry, and teleop controller."""
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        pos0 = self.data.xpos[self.base_id]
        self.odometry.reset((pos0[0], pos0[1], 0.0))
        self.controller.reset()
        self.teleop.emergency_brake()

    def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """Runs the simulation until duration elapsed or trajectory concludes."""
        # Setup interactive MuJoCo 3D Viewer if GUI requested
        viewer = None
        if self.gui:
            try:
                viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self._on_viewer_key,
                )
                print("=" * 70)
                print("   mecanum_sim: Interactive 3D Simulation Viewer Active")
                print(f"   Robot: {self.robot.name} | Mode: {self.mode.upper()}")
                print("   Controls:")
                print("     - Drive: Arrows or Keypad 8/5/4/6 (Forward/Reverse/Left/Right)")
                print("     - Rotate: Keypad 7/9 or [ / ] (Yaw CCW / CW)")
                print("     - Speed Scale: + / - (Step max speed ±10%, range: 10%–200%)")
                print("     - Driving Mode: M (Cycle: CONTINUOUS -> DIRECT -> CRUISE)")
                print("     - Emergency Brake: B / X / Space")
                print("     - Reset Robot: R / Backspace")
                print("     - Camera View: V (Cycle: Free Orbit -> Tracking -> Front Cam)")
                print("     - Close the viewer window to finish simulation.")
                print("=" * 70)
            except Exception as e:
                print(f"[!] Warning: Unable to launch MuJoCo passive viewer: {e}")

        # Determine simulation duration
        if self.gui and self.user_duration is None and self.mode == "free":
            # Interactive free GUI mode runs indefinitely until the viewer is closed
            total_dur = None
        else:
            total_dur = duration or self.user_duration
            if total_dur is None:
                if self.trajectory is not None:
                    total_dur = self.trajectory.total_duration
                else:
                    total_dur = float(self.cfg.simulation.default_duration)

        # Settle robot on ground
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        # Reset odometry to settled position
        pos0 = self.data.xpos[self.base_id]
        self.odometry.reset((pos0[0], pos0[1], 0.0))

        # Video recording setup
        video_frames: List[Image.Image] = []
        renderer: Optional[mujoco.Renderer] = None
        record_interval = 1.0 / 30.0  # 30 FPS
        last_frame_t = -1.0

        if self.record_video_path:
            try:
                renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception as e:
                print(f"[!] Warning: Renderer unavailable for video recording: {e}")

        last_cam_idx = -1
        while True:
            step_start = time.perf_counter()

            # Check termination conditions
            if viewer is not None and not viewer.is_running():
                break
            if total_dur is not None and self.sim_time >= total_dur:
                break
            if self.has_nan:
                break

            # Poll real-time continuous key state if in CONTINUOUS teleop mode
            if viewer is not None and self.teleop.mode == "CONTINUOUS":
                self.teleop.poll_continuous_keys()

            snapshot = self.step()

            # Sync MuJoCo 3D viewer
            if viewer is not None:
                if self.teleop.active_camera_idx != last_cam_idx:
                    self._update_viewer_camera(viewer)
                    last_cam_idx = self.teleop.active_camera_idx
                viewer.sync()

            # Record video frame
            if renderer is not None and (self.sim_time - last_frame_t >= record_interval):
                last_frame_t = self.sim_time
                try:
                    renderer.update_scene(self.data, camera=self.record_camera)
                    rgb = renderer.render()
                    video_frames.append(Image.fromarray(rgb))
                except Exception:
                    pass

            # In GUI mode, pace simulation to real-time (100 Hz control loop)
            if viewer is not None:
                elapsed = time.perf_counter() - step_start
                sleep_time = self.dt_control - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

        # Save video animation if recorded
        saved_video: Optional[Path] = None
        if video_frames and self.record_video_path:
            saved_video = Path(self.record_video_path)
            saved_video.parent.mkdir(parents=True, exist_ok=True)
            video_frames[0].save(
                saved_video,
                save_all=True,
                append_images=video_frames[1:],
                duration=33,  # ~30 FPS
                loop=0,
            )
            print(f"Saved simulation animation: {saved_video}")

        # Export CSV telemetry
        csv_path: Optional[Path] = None
        if not self.no_csv and self.telemetry_records:
            scenario_name = self.preset or ("drawn_path" if self.waypoints else self.mode)
            csv_path = self.output_dir / f"{self.robot.name}_{scenario_name}_telemetry.csv"
            self.export_csv(csv_path)

        metrics = compute_run_metrics(self.telemetry_records)

        return {
            "status": "completed" if not self.has_nan else "diverged_nan",
            "duration": round(self.sim_time, 3),
            "has_nan": self.has_nan,
            "num_steps": len(self.telemetry_records),
            "csv_path": str(csv_path) if csv_path else None,
            "video_path": str(saved_video) if saved_video else None,
            "metrics": metrics,
        }

    def export_csv(self, file_path: Union[str, Path]) -> Path:
        """Writes accumulated telemetry records to CSV."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.telemetry_records:
            return path

        keys = list(self.telemetry_records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.telemetry_records)
        return path

    # ==========================================================================
    # Zero-Dependency ROS 2 Adapter Methods
    # ==========================================================================

    def step_with_twist(self, twist: TwistCommand) -> Dict[str, Any]:
        """Consumes a ROS geometry_msgs/Twist equivalent and executes one step."""
        cmd = [twist.linear.x, twist.linear.y, twist.angular.z]
        return self.step(cmd)

    def get_ros_clock(self) -> ClockMessage:
        sec = int(self.sim_time)
        nanosec = int((self.sim_time - sec) * 1e9)
        return ClockMessage(sec=sec, nanosec=nanosec)

    def get_ros_odometry(self) -> OdometryState:
        clk = self.get_ros_clock()
        odom_x, odom_y, odom_yaw = self.odometry.pose
        vx, vy, wz = self.odometry.twist
        qz = math.sin(odom_yaw / 2.0)
        qw = math.cos(odom_yaw / 2.0)
        return OdometryState(
            header=Header(stamp_sec=clk.sec, stamp_nanosec=clk.nanosec, frame_id="odom"),
            child_frame_id="base_footprint",
            pose_position=Vector3(x=odom_x, y=odom_y, z=0.0),
            pose_orientation=Quaternion(x=0.0, y=0.0, z=qz, w=qw),
            twist_linear=Vector3(x=vx, y=vy, z=0.0),
            twist_angular=Vector3(x=0.0, y=0.0, z=wz),
        )

    def get_ros_joint_states(self) -> JointStateMessage:
        clk = self.get_ros_clock()
        pos, vel, eff = self.actuator.read_feedback()
        return JointStateMessage(
            header=Header(stamp_sec=clk.sec, stamp_nanosec=clk.nanosec, frame_id="base_link"),
            position=[float(x) for x in pos],
            velocity=[float(x) for x in vel],
            effort=[float(x) for x in eff],
        )

    def get_ros_imu(self) -> ImuStateMessage:
        clk = self.get_ros_clock()
        _, quat, _, _, _ = self._get_chassis_state()
        # Read imu sensors if present
        try:
            accel = self.data.sensor("imu_accel").data
            gyro = self.data.sensor("imu_gyro").data
            a_vec = Vector3(float(accel[0]), float(accel[1]), float(accel[2]))
            w_vec = Vector3(float(gyro[0]), float(gyro[1]), float(gyro[2]))
        except Exception:
            a_vec = Vector3(0.0, 0.0, 9.81)
            w_vec = Vector3(0.0, 0.0, 0.0)

        return ImuStateMessage(
            header=Header(stamp_sec=clk.sec, stamp_nanosec=clk.nanosec, frame_id="imu_link"),
            orientation=Quaternion(x=float(quat[1]), y=float(quat[2]), z=float(quat[3]), w=float(quat[0])),
            angular_velocity=w_vec,
            linear_acceleration=a_vec,
        )

    def get_ros_transforms(self) -> List[TransformMessage]:
        clk = self.get_ros_clock()
        odom_x, odom_y, odom_yaw = self.odometry.pose
        qz = math.sin(odom_yaw / 2.0)
        qw = math.cos(odom_yaw / 2.0)

        # 1. odom -> base_footprint
        tf_odom_footprint = TransformMessage(
            header=Header(stamp_sec=clk.sec, stamp_nanosec=clk.nanosec, frame_id="odom"),
            child_frame_id="base_footprint",
            translation=Vector3(x=odom_x, y=odom_y, z=0.0),
            rotation=Quaternion(x=0.0, y=0.0, z=qz, w=qw),
        )
        # 2. base_footprint -> base_link (at z = wheel_radius)
        tf_footprint_base = TransformMessage(
            header=Header(stamp_sec=clk.sec, stamp_nanosec=clk.nanosec, frame_id="base_footprint"),
            child_frame_id="base_link",
            translation=Vector3(x=0.0, y=0.0, z=self.robot.wheel_radius),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return [tf_odom_footprint, tf_footprint_base]

