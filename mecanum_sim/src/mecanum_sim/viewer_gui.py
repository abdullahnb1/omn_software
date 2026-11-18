"""Interactive 3D MuJoCo viewer and teleoperation controller."""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from mecanum_sim.config import RobotConfig


class FreeTeleopController:
    """Headless keyboard teleoperation controller supporting Direct Drive and Cruise modes."""

    def __init__(self, robot_config: RobotConfig):
        self.robot = robot_config
        self.mode = "CONTINUOUS"  # Default: CONTINUOUS (or "DIRECT", "CRUISE")

        lim = self.robot.limits
        self.max_vx = float(lim.max_vx)
        self.max_vy = float(lim.max_vy)
        self.max_wz = float(lim.max_wz)
        self.accel = float(lim.max_ax)
        self.alpha = float(lim.max_awz)

        self.a_decay = 1.5   # m/s^2 natural deceleration on release
        self.a_brake = 3.0   # m/s^2 active counter-steering brake

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.speed_scale: float = 1.0  # Dynamic speed throttle scale (0.10 to 2.00, ±10% step)

        # Key press states
        self.keys_pressed: Dict[str, bool] = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "rot_ccw": False,
            "rot_cw": False,
        }
        self.reset_requested = False
        self.active_camera_idx = 0  # 0: Free, 1: track_camera, 2: front_camera
        self._key_release_times: Dict[str, float] = {}

    def toggle_mode(self) -> str:
        """Toggles between CONTINUOUS, DIRECT, and CRUISE driving modes."""
        modes = ["CONTINUOUS", "DIRECT", "CRUISE"]
        current_idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(current_idx + 1) % len(modes)]
        return self.mode

    def emergency_brake(self) -> None:
        """Instantly zeroes commanded velocities."""
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

    def cycle_camera(self) -> int:
        self.active_camera_idx = (self.active_camera_idx + 1) % 3
        return self.active_camera_idx

    def step_speed_scale(self, delta: float) -> float:
        """Adjusts the dynamic speed scale by delta (+0.1 or -0.1), clamped between 0.10 and 2.00."""
        self.speed_scale = round(max(0.10, min(2.00, self.speed_scale + delta)), 2)
        eff_vx = self.max_vx * self.speed_scale
        eff_vy = self.max_vy * self.speed_scale
        eff_wz = self.max_wz * self.speed_scale
        self.cmd_vx = max(-eff_vx, min(eff_vx, self.cmd_vx))
        self.cmd_vy = max(-eff_vy, min(eff_vy, self.cmd_vy))
        self.cmd_wz = max(-eff_wz, min(eff_wz, self.cmd_wz))
        return self.speed_scale

    def poll_continuous_keys(self) -> None:
        """Polls physical keydown/keyup states for true continuous press-and-hold teleoperation."""
        try:
            import ctypes
            user32 = ctypes.windll.user32

            def is_down(vks: list[int]) -> bool:
                return any(bool(user32.GetAsyncKeyState(vk) & 0x8000) for vk in vks)

            # Translation: Up (arrows / Keypad 8), Down (arrows / Keypad 5), Left (Keypad 4), Right (Keypad 6)
            self.keys_pressed["up"] = is_down([0x26, 0x68, 0x38])
            self.keys_pressed["down"] = is_down([0x28, 0x65, 0x35])
            self.keys_pressed["left"] = is_down([0x25, 0x64, 0x34])
            self.keys_pressed["right"] = is_down([0x27, 0x66, 0x36])

            # Rotation: Keypad 7, Keypad 9, '[', ']', PageUp, PageDown, ',', '.'
            self.keys_pressed["rot_ccw"] = is_down([0x67, 0x37, 0xDB, 0x21, 0xBC])
            self.keys_pressed["rot_cw"] = is_down([0x69, 0x39, 0xDD, 0x22, 0xBE])
        except Exception:
            pass

    def handle_viewer_key(self, key_name: str) -> None:
        """Processes keyboard events from interactive viewer with key repeat / auto-expiration."""
        k = key_name.lower()
        # WASD and QE are explicitly removed to prevent conflict with MuJoCo viewer navigation
        if k in ["w", "a", "s", "d", "q", "e"]:
            return

        now = time.time()
        hold_time = 0.18
        eff_max_vx = self.max_vx * self.speed_scale
        eff_max_vy = self.max_vy * self.speed_scale
        eff_max_wz = self.max_wz * self.speed_scale

        if k in ["up", "8"]:
            if self.mode == "CRUISE":
                self.cmd_vx = min(eff_max_vx, self.cmd_vx + 0.1 * eff_max_vx)
            else:
                self.keys_pressed["up"] = True
                self._key_release_times["up"] = now + hold_time
        elif k in ["down", "5"]:
            if self.mode == "CRUISE":
                self.cmd_vx = max(-eff_max_vx, self.cmd_vx - 0.1 * eff_max_vx)
            else:
                self.keys_pressed["down"] = True
                self._key_release_times["down"] = now + hold_time
        elif k in ["left", "4"]:
            if self.mode == "CRUISE":
                self.cmd_vy = min(eff_max_vy, self.cmd_vy + 0.1 * eff_max_vy)
            else:
                self.keys_pressed["left"] = True
                self._key_release_times["left"] = now + hold_time
        elif k in ["right", "6"]:
            if self.mode == "CRUISE":
                self.cmd_vy = max(-eff_max_vy, self.cmd_vy - 0.1 * eff_max_vy)
            else:
                self.keys_pressed["right"] = True
                self._key_release_times["right"] = now + hold_time
        elif k in ["[", ",", "prior", "pageup", "7"]:
            if self.mode == "CRUISE":
                self.cmd_wz = min(eff_max_wz, self.cmd_wz + 0.1 * eff_max_wz)
            else:
                self.keys_pressed["rot_ccw"] = True
                self._key_release_times["rot_ccw"] = now + hold_time
        elif k in ["]", ".", "next", "pagedown", "9"]:
            if self.mode == "CRUISE":
                self.cmd_wz = max(-eff_max_wz, self.cmd_wz - 0.1 * eff_max_wz)
            else:
                self.keys_pressed["rot_cw"] = True
                self._key_release_times["rot_cw"] = now + hold_time
        elif k in ["+", "="]:
            scale = self.step_speed_scale(+0.1)
            pct = int(round(scale * 100))
            print(f"[Teleop] Speed Scale: {pct}% | Max: vx={self.max_vx * scale:.2f} m/s, vy={self.max_vy * scale:.2f} m/s, wz={self.max_wz * scale:.2f} rad/s")
        elif k in ["-", "_"]:
            scale = self.step_speed_scale(-0.1)
            pct = int(round(scale * 100))
            print(f"[Teleop] Speed Scale: {pct}% | Max: vx={self.max_vx * scale:.2f} m/s, vy={self.max_vy * scale:.2f} m/s, wz={self.max_wz * scale:.2f} rad/s")
        elif k in ["m"]:
            self.toggle_mode()
            print(f"[Teleop] Mode switched to: {self.mode}")
        elif k in ["b", "x", "space"]:
            self.emergency_brake()
            print("[Teleop] Emergency Brake engaged!")
        elif k in ["r", "backspace"]:
            self.reset_requested = True
            print("[Teleop] Reset requested.")
        elif k in ["v"]:
            cam_idx = self.cycle_camera()
            cams = ["Free Orbit Camera", "Chassis Tracking Camera", "Front Onboard Camera"]
            print(f"[Teleop] Active View: {cams[cam_idx]}")

    def handle_key_down(self, key_name: str) -> None:
        k = key_name.lower()
        if k in ["w", "a", "s", "d", "q", "e"]:
            return
        if k in ["up", "8"]:
            self.keys_pressed["up"] = True
        elif k in ["down", "5"]:
            self.keys_pressed["down"] = True
        elif k in ["left", "4"]:
            self.keys_pressed["left"] = True
        elif k in ["right", "6"]:
            self.keys_pressed["right"] = True
        elif k in ["[", ",", "prior", "pageup", "7"]:
            self.keys_pressed["rot_ccw"] = True
        elif k in ["]", ".", "next", "pagedown", "9"]:
            self.keys_pressed["rot_cw"] = True
        elif k in ["m"]:
            self.toggle_mode()
        elif k in ["b", "x"]:
            self.emergency_brake()
        elif k in ["r", "backspace"]:
            self.reset_requested = True
        elif k in ["v"]:
            self.cycle_camera()
        elif k in ["+", "="]:
            self.step_speed_scale(+0.1)
        elif k in ["-", "_"]:
            self.step_speed_scale(-0.1)

    def handle_key_up(self, key_name: str) -> None:
        k = key_name.lower()
        if k in ["w", "a", "s", "d", "q", "e"]:
            return
        if k in ["up", "8"]:
            self.keys_pressed["up"] = False
        elif k in ["down", "5"]:
            self.keys_pressed["down"] = False
        elif k in ["left", "4"]:
            self.keys_pressed["left"] = False
        elif k in ["right", "6"]:
            self.keys_pressed["right"] = False
        elif k in ["[", ",", "prior", "pageup", "7"]:
            self.keys_pressed["rot_ccw"] = False
        elif k in ["]", ".", "next", "pagedown", "9"]:
            self.keys_pressed["rot_cw"] = False

    def update(self, dt: float) -> np.ndarray:
        """Updates internal commanded velocities and returns [cmd_vx, cmd_vy, cmd_wz]."""
        if self.mode != "CONTINUOUS" and self._key_release_times:
            now = time.time()
            for key, release_t in list(self._key_release_times.items()):
                if now > release_t:
                    self.keys_pressed[key] = False
                    del self._key_release_times[key]
        if dt <= 0:
            return np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz], dtype=float)

        # Target direction intents
        target_dir_x = (1.0 if self.keys_pressed["up"] else 0.0) - (1.0 if self.keys_pressed["down"] else 0.0)
        target_dir_y = (1.0 if self.keys_pressed["left"] else 0.0) - (1.0 if self.keys_pressed["right"] else 0.0)
        target_dir_w = (1.0 if self.keys_pressed["rot_ccw"] else 0.0) - (1.0 if self.keys_pressed["rot_cw"] else 0.0)

        def step_axis(current: float, target_dir: float, max_val: float, a_accel: float) -> float:
            if target_dir > 0:
                rate = self.a_brake if current < 0 else a_accel
                return min(max_val, current + rate * dt)
            elif target_dir < 0:
                rate = self.a_brake if current > 0 else a_accel
                return max(-max_val, current - rate * dt)
            else:
                # Key released
                if self.mode in ["DIRECT", "CONTINUOUS"]:
                    # Smooth natural decay to zero
                    if current > 0:
                        return max(0.0, current - self.a_decay * dt)
                    elif current < 0:
                        return min(0.0, current + self.a_decay * dt)
                # CRUISE mode holds current velocity
                return current

        eff_max_vx = self.max_vx * self.speed_scale
        eff_max_vy = self.max_vy * self.speed_scale
        eff_max_wz = self.max_wz * self.speed_scale

        self.cmd_vx = step_axis(self.cmd_vx, target_dir_x, eff_max_vx, self.accel)
        self.cmd_vy = step_axis(self.cmd_vy, target_dir_y, eff_max_vy, self.accel)
        self.cmd_wz = step_axis(self.cmd_wz, target_dir_w, eff_max_wz, self.alpha)

        return np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz], dtype=float)




