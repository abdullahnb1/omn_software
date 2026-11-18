"""Interactive 2D top-down path planner and headless WaypointPlannerCore."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
except Exception:
    matplotlib = None
    plt = None
    Circle = None
    Rectangle = None

from mecanum_sim.config import load_config, RobotConfig
from mecanum_sim.trajectory import (
    CubicSplineTrajectory,
    create_trajectory,
    get_preset_waypoints,
    load_waypoints_json,
    save_waypoints_json,
)


class WaypointPlannerCore:
    """Pure mathematical waypoint planning, spline fitting, and file I/O core (zero GUI dependency)."""

    def __init__(
        self,
        robot: str = "1",
        nominal_speed: float = 0.40,
        arena: str = "auto",
        payload_mass: float = 0.0,
    ):
        self.arena = str(arena)
        self.payload_mass = max(0.0, float(payload_mass))
        self.cfg = load_config(
            robot=robot,
            arena=self.arena if self.arena != "auto" else "open",
            payload_mass=self.payload_mass,
        )
        self.robot = self.cfg.robot
        self.nominal_speed = float(nominal_speed)
        self.waypoints: List[Dict[str, float]] = [
            {"x": 0.0, "y": 0.0, "theta": 0.0}
        ]
        self.spline: Optional[CubicSplineTrajectory] = None

    def select_robot(self, robot: str) -> RobotConfig:
        """Switches active robot profile and updates footprint dimensions."""
        self.cfg = load_config(
            robot=robot,
            arena=self.arena if self.arena != "auto" else "open",
            payload_mass=self.payload_mass,
        )
        self.robot = self.cfg.robot
        return self.robot

    def set_arena(self, arena: str) -> str:
        self.arena = str(arena)
        return self.arena

    def set_payload(self, payload: float) -> float:
        self.payload_mass = max(0.0, float(payload))
        return self.payload_mass

    def set_speed(self, speed: float) -> float:
        self.nominal_speed = max(0.05, float(speed))
        return self.nominal_speed

    def add_waypoint(self, x: float, y: float, theta: float = 0.0) -> None:
        """Appends a new 2D waypoint."""
        self.waypoints.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "theta": round(float(theta), 4),
        })
        self.spline = None

    def remove_last_waypoint(self) -> Optional[Dict[str, float]]:
        """Removes the most recent waypoint (if more than initial point remains)."""
        if len(self.waypoints) > 1:
            removed = self.waypoints.pop()
            self.spline = None
            return removed
        return None

    def clear_waypoints(self, start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Clears all waypoints and resets start point."""
        self.waypoints = [
            {"x": float(start_pos[0]), "y": float(start_pos[1]), "theta": float(start_pos[2])}
        ]
        self.spline = None

    def load_preset(self, preset_name: str) -> List[Dict[str, float]]:
        """Loads canonical preset waypoints."""
        self.waypoints = get_preset_waypoints(preset_name)
        self.spline = None
        return self.waypoints

    def generate_spline(self) -> CubicSplineTrajectory:
        """Fits a smooth cubic spline through the current waypoint set."""
        if len(self.waypoints) < 2:
            raise ValueError("At least 2 waypoints are required to fit a spline trajectory.")
        self.spline = CubicSplineTrajectory(
            waypoints=self.waypoints,
            nominal_speed=self.nominal_speed,
            settle_time=1.0,
        )
        return self.spline

    def save_to_json(self, file_path: Union[str, Path]) -> Path:
        """Exports waypoints and nominal speed to JSON."""
        save_waypoints_json(self.waypoints, file_path, nominal_speed=self.nominal_speed)
        return Path(file_path)

    def load_from_json(self, file_path: Union[str, Path]) -> List[Dict[str, float]]:
        """Imports waypoints and nominal speed from JSON."""
        wps, spd = load_waypoints_json(file_path)
        self.waypoints = wps
        self.nominal_speed = spd
        self.spline = None
        return self.waypoints

    def get_robot_footprint_polygon(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> np.ndarray:
        """Transforms 2D footprint polygon into arena frame for visualization."""
        base_poly = np.array(self.robot.nav2_footprint, dtype=float)  # N x 2
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        transformed = (base_poly @ rot_mat.T) + np.array([x, y])
        return transformed


class DrawPathApp:
    """Interactive 2D top-down path planning UI in Tkinter + Matplotlib."""

    def __init__(
        self,
        robot: str = "1",
        preset: Optional[str] = None,
        waypoints_file: Optional[str] = None,
        speed: float = 0.40,
        arena: str = "auto",
        payload_mass: float = 0.0,
        on_execute_callback: Optional[Any] = None,
    ):
        self.core = WaypointPlannerCore(
            robot=robot,
            nominal_speed=speed,
            arena=arena,
            payload_mass=payload_mass,
        )
        self.on_execute = on_execute_callback
        self.execution_requested = False
        self.execution_params: Dict[str, Any] = {}

        if waypoints_file:
            self.core.load_from_json(waypoints_file)
        elif preset:
            self.core.load_preset(preset)

        self._init_gui()

    def _init_gui(self) -> None:
        try:
            global plt, matplotlib
            import tkinter as tk
            from tkinter import ttk, messagebox, filedialog
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.pyplot as plt

            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox
            self.filedialog = filedialog

            self.root = tk.Tk()
            self.root.title("Mecanum AMR: Interactive 2D Path Planner")
            self.root.geometry("1024x720")

            # Main layout: Left canvas, Right control panel
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Matplotlib Figure
            self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
            self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Connect mouse click events
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

            # Control Panel (Right)
            ctrl_frame = ttk.LabelFrame(main_frame, text="Controls & Presets", padding=10)
            ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

            # 1. Arena Selector (Top of controls & presets)
            ttk.Label(ctrl_frame, text="Select Arena:").pack(anchor=tk.W, pady=(0, 2))
            self.arena_var = tk.StringVar(value=self.core.arena)
            arena_cb = ttk.Combobox(
                ctrl_frame,
                textvariable=self.arena_var,
                values=["auto", "full", "doorway", "open", "ramps"],
                state="readonly",
            )
            arena_cb.pack(fill=tk.X, pady=(0, 10))
            arena_cb.bind("<<ComboboxSelected>>", self._on_arena_changed)

            # 2. Robot Selector
            ttk.Label(ctrl_frame, text="Select Robot Profile:").pack(anchor=tk.W, pady=(0, 2))
            self.robot_var = tk.StringVar(value=self.core.robot.name)
            robot_cb = ttk.Combobox(
                ctrl_frame,
                textvariable=self.robot_var,
                values=["robot_1_compact", "robot_2_heavy", "robot_3_custom"],
                state="readonly",
            )
            robot_cb.pack(fill=tk.X, pady=(0, 10))
            robot_cb.bind("<<ComboboxSelected>>", self._on_robot_changed)

            # 3. Preset Selector
            ttk.Label(ctrl_frame, text="Load Trajectory Preset:").pack(anchor=tk.W, pady=(0, 2))
            self.preset_var = tk.StringVar(value="Select preset...")
            preset_cb = ttk.Combobox(
                ctrl_frame,
                textvariable=self.preset_var,
                values=[
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
                ],
                state="readonly",
            )
            preset_cb.pack(fill=tk.X, pady=(0, 10))
            preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)

            # 4. Cruising Speed Slider
            self.speed_lbl = ttk.Label(ctrl_frame, text=f"Nominal Speed: {self.core.nominal_speed:.2f} m/s")
            self.speed_lbl.pack(anchor=tk.W, pady=(0, 2))
            self.speed_var = tk.DoubleVar(value=self.core.nominal_speed)
            speed_scale = ttk.Scale(
                ctrl_frame,
                from_=0.10,
                to=2.00,
                variable=self.speed_var,
                command=self._on_speed_changed,
            )
            speed_scale.pack(fill=tk.X, pady=(0, 10))

            # 5. Payload Mass Slider
            self.payload_lbl = ttk.Label(ctrl_frame, text=f"Payload Mass: {self.core.payload_mass:.1f} kg")
            self.payload_lbl.pack(anchor=tk.W, pady=(0, 2))
            self.payload_var = tk.DoubleVar(value=self.core.payload_mass)
            payload_scale = ttk.Scale(
                ctrl_frame,
                from_=0.0,
                to=50.0,
                variable=self.payload_var,
                command=self._on_payload_changed,
            )
            payload_scale.pack(fill=tk.X, pady=(0, 10))

            # Buttons
            ttk.Button(ctrl_frame, text="Generate Spline", command=self._generate_spline_btn).pack(fill=tk.X, pady=4)
            ttk.Button(ctrl_frame, text="Save JSON", command=self._save_json_btn).pack(fill=tk.X, pady=4)
            ttk.Button(ctrl_frame, text="Load JSON", command=self._load_json_btn).pack(fill=tk.X, pady=4)
            ttk.Button(ctrl_frame, text="Reset Canvas", command=self._reset_btn).pack(fill=tk.X, pady=4)

            ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

            exec_btn = ttk.Button(ctrl_frame, text="EXECUTE PATH", command=self._execute_btn)
            exec_btn.pack(fill=tk.X, pady=6)

            self.status_lbl = ttk.Label(ctrl_frame, text="LClick: Add Point | RClick: Delete Point", wraplength=150)
            self.status_lbl.pack(fill=tk.X, pady=(15, 0))

            self.root.protocol("WM_DELETE_WINDOW", self._close_window)

            self._draw_canvas()
        except Exception as e:
            print(f"[!] Warning: GUI initialization failed (headless environment): {e}")

    def _draw_canvas(self) -> None:
        """Renders arena obstacles, doorways, thresholds, waypoints, and spline."""
        self.ax.clear()
        self.ax.set_xlim(-4.2, 4.2)
        self.ax.set_ylim(-4.2, 4.2)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_title(f"Top-Down Arena [{self.core.arena}]: {self.core.robot.name}")
        self.ax.set_xlabel("X (m) [Forward]")
        self.ax.set_ylabel("Y (m) [Left]")

        # 1. Perimeter Walls (8x8m) - in all arenas
        self.ax.plot([-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4], "k-", linewidth=2.5, label="Arena Walls")

        arena = self.core.arena.lower()
        # 2. Partition Wall with 800mm doorway at X=0 (doorway, full, and auto)
        if arena in ("doorway", "full", "auto"):
            self.ax.plot([0, 0], [0.4, 4.0], "gray", linewidth=3.5)
            self.ax.plot([0, 0], [-4.0, -0.4], "gray", linewidth=3.5)
            # 8mm Doorway Threshold
            self.ax.plot([0, 0], [-0.4, 0.4], "orange", linewidth=4.0, label="8mm Doorway Threshold")

        # 3. 5mm Open Threshold, Pillars, and Crates (full and auto)
        if arena in ("full", "auto"):
            # 5mm Open Threshold at (-1.5, 2.0)
            self.ax.plot([-1.5, -1.5], [1.5, 2.5], color="darkorange", linewidth=3.0, linestyle=":", label="5mm Threshold")

            # Pillars at (2.2, +-2.0)
            for py in [2.0, -2.0]:
                pillar = Circle((2.2, py), 0.15, color="teal", alpha=0.6) if Circle else plt.Circle((2.2, py), 0.15, color="teal", alpha=0.6)
                self.ax.add_patch(pillar)

            # Crates
            crates = [(-2.5, 2.0), (-2.5, -2.0), (2.5, 0.85), (2.5, -0.85)]
            for cx, cy in crates:
                crate = Rectangle((cx - 0.15, cy - 0.15), 0.30, 0.30, color="saddlebrown", alpha=0.5) if Rectangle else plt.Rectangle((cx - 0.15, cy - 0.15), 0.30, 0.30, color="saddlebrown", alpha=0.5)
                self.ax.add_patch(crate)

        elif arena == "ramps":
            # North Lane: Bridge & Dual Ramps (Y in [1.3, 2.7], centered at Y = 2.0m, width 1.4m)
            # Climb ramp: X in [-2.4, -0.8]
            climb = Rectangle((-2.4, 1.3), 1.6, 1.4, color="slategray", alpha=0.5, label="Climb Ramp (~3°)") if Rectangle else None
            plateau = Rectangle((-0.8, 1.3), 1.6, 1.4, color="darkslateblue", alpha=0.6, label="Bridge Plateau (80mm)") if Rectangle else None
            desc = Rectangle((0.8, 1.3), 1.6, 1.4, color="slategray", alpha=0.5, label="Descent Ramp (~3°)") if Rectangle else None
            for p_patch in [climb, plateau, desc]:
                if p_patch:
                    self.ax.add_patch(p_patch)

            # Plateau Safety Guardrails
            self.ax.plot([-0.8, 0.8], [2.72, 2.72], color="gold", linewidth=3.0, label="Bridge Rails")
            self.ax.plot([-0.8, 0.8], [1.28, 1.28], color="gold", linewidth=3.0)

            # South Lane: Slalom Pylons, Speed Tables, Rumble Strips, Camber Wedge (Y centered at -2.0m)
            for px in [-1.0, 0.0, 1.0]:
                pyl = Circle((px, -2.0), 0.12, color="teal", alpha=0.7) if Circle else None
                if pyl:
                    self.ax.add_patch(pyl)
            ht_w = Rectangle((-2.75, -2.6), 0.7, 1.2, color="darkorange", alpha=0.4, label="Speed Tables") if Rectangle else None
            ht_e = Rectangle((2.05, -2.6), 0.7, 1.2, color="darkorange", alpha=0.4) if Rectangle else None
            if ht_w:
                self.ax.add_patch(ht_w)
            if ht_e:
                self.ax.add_patch(ht_e)

            for rx in [-0.50, -0.25]:
                self.ax.plot([rx, rx], [-2.6, -1.4], color="crimson", linewidth=2.0)
            wedge_patch = Rectangle((0.3, -2.5), 1.4, 1.0, color="purple", alpha=0.35, label="Banked Camber (3.4°)") if Rectangle else None
            if wedge_patch:
                self.ax.add_patch(wedge_patch)

            # Center Runway indicator (Y in [-1.0, 1.0])
            self.ax.plot([-4.0, 4.0], [1.0, 1.0], "gray", linestyle=":", alpha=0.4)
            self.ax.plot([-4.0, 4.0], [-1.0, -1.0], "gray", linestyle=":", alpha=0.4)

        # 4. Origin Marker
        self.ax.plot(0, 0, "go", markersize=8, label="Origin (0,0)")

        # 5. Draw Waypoints
        if self.core.waypoints:
            xs = [wp["x"] for wp in self.core.waypoints]
            ys = [wp["y"] for wp in self.core.waypoints]
            self.ax.plot(xs, ys, "b--o", markersize=6, label="Waypoints")

            # Draw Robot Footprint at initial waypoint
            poly = self.core.get_robot_footprint_polygon(xs[0], ys[0], self.core.waypoints[0].get("theta", 0.0))
            poly_closed = np.vstack([poly, poly[0]])
            self.ax.plot(poly_closed[:, 0], poly_closed[:, 1], "g-", linewidth=2.0, label="Robot Footprint")

        # 6. Draw Spline if generated
        if self.core.spline is not None:
            ts = np.linspace(0.0, self.core.spline.total_duration, 200)
            spline_pts = [self.core.spline.evaluate(t) for t in ts]
            s_xs = [p.x for p in spline_pts]
            s_ys = [p.y for p in spline_pts]
            self.ax.plot(s_xs, s_ys, "r-", linewidth=2.0, label="Fitted Spline")

        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def _on_canvas_click(self, event: Any) -> None:
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left Click: Add WP
            self.core.add_waypoint(event.xdata, event.ydata)
            self.status_lbl.config(text=f"Added WP: ({event.xdata:.2f}, {event.ydata:.2f})")
            self._draw_canvas()
        elif event.button == 3:  # Right Click: Remove WP
            rem = self.core.remove_last_waypoint()
            if rem:
                self.status_lbl.config(text=f"Removed WP: ({rem['x']:.2f}, {rem['y']:.2f})")
            self._draw_canvas()

    def _on_arena_changed(self, event: Any) -> None:
        self.core.set_arena(self.arena_var.get())
        self._draw_canvas()

    def _on_speed_changed(self, val: Any) -> None:
        spd = float(val)
        self.core.set_speed(spd)
        self.speed_lbl.config(text=f"Nominal Speed: {spd:.2f} m/s")

    def _on_payload_changed(self, val: Any) -> None:
        pl = round(float(val), 1)
        self.core.set_payload(pl)
        self.payload_lbl.config(text=f"Payload Mass: {pl:.1f} kg")

    def _on_robot_changed(self, event: Any) -> None:
        r_name = self.robot_var.get()
        self.core.select_robot(r_name)
        self._draw_canvas()

    def _on_preset_selected(self, event: Any) -> None:
        p_name = self.preset_var.get()
        self.core.load_preset(p_name)
        self.core.generate_spline()
        self._draw_canvas()

    def _generate_spline_btn(self) -> None:
        try:
            self.core.generate_spline()
            self.status_lbl.config(text=f"Spline generated! Dur: {self.core.spline.total_duration:.1f}s")
            self._draw_canvas()
        except Exception as e:
            self.messagebox.showerror("Spline Error", str(e))

    def _save_json_btn(self) -> None:
        f = self.filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if f:
            self.core.save_to_json(f)
            self.status_lbl.config(text=f"Saved path to {Path(f).name}")

    def _load_json_btn(self) -> None:
        f = self.filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if f:
            self.core.load_from_json(f)
            self.speed_var.set(self.core.nominal_speed)
            self.speed_lbl.config(text=f"Nominal Speed: {self.core.nominal_speed:.2f} m/s")
            self.core.generate_spline()
            self._draw_canvas()

    def _reset_btn(self) -> None:
        self.core.clear_waypoints()
        self._draw_canvas()

    def _close_window(self) -> None:
        try:
            if hasattr(self, "root") and self.root:
                self.root.quit()
                self.root.destroy()
                self.root = None
        except Exception:
            pass

    def _execute_btn(self) -> None:
        try:
            if len(self.core.waypoints) < 2:
                self.messagebox.showwarning(
                    "Execution Warning",
                    "At least 2 waypoints are required to execute a path.\nLeft-click on the arena canvas to add waypoints.",
                )
                return
            if self.core.spline is None:
                self.core.generate_spline()
            self.execution_params = {
                "robot": self.robot_var.get(),
                "arena": self.arena_var.get(),
                "waypoints": [dict(wp) for wp in self.core.waypoints],
                "speed": float(self.core.nominal_speed),
                "payload_mass": float(self.core.payload_mass),
            }
            self.execution_requested = True
            if self.on_execute:
                self.on_execute(self.core.waypoints, self.core.nominal_speed)
            self._close_window()
        except Exception as e:
            self.messagebox.showerror("Execution Error", str(e))

    def start(self) -> None:
        if hasattr(self, "root") and self.root:
            self.root.mainloop()


def run_draw_mode(
    robot: str = "1",
    preset: Optional[str] = None,
    waypoints_file: Optional[str] = None,
    speed: float = 0.40,
    arena: str = "auto",
    payload_mass: float = 0.0,
    non_interactive: bool = False,
    output_dir: str = "results",
    **sim_kwargs: Any,
) -> Dict[str, Any]:
    """Runs draw mode interactively or executes headlessly when non_interactive=True."""
    from mecanum_sim.simulation import MecanumSimulation

    core = WaypointPlannerCore(
        robot=robot,
        nominal_speed=speed,
        arena=arena,
        payload_mass=payload_mass,
    )
    if waypoints_file:
        core.load_from_json(waypoints_file)
    elif preset:
        core.load_preset(preset)
    elif non_interactive:
        # Default doorway preset if none specified in non-interactive mode
        core.load_preset("doorway_and_threshold")

    if len(core.waypoints) >= 2:
        core.generate_spline()

    if non_interactive:
        print(f"[Draw Mode] Running non-interactively with {len(core.waypoints)} waypoints...")
        sim = MecanumSimulation(
            robot=robot,
            arena=arena,
            mode="path",
            preset=preset if preset else None,
            waypoints=core.waypoints if preset is None else None,
            speed=speed,
            payload_mass=payload_mass,
            output_dir=output_dir,
            gui=False,
            **sim_kwargs,
        )
        return sim.run()

    # Launch GUI
    app = DrawPathApp(
        robot=robot,
        preset=preset,
        waypoints_file=waypoints_file,
        speed=speed,
        arena=arena,
        payload_mass=payload_mass,
    )
    app.start()

    if app.execution_requested:
        p = app.execution_params
        print(
            f"[Draw Mode] Executing path with {len(p['waypoints'])} waypoints in arena '{p['arena']}' "
            f"(robot: {p['robot']}, payload: {p['payload_mass']} kg, speed: {p['speed']:.2f} m/s)..."
        )
        sim = MecanumSimulation(
            robot=p["robot"],
            arena=p["arena"],
            mode="path",
            preset=None,
            waypoints=p["waypoints"],
            speed=p["speed"],
            payload_mass=p["payload_mass"],
            output_dir=output_dir,
            gui=True,
            **sim_kwargs,
        )
        return sim.run()

    return {}

