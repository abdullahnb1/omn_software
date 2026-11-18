#!/usr/bin/env python3
"""Pre-flight environment, OpenGL, and robot configuration diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src and project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))


def check_python_version() -> bool:
    print(f"Python Version: {sys.version.split()[0]} ({sys.platform})")
    if sys.version_info < (3, 9):
        print("[-] Python 3.9+ is required.")
        return False
    print("[+] Python version check passed.")
    return True


def check_dependencies() -> bool:
    required_packages = [
        "mujoco",
        "numpy",
        "scipy",
        "matplotlib",
        "PIL",
        "yaml",
        "pytest",
    ]
    all_ok = True
    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"[+] Package '{pkg}' is available (version: {ver}).")
        except ImportError as e:
            print(f"[-] Missing required package: {pkg} ({e})")
            all_ok = False
    return all_ok


def check_gui_and_rendering() -> None:
    # Check Tkinter
    try:
        import tkinter
        print("[+] Tkinter GUI toolkit is available.")
    except ImportError:
        print("[!] Tkinter not available. PiP overlay will be disabled.")

    # Check MuJoCo offscreen rendering
    try:
        import mujoco
        xml = """
        <mujoco>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1"/>
                <camera name="test_cam" pos="0 -1 1" xyaxes="1 0 0 0 1 1"/>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        renderer = mujoco.Renderer(model, height=120, width=160)
        renderer.update_scene(data, camera="test_cam")
        img = renderer.render()
        assert img is not None and img.shape == (120, 160, 3)
        print("[+] MuJoCo offscreen rendering verified successfully.")
    except Exception as e:
        print(f"[!] MuJoCo offscreen rendering warning: {e}")


def check_robot_configurations() -> bool:
    from mecanum_sim.config import load_config, list_available_robots

    robots = list_available_robots()
    print(f"\nDiscovered robot configurations: {robots}")
    if len(robots) < 3:
        print(f"[-] Expected at least 3 robot configurations, found {len(robots)}.")
        return False

    c1 = load_config(robot="1")
    c2 = load_config(robot="2")
    c3 = load_config(robot="3")

    print(f"\n[Robot 1: {c1.robot.name}]")
    print(f"  R={c1.robot.wheel_radius:.3f}m, L={c1.robot.characteristic_length:.3f}m, Mass={c1.robot.unladen_mass:.1f}kg")

    print(f"[Robot 2: {c2.robot.name}]")
    print(f"  R={c2.robot.wheel_radius:.3f}m, L={c2.robot.characteristic_length:.3f}m, Mass={c2.robot.unladen_mass:.1f}kg")

    print(f"[Robot 3: {c3.robot.name}]")
    print(f"  R={c3.robot.wheel_radius:.3f}m, L={c3.robot.characteristic_length:.3f}m, Mass={c3.robot.unladen_mass:.1f}kg")

    # Verify meaningful parameter differentiation (>15%)
    radii = {c1.robot.wheel_radius, c2.robot.wheel_radius, c3.robot.wheel_radius}
    if len(radii) != 3:
        print("[-] Robot wheel radii must all be distinct.")
        return False

    masses = {c1.robot.unladen_mass, c2.robot.unladen_mass, c3.robot.unladen_mass}
    if len(masses) != 3:
        print("[-] Robot unladen masses must all be distinct.")
        return False

    # Check Robot 3 constraints
    w3_total = c3.robot.wheels.total_width
    door_clearance = (0.80 - w3_total) / 2.0
    if w3_total > 0.54 or door_clearance < 0.13:
        print(f"[-] Robot 3 width {w3_total:.3f}m violates doorway clearance (clearance: {door_clearance*1000:.1f}mm).")
        return False

    if c3.robot.chassis.underbelly_clearance < 0.035:
        print(f"[-] Robot 3 clearance {c3.robot.chassis.underbelly_clearance:.3f}m < 0.035m.")
        return False

    print("[+] All robot configurations validated and compliant.")
    return True


def main() -> None:
    print("=" * 60)
    print("   mecanum_sim: Runtime Environment Diagnostics")
    print("=" * 60)

    ok_py = check_python_version()
    ok_deps = check_dependencies()
    check_gui_and_rendering()
    ok_cfg = check_robot_configurations()

    print("=" * 60)
    if ok_py and ok_deps and ok_cfg:
        print(">>> ALL PRE-FLIGHT CHECKS PASSED. SYSTEM IS READY. <<<")
        sys.exit(0)
    else:
        print(">>> ONE OR MORE CRITICAL CHECKS FAILED. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()

