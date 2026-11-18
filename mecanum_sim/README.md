# `mecanum_sim`: 4-Wheel Mecanum Simulation Platform

`mecanum_sim` is a high-fidelity, modular, model-agnostic robotics simulation platform developed in MuJoCo for 4-wheel mecanum Autonomous Mobile Robots (AMRs). Built for payload logistics, domestic utility, and autonomous navigation research, the platform implements physics-grounded multibody contact dynamics, velocity-controlled servo actuation, direction-preserving holonomic kinematics, native ROS 2 compatibility, and embedded firmware C-header generation for real-world MCU deployment (STM32/ESP32).

---

## 1. Architectural Highlights

- **Model-Agnostic Parametric Multi-Robot Engine**: Complete procedural synthesis of robot MJCF models from YAML configuration files. Ships with three distinct, selectable 4-mecanum-wheel robots covering diverse payload, footprint, and dynamic envelopes.
- **True Multibody Contact Physics with Velocity Servos**: Traction is generated entirely through physical ground contact with passive $45^\circ$ crowned barrel rollers mounted on 4 independent wheel hubs. All drive joints use MuJoCo `<velocity>` servos enforcing torque limits via `forcerange`, rejecting ungrounded force injection or teleportation.
- **Direction-Preserving Wheel Velocity Desaturation**: Preserves the commanded instantaneous heading vector $[v_x, v_y, \omega_z]$ during motor saturation by uniform scaling, preventing uncommanded holonomic veering.
- **Decoupled True SE(2) Dead-Reckoned Odometry**: `WheelOdometryEstimator` accumulates encoder ticks via midpoint SE(2) integration, strictly decoupled from MuJoCo ground truth, allowing authentic slip and drift evaluation.
- **Modular Scene Composition via `<include>`**: Standalone robot bodies (`models/robots/`) and arena scenes (`models/scenes/`) are wrapped in `<mujocoinclude>` and composed cleanly via native MuJoCo `<include>`.
- **Onboard Front-Facing Camera & 3D Video Rendering**: Features an upright onboard nose camera (`front_camera`) streamed live in an interactive Picture-in-Picture (PiP) window during GUI runs, and offscreen 30 FPS animation export (`--record-video`).
- **Zero-Dependency ROS 2 Architecture**: Pure-Python standard library dataclasses strictly conforming to REP-103 (SI units) and REP-105 (coordinate frame conventions) without requiring a ROS 2 installation.

---

## 2. Multi-Robot Platform Specifications

The platform ships with three distinct, physically realistic 4-mecanum-wheel robots in canonical **O-configuration**:

| Specification Metric | Robot 1 (`robot_1_compact`) | Robot 2 (`robot_2_heavy`) | Robot 3 (`robot_3_custom`) |
|---|---|---|---|
| **Role & Domain** | Domestic / Educational AMR | Heavy Industrial Logistics Carrier | Agile Service / Inspection AMR |
| **Unladen Mass ($m_{\text{total}}$)** | $6.000\text{ kg}$ | $20.800\text{ kg}$ | $12.000\text{ kg}$ |
| **Chassis Core Mass** | $5.000\text{ kg}$ | $16.000\text{ kg}$ | $9.600\text{ kg}$ |
| **Wheel Assembly Mass ($m_w$)** | $0.250\text{ kg}$ ($\times 4 = 1.0\text{ kg}$) | $1.200\text{ kg}$ ($\times 4 = 4.8\text{ kg}$) | $0.600\text{ kg}$ ($\times 4 = 2.4\text{ kg}$) |
| **Payload Capacity** | $0 - 20\text{ kg}$ | $0 - 30\text{ kg}$ | $0 - 25\text{ kg}$ |
| **Chassis Dimensions ($L \times W \times H$)** | $0.36 \times 0.32 \times 0.08\text{ m}$ | $0.58 \times 0.42 \times 0.11\text{ m}$ | $0.46 \times 0.26 \times 0.09\text{ m}$ |
| **Underbelly Ground Clearance** | $0.030\text{ m}$ ($30\text{ mm}$) | $0.050\text{ m}$ ($50\text{ mm}$) | $0.040\text{ m}$ ($40\text{ mm}$) |
| **Wheel Outer Radius ($R$)** | $0.050\text{ m}$ ($50\text{ mm}$) | $0.080\text{ m}$ ($80\text{ mm}$) | $0.065\text{ m}$ ($65\text{ mm}$) |
| **Wheel Width ($w_w$)** | $0.035\text{ m}$ ($35\text{ mm}$) | $0.055\text{ m}$ ($55\text{ mm}$) | $0.045\text{ m}$ ($45\text{ mm}$) |
| **Half-Base / Half-Track ($L_x, L_y$)** | $L_x=0.12\text{ m}, L_y=0.12\text{ m}$ | $L_x=0.22\text{ m}, L_y=0.18\text{ m}$ | $L_x=0.17\text{ m}, L_y=0.15\text{ m}$ |
| **Characteristic Length ($L = L_x+L_y$)**| $0.240\text{ m}$ | $0.400\text{ m}$ | $0.320\text{ m}$ |
| **Total Stance Width ($2L_y + w_w$)** | $0.275\text{ m}$ | $0.415\text{ m}$ | $0.345\text{ m}$ |
| **Doorway Clearance ($800\text{ mm}$ door)**| $262.5\text{ mm}$ / side | $192.5\text{ mm}$ / side | $227.5\text{ mm}$ / side ($\ge 130\text{ mm}$) |
| **Rollers per Wheel ($N$)** | 8 ($32$ rollers total) | 12 ($48$ rollers total) | 10 ($40$ rollers total) |
| **Roller Radius ($r_{\text{roller}}$)** | $0.012\text{ m}$ | $0.014\text{ m}$ | $0.013\text{ m}$ |
| **Pitch Radius ($R_p = R - r_{\text{roller}}$)**| $0.038\text{ m}$ | $0.066\text{ m}$ | $0.052\text{ m}$ |
| **Max Wheel Speed ($\omega_{\max}$)** | $35.0\text{ rad/s}$ | $25.0\text{ rad/s}$ | $30.0\text{ rad/s}$ |
| **Motor Peak Torque Saturation ($\tau_{\max}$)**| $2.50\text{ N}\cdot\text{m}$ | $10.00\text{ N}\cdot\text{m}$ | $6.00\text{ N}\cdot\text{m}$ |
| **Velocity Servo Gain ($k_v$)** | $0.80\text{ N}\cdot\text{m}/(\text{rad/s})$ | $2.50\text{ N}\cdot\text{m}/(\text{rad/s})$ | $1.60\text{ N}\cdot\text{m}/(\text{rad/s})$ |
| **Nominal Cruising Speed ($v_{\text{nom}}$)**| $0.40\text{ m/s}$ | $0.35\text{ m/s}$ | $0.38\text{ m/s}$ |
| **Max Linear Velocity ($v_{\max}$)** | $0.80\text{ m/s}$ | $0.70\text{ m/s}$ | $0.75\text{ m/s}$ |
| **Max Angular Velocity ($\omega_{z,\max}$)**| $1.50\text{ rad/s}$ | $1.20\text{ rad/s}$ | $1.35\text{ rad/s}$ |
| **Max Linear Accel ($a_{\max}$)** | $1.50\text{ m/s}^2$ | $1.00\text{ m/s}^2$ | $1.25\text{ m/s}^2$ |
| **Max Angular Accel ($\alpha_{z,\max}$)**| $3.00\text{ rad/s}^2$ | $2.00\text{ rad/s}^2$ | $2.50\text{ rad/s}^2$ |
| **Visual Color Scheme** | Royal Blue & Crimson | Charcoal & Safety Orange | Emerald Teal & Gold |

---

## 3. Kinematics, Control, and Physics Formulation

### 3.1 Inverse & Forward Kinematics
For wheel outer radius $R$, half-base $L_x$, and half-track $L_y$ ($L = L_x + L_y$):

$$\begin{bmatrix} \omega_{fl} \\ \omega_{fr} \\ \omega_{rl} \\ \omega_{rr} \end{bmatrix} = \frac{1}{R} \begin{bmatrix} 1 & -1 & -L \\ 1 & 1 & L \\ 1 & 1 & -L \\ 1 & -1 & L \end{bmatrix} \begin{bmatrix} v_x \\ v_y \\ \omega_z \end{bmatrix} = J_{\text{inv}} \mathbf{v}$$

$$\begin{bmatrix} v_x \\ v_y \\ \omega_z \end{bmatrix} = \frac{R}{4} \begin{bmatrix} 1 & 1 & 1 & 1 \\ -1 & 1 & 1 & -1 \\ -\frac{1}{L} & \frac{1}{L} & -\frac{1}{L} & \frac{1}{L} \end{bmatrix} \begin{bmatrix} \omega_{fl} \\ \omega_{fr} \\ \omega_{rl} \\ \omega_{rr} \end{bmatrix} = J_{\text{fwd}} \mathbf{\omega}$$

Exact algebraic inversion satisfies $J_{\text{fwd}} \cdot J_{\text{inv}} = I_{3\times 3}$.

### 3.2 Direction-Preserving Desaturation
When combined commands exceed the motor angular velocity bound $\omega_{\text{wheel},\max}$, independent per-wheel clipping alters the wheel speed ratios, causing steering drift. The system uniformly desaturates:
$$\omega_{\text{peak}} = \max_i |\omega_i|,\quad \mathbf{\omega}_{\text{desat}} = \begin{cases} \mathbf{\omega} \cdot \dfrac{\omega_{\text{wheel},\max}}{\omega_{\text{peak}}} & \text{if } \omega_{\text{peak}} > \omega_{\text{wheel},\max} \\ \mathbf{\omega} & \text{otherwise} \end{cases}$$

### 3.3 Two-Layer Decoupled Velocity Control Loop
1. **Outer Loop**: Combines feedforward spline velocity with proportional body-frame pose correction:
   $$\begin{bmatrix} e_x \\ e_y \end{bmatrix}_{\text{body}} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_d - x \\ y_d - y \end{bmatrix}_{\text{world}},\quad \mathbf{v}_{\text{cmd}} = \mathbf{v}_{\text{ff}} + \mathbf{K}_p \mathbf{e}_{\text{body}}$$
2. **Chassis Rate Limiter**: Clamps linear and angular acceleration:
   $$v_{k} = v_{k-1} + \text{clip}(v_{\text{target}} - v_{k-1}, -a_{\max}\Delta t, +a_{\max}\Delta t)$$
3. **Inner Loop**: Wheel angular acceleration slew limiting ($\alpha_{\text{wheel},\max}$) with anti-windup PI velocity trim:
   $$\omega_{\text{slew},i}(k) = \omega_{\text{slew},i}(k-1) + \text{clip}(\omega_{d,i}(k) - \omega_{\text{slew},i}(k-1), -\alpha_{\max}\Delta t, +\alpha_{\max}\Delta t)$$
   $$\omega_{\text{cmd},i} = \text{clip}\left(\omega_{\text{slew},i} + K_p e_{\omega,i} + K_i \int e_{\omega,i} dt, -\omega_{\max}, +\omega_{\max}\right)$$

---

## 4. Repository Structure

```text
mecanum_sim/
├── README.md                           # Master system documentation, multi-robot specs & CLI guide
├── pyproject.toml                      # Modern packaging, metadata, dependencies & console scripts
├── .gitignore                          # Standard clean git exclusions
├── docs/
│   └── adr_001_architecture_and_robot3.md # Architectural Decision Record & Robot 3 design specs
├── configs/
│   ├── default_config.yaml             # Global simulation, arena & timing configuration
│   └── robots/                         # Authoritative per-robot configurations:
│       ├── robot_1_compact.yaml        # Robot 1: Compact educational baseline
│       ├── robot_2_heavy.yaml          # Robot 2: Heavy logistics carrier
│       └── robot_3_custom.yaml         # Robot 3: Agent-designed agile service AMR
├── models/
│   ├── generate_model.py               # Parametric MJCF procedural generator
│   ├── robots/                         # Standalone robot body XML submodels (<mujocoinclude>):
│   │   ├── robot_1_compact_body.xml
│   │   ├── robot_2_heavy_body.xml
│   │   └── robot_3_custom_body.xml
│   ├── scenes/                         # Modular arena submodels (<mujocoinclude>):
│   │   ├── arena_open.xml              # Unobstructed 8x8m arena
│   │   ├── arena_doorway.xml           # 800mm doorway & 8mm threshold
│   │   └── arena_full.xml              # Doorway, 8mm & 5mm thresholds, pillars, crates, cables
│   ├── robot_1_compact.xml             # Composed full arena scene for Robot 1
│   ├── robot_1_doorway.xml             # Composed doorway scene for Robot 1
│   ├── robot_1_open.xml                # Composed open arena scene for Robot 1
│   ├── robot_2_heavy.xml               # Composed full arena scene for Robot 2
│   ├── robot_2_doorway.xml             # Composed doorway scene for Robot 2
│   ├── robot_2_open.xml                # Composed open arena scene for Robot 2
│   ├── robot_3_custom.xml              # Composed full arena scene for Robot 3
│   ├── robot_3_doorway.xml             # Composed doorway scene for Robot 3
│   └── robot_3_open.xml                # Composed open arena scene for Robot 3
├── src/
│   └── mecanum_sim/
│       ├── __init__.py                 # Package version and export definitions
│       ├── config.py                   # Type-safe dataclasses, configuration loader & inertia math
│       ├── kinematics.py               # Kinematics, desaturation & SE(2) odometry estimator
│       ├── controller.py               # Rate limiter, slew limiter & wheel velocity controllers
│       ├── interfaces.py               # ROS 2 message dataclasses & binary serial bridge
│       ├── trajectory.py               # Cubic spline generator, JSON I/O & 10 presets
│       ├── path_tracker.py             # Outer-loop body-frame pose tracker
│       ├── simulation.py               # Multi-rate simulation engine & telemetry logger
│       ├── metrics.py                  # Tracking RMSE, slip velocity, and energy evaluation
│       ├── viewer_gui.py               # 3D MuJoCo viewer & teleoperation controller
│       ├── drawpath_ui.py              # Interactive 2D path planner & WaypointPlannerCore
│       └── cli.py                      # Console script entry point
├── scripts/
│   ├── check_env.py                    # Pre-flight environment & robot diagnostics
│   ├── run_sim.py                      # Master canonical CLI launcher
│   ├── run_all_scenarios.py            # Automated batch benchmark runner
│   ├── generate_plots.py               # Single & cross-robot dashboard generator
│   └── export_mcu_header.py            # Exports C/C++ firmware headers for STM32/ESP32
├── tests/
│   ├── test_kinematics.py              # Kinematics identity, sign checks & SE(2) odometry
│   ├── test_model.py                   # MJCF compilation, velocity servos & closed-loop signs
│   ├── test_controller.py              # Slew limits, anti-windup & binary CRC16 serial frames
│   ├── test_spline_paths.py            # C^1 continuity, 10 presets & JSON waypoint save/load
│   ├── test_teleop.py                  # Direct, Cruise, Continuous modes, Keypad & brake
│   ├── test_drawpath_ui.py             # WaypointPlannerCore & non-interactive execution
│   ├── test_arena_obstacles.py         # Doorway clearances & 8mm / 5mm ground thresholds
│   ├── test_simulation.py              # Multi-rate stepping, camera video & CSV logging
│   └── test_ros_interfaces.py          # Zero-dependency ROS 2 schemas & MCU header export
└── results/                            # Telemetry CSVs, PNG dashboards, animations & .h headers
```

---

## 5. Master CLI Usage & Execution Modes

`scripts/run_sim.py` (or console script `mecanum-sim`) is the central entry point.

### Common CLI Options:
```text
--robot <1|2|3|all|compact|heavy|custom>  Robot model to simulate (default: 1)
--arena <auto|open|doorway|full|ramps>    Arena environment (default: auto)
--mode <free|path|draw>                   Operation mode (default: free)
--gui                                     Launch 3D viewer (press 'V' to cycle cameras)
--preset <name>                           Predefined benchmark trajectory name
--waypoints-file <path.json>              Path to JSON waypoints file
--speed <m/s>                             Nominal cruising speed override
--duration <seconds>                      Simulation duration override
--payload-mass <kg>                       External payload mass (default: 0.0)
--payload-offset <x y z>                  Payload offset relative to deck (meters)
--floor-friction <float>                  Floor friction coefficient (default: 0.8)
--no-pose-correction                      Disable outer-loop pose correction
--record-video <path.gif|.mp4>            Save offscreen 30 FPS animation
--camera <track_camera|front_camera>      Camera for video recording (default: track_camera)
--export-mcu-header                       Export C firmware header to results/
--output-dir <path>                       Output directory (default: results/)
--no-plot                                 Suppress automatic PNG generation
--no-csv                                  Suppress CSV export
--non-interactive                         Headless execution for draw mode
```

### Example Commands:

1. **Interactive 3D Driving (Free Teleop + Live Front Camera PiP)**:
   ```bash
   python scripts/run_sim.py --robot 1 --mode free --gui
   ```
   *Controls: Arrows or Keypad `8`/`5`/`4`/`6` (Forward/Reverse/Left/Right), Keypad `7`/`9` or `[` / `]` (Yaw CCW/CW), `M` (Cycle DIRECT -> CRUISE -> CONTINUOUS), `B` / `Space` (Brake), `V` (Cycle Camera).*

2. **Benchmark Trajectory Tracking (Path Mode)**:
   ```bash
   python scripts/run_sim.py --robot 2 --mode path --preset doorway_and_threshold --duration 8.0
   ```

3. **Sequential Execution & Comparative Dashboard Across All 3 Robots**:
   ```bash
   python scripts/run_sim.py --robot all --mode path --preset figure8_continuous --duration 10.0 --output-dir results/comparison_fig8
   ```

4. **Offscreen 3D Video Recording from Front Nose Camera**:
   ```bash
   python scripts/run_sim.py --robot 3 --mode path --preset pure_lateral_crabbing --duration 4.0 --record-video results/front_cam.gif --camera front_camera --no-plot
   ```

5. **Firmware C-Header Export for STM32/ESP32 Deployment**:
   ```bash
   python scripts/export_mcu_header.py --robot all --output-dir results/firmware_headers
   ```

6. **Interactive 2D Path Planner (Draw Mode)**:
   ```bash
   python scripts/run_sim.py --robot 1 --mode draw
   ```

---

## 6. Embedded Firmware Deployment (C/C++ Header)

Running `--export-mcu-header` generates clean, standalone C headers (`results/<robot_name>_config.h`) with zero external runtime dependencies:
- Fixed mechanical dimensions (`WHEEL_RADIUS_M`, `OFFSET_LX_M`, `OFFSET_LY_M`).
- Operational dynamic bounds (`MAX_WHEEL_RAD_S`, `MAX_MOTOR_TORQUE_NM`, `MAX_VX_M_S`).
- Row-major $4\times 3$ `J_INV_MATRIX` and $3\times 4$ `J_FWD_MATRIX` kinematic arrays.
- Packed 20-byte `SerialVelocityFrame_t` struct with `0xAA55` frame header and CRC-16/CCITT-FALSE checksum for serial/CAN communication.

---

## 7. ROS 2 Architecture & Interfaces

The simulation adheres strictly to ROS 2 REP-103 and REP-105 standards using pure Python standard-library dataclasses:
- `geometry_msgs/Twist` -> `TwistCommand`
- `nav_msgs/Odometry` -> `OdometryState` (36-element covariances, child_frame_id `"base_footprint"`)
- `sensor_msgs/JointState` -> `JointStateMessage` (4-wheel positions, velocities, torques)
- `sensor_msgs/Imu` -> `ImuStateMessage`
- `geometry_msgs/TransformStamped` -> `TransformMessage` (`odom -> base_footprint -> base_link`)
- `rosgraph_msgs/Clock` -> `ClockMessage`

---

## 8. Verification & Test Suite

All modules are accompanied by comprehensive unit and integration tests under `tests/`.

Run full test suite:
```bash
pytest -v
```

Run complete automated cross-robot batch benchmarks:
```bash
python scripts/run_all_scenarios.py --duration 4.0 --no-plot
```
