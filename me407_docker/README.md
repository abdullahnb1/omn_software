# ME407 ROS 2 Humble + MuJoCo + RViz2 Docker Environment

This directory (`/home/nacib/Desktop/me407_docker`) provides a complete, self-contained **Ubuntu 22.04 + ROS 2 Humble** Docker environment with **RViz2**, **MuJoCo** (Python + C/C++ + `mujoco_ros2_control`), **C++ & Python toolchains**, and **real-time bidirectional Git submodule sharing** with `/home/nacib/Desktop/ME407/6-Software`.

---

## Directory Structure

```text
/home/nacib/Desktop/me407_docker/
├── Dockerfile                              # Image definition (Ubuntu 22.04, ROS 2 Humble, RViz2, MuJoCo, C++, Python, nano)
├── docker-compose.yml                      # Container orchestration (X11 GUI, GPU, SSH keys, ME407 bind mounts)
├── entrypoint.sh                           # Auto-sources ROS 2, mujoco_ws, ros2_ws, and links /ros2_ws/src -> ME407/6-Software
├── requirements.txt                        # Python packages (mujoco, numpy, scipy, matplotlib, pyyaml, opencv-python)
├── setup_submodule.sh                      # Helper script to attach your GitHub repo as a submodule at ME407/6-Software
├── workspace/                              # Persistent colcon build/install/log directory (created automatically)
└── config/                                 # Mounted to /site_config inside the container
    ├── ros2_params.yaml                    # ROS 2 node & holonomic omniwheel robot parameters
    └── mujoco_ros2_control_params.yaml     # ros2_control & MuJoCo simulation controller parameters
```

---

## 1. How to Build and Start the Container

### Step 1: Allow Docker to open GUI windows (RViz2 & MuJoCo Viewer) on your host PC
Run this on your host terminal before starting GUI apps:
```bash
xhost +local:docker
```

### Step 2: Build the Docker Image
```bash
cd /home/nacib/Desktop/me407_docker
docker compose build
```

### Step 3: Start and Enter the Container
```bash
cd /home/nacib/Desktop/me407_docker
docker compose up -d
docker compose exec ros2_mujoco bash
```
*(Or for a one-off interactive session: `docker compose run --rm ros2_mujoco`)*

---

## 2. How Host <-> Container File Sharing & Git Submodules Work

In `docker-compose.yml`, the following volumes are shared in real time between your host PC and the container:
1. **`/home/nacib/Desktop/ME407` <-> `/home/nacib/Desktop/ME407`**:
   Mounted at the exact same path on both host and container so Git submodule relative `.git` pointers (`gitdir: ../.git/modules/6-Software`) work from both your host PC and inside the container.
2. **`/ros2_ws/src` -> `/home/nacib/Desktop/ME407/6-Software`**:
   Inside the container, `/ros2_ws/src` is automatically symlinked to `/home/nacib/Desktop/ME407/6-Software`.
   - Any file you `git clone`, edit with `nano`, or generate inside `/ros2_ws/src` in Docker **immediately appears** in `/home/nacib/Desktop/ME407/6-Software` on your host PC.
   - Any code you edit in VS Code / Antigravity on your host PC inside `ME407/6-Software` **immediately appears** inside `/ros2_ws/src` in Docker.
   - Meanwhile, `colcon build` stores `build/`, `install/`, and `log/` inside `me407_docker/workspace/` so build artifacts never clutter your `ME407` Git repository.
3. **SSH Keys & Git Identity**:
   Your host `~/.ssh` and `~/.gitconfig` (`abdullahnb1`) are mounted into the container, so `git clone`, `git pull`, `git commit`, and `git push` work seamlessly inside the container.

### Option A: Connect `ME407/6-Software` as a Git Submodule (Host or Container)
Once you create your software repository on GitHub (e.g. `git@github.com:abdullahnb1/ME407-Software.git`), run either on the host or inside the container:
```bash
/home/nacib/Desktop/me407_docker/setup_submodule.sh git@github.com:abdullahnb1/ME407-Software.git
```
Or manually:
```bash
cd /home/nacib/Desktop/ME407
git rm -f 6-Software/.gitkeep
rmdir 6-Software 2>/dev/null || true
git submodule add git@github.com:abdullahnb1/ME407-Software.git 6-Software
git submodule update --init --recursive
```

### Option B: Clone Individual Packages Inside Docker (`/ros2_ws/src`)
Inside the container, you are placed in `/ros2_ws` by default:
```bash
cd /ros2_ws/src
git clone git@github.com:abdullahnb1/<your_repo>.git
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```
Note: If you run `git clone` or create files as `root` inside the container and want to fix ownership back to `nacib` on the host, you can run `chown -R 1000:1000 /ros2_ws/src` inside the container at any time.

---

## 3. Quick Verification Commands Inside the Container

- **Launch RViz2**:
  ```bash
  rviz2
  ```
- **Launch MuJoCo Interactive Viewer (Python)**:
  ```bash
  python3 -m mujoco.viewer
  ```
- **Launch MuJoCo C++ Standalone Simulator**:
  ```bash
  simulate
  ```
- **Check `mujoco_ros2_control` Packages**:
  ```bash
  ros2 pkg list | grep mujoco
  ```
- **Use Config YAML Files**:
  Your YAML configurations are available at `/site_config/ros2_params.yaml` and `/site_config/mujoco_ros2_control_params.yaml`:
  ```bash
  ros2 run <your_pkg> <your_node> --ros-args --params-file /site_config/ros2_params.yaml
  ```
