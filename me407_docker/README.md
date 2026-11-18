# ME407 Docker Guide (For Complete Beginners)

Welcome! This guide will walk you step-by-step through setting up and using our **ME407 Robotics Software Environment** (`ROS 2 Humble` + `MuJoCo` + `RViz2` + `Python` + `C++`) using **Docker**.

**Never used Docker before? Don't worry!** Every concept and command is explained in plain English below.

---

## 1. What is Docker & Why Are We Using It?

Normally, setting up a robotics computer takes hours: you have to install the exact version of Ubuntu (`22.04`), install `ROS 2 Humble`, install `MuJoCo`, configure C++ compilers, and install dozens of Python libraries. Even worse, if one teammate has Ubuntu 24.04 or a different Python version, code that works on one laptop breaks on another ("*It worked on my machine!*").

**Docker solves this** by running a lightweight, isolated "mini-computer" inside your PC that already has the exact operating system and all robotics software pre-installed.

### Three Words You Need to Know:
1. **Dockerfile**: A recipe (text file) that lists every piece of software we want installed (Ubuntu 22.04, ROS 2 Humble, RViz2, MuJoCo, Python, C++, `nano`, etc.).
2. **Docker Image**: When you "build" the `Dockerfile`, Docker downloads everything on the list and saves a frozen snapshot called an **Image**. Think of this like an installer or game disc. You only build it **once**.
3. **Docker Container**: When you "run" the Image, it boots up a live environment called a **Container**. You can open a terminal inside it, run simulations, and open 3D windows (`RViz2`, `MuJoCo`).
4. **Shared Folders (Volumes)**: Our container is connected directly to the `omn_software` folder on your regular PC. **When you edit a Python or C++ file in VS Code on your regular PC, it changes instantly inside Docker—and vice versa!**

---

## 2. Folder Structure Overview

```text
ME407/
└── 6-Software/
    └── omn_software/                           # Our Software Git Submodule Repo
        ├── mecanum_sim/                        # 4-Wheel Mecanum MuJoCo Simulation Package
        ├── working/                            # Path & Location Finding Scripts
        └── me407_docker/                       # <--- YOU ARE HERE (Docker Setup)
            ├── Dockerfile                      # The recipe that installs ROS 2, MuJoCo, Python, C++
            ├── docker-compose.yml              # Settings for starting the container, 3D GUI & shared folders
            ├── entrypoint.sh                   # Startup script that automatically loads ROS 2 & paths
            ├── requirements.txt                # List of Python libraries installed in Docker
            ├── README.md                       # This beginner guide
            ├── workspace/                      # Stores ROS 2 compiled files (build/, install/, log/)
            └── config/                         # YAML configuration files (mounted at /site_config)
                ├── ros2_params.yaml
                └── mujoco_ros2_control_params.yaml
```

---

## 3. Step-by-Step Setup (First Time Only)

### Step 0: Install Docker on Your Ubuntu PC (Skip if already installed)
To check if Docker is already installed, open a terminal (`Ctrl+Alt+T`) and run:
```bash
docker --version
docker compose version
```
If both print a version number, **skip to Step 1**!

If not, run these commands to install Docker:
```bash
# 1. Download and run the official Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Allow your user to run Docker without typing 'sudo' every time
sudo usermod -aG docker $USER
```
> **IMPORTANT**: After running `sudo usermod -aG docker $USER`, **log out and log back into Ubuntu** (or restart your PC) so the permission takes effect!

---

### Step 1: Clone the Repository (If You Haven't Already)
Because `omn_software` is a **Git Submodule** inside the main `ME407` repository:

- **If you are cloning the full `ME407` repository for the first time**, include `--recurse-submodules` so it downloads `omn_software` too:
  ```bash
  git clone --recurse-submodules git@github.com:abdullahnb1/ME407.git
  ```
- **If you already cloned `ME407` earlier, but `6-Software/omn_software` is empty**, run:
  ```bash
  cd ~/Desktop/ME407
  git submodule update --init --recursive
  ```

---

### Step 2: Build the Docker Image (First Time Only)
Now we will tell Docker to read our `Dockerfile` and build the `me407_ros2_mujoco:humble` image.

1. Go into the `me407_docker` folder:
   ```bash
   cd ~/Desktop/ME407/6-Software/omn_software/me407_docker
   ```
2. Build the image:
   ```bash
   docker compose build
   ```
> **What to expect**: The very first time you run this, it will download Ubuntu 22.04, ROS 2 Humble, MuJoCo, and compile `mujoco_ros2_control`. This takes **5 to 10 minutes** depending on your internet speed. Grab a coffee! Once it finishes, Docker saves the image permanently on your PC—you won't have to wait again.

---

## 4. Daily Workflow: How to Start, Use, and Stop Docker

Every time you sit down to work on the project, follow these 3 simple steps:

### Step 1: Allow Docker to Open 3D Windows on Your Screen
Because `RViz2` and `MuJoCo` open graphical windows, run this command on your regular PC terminal once per reboot:
```bash
xhost +local:docker
```
*(You should see: `non-network local connections being added to access control list`)*

### Step 2: Start the Container in the Background
Go to the `me407_docker` folder and start the container:
```bash
cd ~/Desktop/ME407/6-Software/omn_software/me407_docker
docker compose up -d
```
*(The `-d` means "detached / run in the background". Your container is now turned on!)*

### Step 3: Enter the Container Terminal
Now jump inside the running container:
```bash
docker compose exec ros2_mujoco bash
```
You will see your terminal prompt change to something like:
```text
root@me407-docker:/ros2_ws#
```
**Congratulations! You are now inside the Docker container!**
- Everything you type in this terminal runs inside Ubuntu 22.04 with ROS 2 Humble and MuJoCo already loaded.
- **Want a second or third terminal inside the same container?** Just open a new terminal tab on your PC, go to `me407_docker`, and run `docker compose exec ros2_mujoco bash` again!

### Step 4: How to Exit and Turn Off the Container
- **To leave the container terminal** (while keeping the container running in the background):
  ```bash
  exit
  ```
- **To completely shut down the container** when you are done working for the day (run on your PC inside `me407_docker`):
  ```bash
  docker compose down
  ```

---

## 5. Where Are My Files Inside Docker? (File Sharing Guide)

You do **NOT** need to copy files back and forth between your PC and Docker. They share the exact same folders in real time!

| Folder on Your Regular PC | Folder Inside Docker Container | What is it for? |
| :--- | :--- | :--- |
| `.../omn_software/` | `/ros2_ws/src/omn_software` | **All our project code** (`mecanum_sim`, `working`, etc.). Edit in VS Code on your PC, run inside Docker! |
| `.../me407_docker/config/` | `/site_config` | **YAML configuration files** (`ros2_params.yaml`, `mujoco_ros2_control_params.yaml`). |
| `.../me407_docker/workspace/` | `/ros2_ws` | **ROS 2 Workspace root**. Stores compiled outputs (`build/`, `install/`, `log/`). |
| `~/.ssh` & `~/.gitconfig` | `/root/.ssh` & `/etc/gitconfig` | **Your GitHub SSH keys & name**, so `git pull` / `git push` work inside Docker too. |

---

## 6. Things to Try Inside the Container!

Once you are inside the container (`root@me407-docker:/ros2_ws#`), try these commands:

### A. Run the `mecanum_sim` MuJoCo Simulation & Tests
1. Go to the `mecanum_sim` folder inside the container:
   ```bash
   cd /ros2_ws/src/omn_software/mecanum_sim
   ```
2. Check the environment & run all unit tests:
   ```bash
   python3 scripts/check_env.py
   pytest
   ```
3. Launch the interactive Mecanum simulation GUI:
   ```bash
   python3 scripts/run_sim.py
   ```

### B. Open the Standalone MuJoCo 3D Viewer
- Using Python:
  ```bash
  python3 -m mujoco.viewer
  ```
- Or using the C++ binary:
  ```bash
  simulate
  ```
*(You can drag-and-drop any `.xml` robot model from `mecanum_sim/models/` right into the 3D window!)*

### C. Open RViz2 (ROS 2 Visualization)
```bash
rviz2
```

### D. Build ROS 2 Packages (`colcon build`)
Whenever we add or modify ROS 2 C++/Python packages inside `/ros2_ws/src`:
```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### E. Use the YAML Config Files in `/site_config`
Our custom YAML parameter files are located in `/site_config/`:
- `/site_config/ros2_params.yaml`
- `/site_config/mujoco_ros2_control_params.yaml`

You can pass them to any ROS 2 node using:
```bash
ros2 run <package_name> <node_name> --ros-args --params-file /site_config/ros2_params.yaml
```

---

## 7. Quick Reference Cheat Sheet

| What I Want To Do | Where to Run It | Command |
| :--- | :--- | :--- |
| Allow Docker to open GUI windows | Host PC | `xhost +local:docker` |
| Build the image (first time or after editing `Dockerfile`) | Host PC (`me407_docker/`) | `docker compose build` |
| Start the container in the background | Host PC (`me407_docker/`) | `docker compose up -d` |
| Open a bash terminal inside the container | Host PC (`me407_docker/`) | `docker compose exec ros2_mujoco bash` |
| Check if the container is currently running | Host PC | `docker ps` |
| Stop and shut down the container | Host PC (`me407_docker/`) | `docker compose down` |
| Edit a file inside the container terminal | Inside Container | `nano <filename>` |

---

## 8. Troubleshooting Common Beginner Issues

### 1. Error: `permission denied while trying to connect to the Docker daemon socket`
- **Cause**: Your Linux user isn't in the `docker` group yet.
- **Fix**: Run `sudo usermod -aG docker $USER`, then **log out and log back into Ubuntu** (or run `newgrp docker` in your current terminal).

### 2. Error: `qt.qpa.xcb: could not connect to display` or MuJoCo/RViz window won't open
- **Cause**: Your PC's display server blocked Docker from opening a window.
- **Fix**: Open a terminal on your **Host PC** (outside Docker) and run:
  ```bash
  xhost +local:docker
  ```

### 3. Files created inside Docker show a "Lock" icon on my PC
- **Cause**: Inside the Docker container, you run as `root`, so new files created inside Docker are owned by `root`.
- **Fix**: Inside the container terminal, run this command to give ownership back to your regular PC user (`1000:1000`):
  ```bash
  chown -R 1000:1000 /ros2_ws/src/omn_software
  ```
