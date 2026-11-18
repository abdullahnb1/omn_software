#!/bin/bash
set -e

# 1. Source ROS 2 Humble base environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# 2. Source pre-built mujoco_ros2_control underlay workspace
if [ -f "/opt/mujoco_ws/install/setup.bash" ]; then
    source "/opt/mujoco_ws/install/setup.bash"
fi

# 3. Source user development workspace (if already built with colcon)
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source "/ros2_ws/install/setup.bash"
fi

# 4. Add mecanum_sim Python package to PYTHONPATH so it works immediately
if [ -d "/ros2_ws/src/omn_software/mecanum_sim/src" ]; then
    export PYTHONPATH="/ros2_ws/src/omn_software/mecanum_sim/src:/ros2_ws/src/omn_software/mecanum_sim:${PYTHONPATH}"
fi

# 5. Configure Git safe.directory so root inside container can operate on host-owned repos
git config --system --add safe.directory '*' 2>/dev/null || true

# 6. Default environment variables for ROS 2, MuJoCo, and Config YAMLs
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export SITE_CONFIG_DIR="${SITE_CONFIG_DIR:-/site_config}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

exec "$@"
