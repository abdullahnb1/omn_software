# ==============================================================================
# Base Image: Ubuntu 22.04 (Jammy) + ROS 2 Humble Desktop (includes RViz2)
# ==============================================================================
FROM osrf/ros:humble-desktop

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# 1. System Utilities, C++ Toolchain, Python Base, OpenGL/GUI & ROS 2 Packages
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Editors, SSH & CLI tools
    nano \
    vim \
    git \
    openssh-client \
    curl \
    wget \
    tmux \
    htop \
    tree \
    unzip \
    bash-completion \
    iputils-ping \
    net-tools \
    # C++ Build Toolchain & Libraries
    build-essential \
    cmake \
    ninja-build \
    gdb \
    pkg-config \
    clang-format \
    libeigen3-dev \
    libboost-all-dev \
    # Python 3 Development & Pip
    python3-pip \
    python3-venv \
    python3-dev \
    python3-tk \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    # OpenGL / X11 / GLFW Dependencies for MuJoCo & RViz2 GUI
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libosmesa6-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    xorg-dev \
    mesa-utils \
    x11-apps \
    # Additional ROS 2 Humble Packages for Control & Simulation
    ros-humble-rviz2 \
    ros-humble-rqt* \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-tools \
    ros-humble-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 2. Python Libraries & MuJoCo Python Bindings
# ==============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ==============================================================================
# 3. Standalone MuJoCo C/C++ Library in /opt/mujoco
# ==============================================================================
ENV MUJOCO_VERSION=3.2.6
ENV MUJOCO_DIR=/opt/mujoco
RUN mkdir -p ${MUJOCO_DIR} && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C ${MUJOCO_DIR} --strip-components=1 && \
    rm /tmp/mujoco.tar.gz

ENV LD_LIBRARY_PATH=${MUJOCO_DIR}/lib:${LD_LIBRARY_PATH}
ENV PATH=${MUJOCO_DIR}/bin:${PATH}

# ==============================================================================
# 4. Build mujoco_ros2_control from Source in Underlay (/opt/mujoco_ws)
# ==============================================================================
ENV MUJOCO_WS=/opt/mujoco_ws
RUN mkdir -p ${MUJOCO_WS}/src && \
    cd ${MUJOCO_WS}/src && \
    git clone https://github.com/ros-controls/mujoco_ros2_control.git -b main && \
    git clone https://github.com/pal-robotics/mujoco_vendor.git -b master

RUN apt-get update && \
    . /opt/ros/humble/setup.sh && \
    (rosdep init || true) && \
    rosdep update && \
    rosdep install --from-paths ${MUJOCO_WS}/src --ignore-src -r -y --rosdistro humble && \
    . /opt/ros/humble/setup.sh && \
    cd ${MUJOCO_WS} && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/* ${MUJOCO_WS}/build ${MUJOCO_WS}/log

# ==============================================================================
# 5. User Workspace, Config Files, and Entrypoint Setup
# ==============================================================================
ENV ROS_WS=/ros2_ws
RUN mkdir -p ${ROS_WS}/src /site_config

# Copy YAML configuration files into the container
COPY config/ /site_config/

# Copy and configure the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Auto-source environments in interactive bash sessions (e.g. docker exec)
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "[ -f /opt/mujoco_ws/install/setup.bash ] && source /opt/mujoco_ws/install/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export PYTHONPATH=/ros2_ws/src/omn_software/mecanum_sim/src:/ros2_ws/src/omn_software/mecanum_sim:\$PYTHONPATH" >> /root/.bashrc

WORKDIR ${ROS_WS}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
