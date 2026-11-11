# syntax=docker/dockerfile:1
#
# ROS 2 Jazzy + MoveIt 2 + ros2_control + Gazebo Harmonic + Clearpath A200 sim.
# Image for the mobile-manipulator workspace (Husky A200 + Kinova Gen3 7-DoF +
# Unitree Dex3-1 hand). Layers 1, 2 and 4 are byte-identical to the nav2_ws
# Dockerfile on purpose so Docker reuses those cached layers.
#
# Gazebo comes in through ros-${ROS_DISTRO}-ros-gz, which on Jazzy depends on the
# gz-*-vendor packages (gz-sim-vendor == Harmonic / gz-sim 8) published in the ROS
# apt repo. There is deliberately no packages.osrfoundation.org source here: mixing
# the vendored and upstream Gazebo packages is the usual cause of duplicate-symbol
# and plugin-not-found failures.

ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ARG USERNAME=ros
ARG USER_UID=1001
ARG USER_GID=1001

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=${ROS_DISTRO}

# --- 1. Desktop tooling: RViz2, rqt, Qt stack -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-desktop \
        ros-${ROS_DISTRO}-rqt-common-plugins \
        ros-${ROS_DISTRO}-rqt-tf-tree \
        ros-${ROS_DISTRO}-tf2-tools \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Gazebo Harmonic + the ros_gz bridge ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ros-gz-sim \
        ros-${ROS_DISTRO}-ros-gz-bridge \
        ros-${ROS_DISTRO}-ros-gz-image \
    && rm -rf /var/lib/apt/lists/*

# --- 3. Manipulation stack: MoveIt 2, ros2_control, Clearpath A200 simulator -
# kortex_description is NOT installed from apt on purpose: the robot description
# resolves Kinova meshes from the checked-in copy under
# mobile_manipulator/external/ros2_kortex, registered into the ament index by
# scripts/runRobot.sh (.kortex_ament_prefix). Installing the apt package would
# silently switch the resolution to /opt/ros and bypass that copy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-moveit \
        ros-${ROS_DISTRO}-moveit-ros-move-group \
        ros-${ROS_DISTRO}-moveit-kinematics \
        ros-${ROS_DISTRO}-moveit-planners \
        ros-${ROS_DISTRO}-moveit-simple-controller-manager \
        ros-${ROS_DISTRO}-moveit-ros-visualization \
        ros-${ROS_DISTRO}-moveit-ros-warehouse \
        ros-${ROS_DISTRO}-moveit-configs-utils \
        ros-${ROS_DISTRO}-moveit-setup-assistant \
        ros-${ROS_DISTRO}-moveit-servo \
        ros-${ROS_DISTRO}-warehouse-ros-sqlite \
        ros-${ROS_DISTRO}-ros2-control \
        ros-${ROS_DISTRO}-ros2-controllers \
        ros-${ROS_DISTRO}-gz-ros2-control \
        ros-${ROS_DISTRO}-joint-state-publisher-gui \
        ros-${ROS_DISTRO}-clearpath-simulator \
        ros-${ROS_DISTRO}-clearpath-desktop \
        ros-${ROS_DISTRO}-clearpath-manipulators \
        ros-${ROS_DISTRO}-clearpath-manipulators-description \
        ros-${ROS_DISTRO}-robotiq-description \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rosbag2-storage-mcap \
        ros-${ROS_DISTRO}-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

# --- 4. Developer tooling ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion build-essential ccache clang-format cmake curl \
        gdb git gnupg htop less lsb-release mesa-utils nano openssh-client \
        x11-utils \
        python3-colcon-common-extensions python3-colcon-mixin python3-pip \
        python3-rosdep python3-vcstool ripgrep sudo tmux vim wget x11-apps \
    && rm -rf /var/lib/apt/lists/*

# --- 4b. Perception: YOLO (ultralytics + CUDA torch) in its own venv ----------
# Kept out of the system Python so pip cannot replace numpy/opencv under ROS.
# --system-site-packages keeps rclpy, cv_bridge and tf2_ros importable.
# torch+cu124 is a ~900 MB wheel (plus ~2 GB of nvidia-* wheels) from
# download.pytorch.org, which can crawl at <1 MB/s. pip's default 15 s read timeout
# then kills the build mid-download, so: long timeout, resumable downloads (pip >=
# 25.1), and a BuildKit cache mount so finished wheels survive a failed rebuild
# (the cache lives outside the image, hence no --no-cache-dir).
COPY yolo-requirements.txt /tmp/yolo-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages /opt/yolo_venv \
    && /opt/yolo_venv/bin/pip install --timeout 120 --retries 10 --upgrade 'pip>=25.1' \
    && /opt/yolo_venv/bin/pip install --timeout 120 --retries 10 --resume-retries 10 \
        -r /tmp/yolo-requirements.txt \
    && rm /tmp/yolo-requirements.txt

# --- 5. Non-root user matching the host UID/GID -----------------------------
# Noble images ship an `ubuntu` user at UID 1000. If the host UID collides with an
# existing account we remove it, so ${USERNAME}/${HOME} are always predictable.
RUN if getent passwd ${USER_UID} >/dev/null; then \
        userdel -r "$(getent passwd ${USER_UID} | cut -d: -f1)" 2>/dev/null || true; \
    fi \
    && if getent group ${USER_GID} >/dev/null; then \
        groupmod -n ${USERNAME} "$(getent group ${USER_GID} | cut -d: -f1)"; \
    else \
        groupadd --gid ${USER_GID} ${USERNAME}; \
    fi \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && usermod -aG video,audio,dialout ${USERNAME}

USER ${USER_UID}:${USER_GID}
ENV HOME=/home/${USERNAME}
SHELL ["/bin/bash", "-lc"]

# Named volumes mount over these paths; creating them as ${USERNAME} first means
# Docker seeds the volume with the right ownership instead of a root-owned dir.
RUN mkdir -p ${HOME}/.ros ${HOME}/.gz ${HOME}/.ignition ${HOME}/.ccache ${HOME}/.config

# The robot description, the Clearpath robot.yaml and the generated MoveIt URDF
# all hard-code /home/ros/mobile_manipulator and /home/ros/clearpath
# ($(env HOME)/mobile_manipulator/..., file:///home/ros/mobile_manipulator/...).
# Point those at the bind-mounted workspace instead of copying anything.
RUN ln -s /workspaces/mm_ws/mobile_manipulator ${HOME}/mobile_manipulator \
    && ln -s /workspaces/mm_ws/clearpath ${HOME}/clearpath

# rosdep's root-owned cache is useless to us; seed a user-owned one.
RUN rosdep update --rosdistro ${ROS_DISTRO} || true

COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chmod=0644 shellrc.sh /etc/profile.d/ros-devcontainer.sh
RUN echo '[ -f /etc/profile.d/ros-devcontainer.sh ] && source /etc/profile.d/ros-devcontainer.sh' >> ${HOME}/.bashrc

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
