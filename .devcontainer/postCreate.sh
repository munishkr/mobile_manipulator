#!/usr/bin/env bash
# Runs once, after the container is created.
set -euo pipefail

ROOT="${MOBILE_MANIPULATOR_ROOT:-$HOME/mobile_manipulator}"
WS="${COLCON_WS:-$ROOT/moveit_ws}"

echo "==> Resolving dependencies declared in ${WS}/src ..."
sudo apt-get update -qq
# warehouse_ros_mongo is listed by the generated MoveIt package but is not
# released for Jazzy; warehouse_ros_sqlite is installed in the image instead.
rosdep install --from-paths "${WS}/src" --ignore-src -y -r --rosdistro "${ROS_DISTRO}" \
    --skip-keys "warehouse_ros_mongo" || \
    echo "    (rosdep reported unresolved keys -- see above; the image already carries the stack)"

echo "==> Building the MoveIt config workspace (${WS}) ..."
( cd "${WS}" && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo )
set +u; source "${WS}/install/setup.bash"; set -u   # colcon setup reads unset COLCON_TRACE

echo "==> Verifying the toolchain ..."
printf '    ROS 2      : %s (rclcpp %s)\n' "${ROS_DISTRO}" \
    "$(ros2 pkg xml rclcpp -t version 2>/dev/null || echo '?')"
printf '    Gazebo     : %s\n' "$(gz sim --versions 2>/dev/null | head -1 || echo 'not on PATH')"
printf '    MoveIt     : %s\n' "$(ros2 pkg xml moveit_ros_move_group -t version 2>/dev/null || echo FAILED)"
printf '    Clearpath  : %s\n' "$(ros2 pkg xml clearpath_gz -t version 2>/dev/null || echo FAILED)"
printf '    RMW        : %s\n' "${RMW_IMPLEMENTATION:-default}"

echo "==> Preflight of the robot bring-up (no processes started) ..."
"${ROOT}/scripts/runRobot.sh" --check

echo "==> Checking the X11 socket ..."
if [ -n "${DISPLAY:-}" ] && timeout 5 xdpyinfo >/dev/null 2>&1; then
    echo "    OK — GUI apps (RViz2, Gazebo) can open windows on ${DISPLAY}"
else
    echo "    WARNING — cannot reach ${DISPLAY:-<unset>}."
    echo "    Run 'xhost +local:' on the HOST, then rebuild/restart the container."
fi

echo "==> Ready. 'mm-run' brings up Gazebo + A200/Kinova/Dex3 + MoveIt + RViz."
