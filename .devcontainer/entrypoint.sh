#!/usr/bin/env bash
# Sources the ROS underlay (and the colcon overlay, once it exists) for every
# process the container starts — including VS Code's own shells and tasks.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [ -f "${COLCON_WS}/install/setup.bash" ]; then
    source "${COLCON_WS}/install/setup.bash"
fi

exec "$@"
