#!/usr/bin/env bash
# HOST-side. Grants local containers access to the X server so RViz2 and Gazebo
# can open windows. Scoped to local connections only — no network clients.
set -u
command -v xhost >/dev/null 2>&1 || { echo "xhost not found; install x11-xserver-utils"; exit 0; }
xhost +local: >/dev/null 2>&1 && echo "x11: local connections allowed (${DISPLAY:-unset})"
exit 0
