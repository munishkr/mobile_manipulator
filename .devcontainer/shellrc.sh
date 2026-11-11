# Interactive-shell conveniences. Sourced from ~/.bashrc and /etc/profile.d.

source "/opt/ros/${ROS_DISTRO}/setup.bash"
[ -f "${COLCON_WS}/install/setup.bash" ] && source "${COLCON_WS}/install/setup.bash"

source /usr/share/colcon_cd/function/colcon_cd.sh 2>/dev/null || true
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash 2>/dev/null || true
export _colcon_cd_root="${COLCON_WS}"

# colcon runs in the MoveIt workspace (COLCON_WS=~/mobile_manipulator/moveit_ws).
# Build with symlinks so launch/param edits take effect without rebuilding.
alias cb='(cd "${COLCON_WS}" && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo)'
alias cbp='cb --packages-select'
alias ct='(cd "${COLCON_WS}" && colcon test && colcon test-result --verbose)'
alias cclean='rm -rf "${COLCON_WS}/build" "${COLCON_WS}/install" "${COLCON_WS}/log"'
alias sws='source "${COLCON_WS}/install/setup.bash"'

# The whole robot: Gazebo tabletop + A200/Kinova/Dex3 + MoveIt + RViz. See CLAUDE.md.
alias mm-run='"${MOBILE_MANIPULATOR_ROOT}/scripts/runRobot.sh"'
alias mm-check='"${MOBILE_MANIPULATOR_ROOT}/scripts/runRobot.sh" --check'
alias mm-gz='gz sim -r "${MOBILE_MANIPULATOR_ROOT}/gazebo_worlds/tabletop.sdf"'
alias mm-rviz='ros2 launch mobile_manipulator_moveit_config moveit_rviz.launch.py'
alias mm-controllers='ros2 control list_controllers -c /a200_0000/controller_manager'

PS1='\[\e[1;35m\][mm:'"${ROS_DISTRO}"']\[\e[0m\] \w\$ '

# gz-transport on loopback (also set in compose.yaml; this covers containers created
# before it). Every gz process and `gz` CLI call must agree on it.
export GZ_IP=127.0.0.1
