#!/usr/bin/env bash
# Reproduces the verified six-terminal setup for Gazebo Harmonic + ROS 2 Jazzy.
# Usage: ./runRobot.sh [--check]
# Ctrl+C stops only the processes launched by this script.
set -Ee -o pipefail

ROOT="${MOBILE_MANIPULATOR_ROOT:-$HOME/mobile_manipulator}"
WS="$ROOT/moveit_ws"
WORLD="$ROOT/gazebo_worlds/tabletop.sdf"
KINOVA_ROOT="$ROOT/external/ros2_kortex"
KINOVA_PKG="$KINOVA_ROOT/kortex_description"
DEX3_CFG="$ROOT/config/dex3_controllers.yaml"
SCENE="$ROOT/scripts/add_scene_objects.py"
MAP="$ROOT/maps/tabletop_nav.yaml"  # generated from the world: scripts/make_map_from_world.py
NAV2_PARAMS="$ROOT/config/nav2/a200_nav2_slow.yaml"  # slowed base speed limits
LOG_DIR=""
PIDS=()
NAMES=()
ALLOW_RVIZ_EXIT=0

message() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
cleanup() {
  trap - EXIT INT TERM
  if ((${#PIDS[@]})); then
    message 'Stopping launched components...'
    local i pid
    for (( i=${#PIDS[@]}-1; i>=0; i-- )); do
      pid="${PIDS[$i]}"
      # Each background command is its own session/process group via setsid.
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
    done
  fi
  [[ -z "$LOG_DIR" ]] || printf 'Logs: %s\n' "$LOG_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  if [[ -n "$LOG_DIR" && -d "$LOG_DIR" ]]; then
    local i file
    for i in "${!NAMES[@]}"; do
      file="$LOG_DIR/${NAMES[$i]}.log"
      if [[ -f "$file" ]]; then
        printf '\n-- %s (last 25 lines) --\n' "$file" >&2
        tail -n 25 "$file" >&2 || true
      fi
    done
  fi
  exit 1
}
need_file() { [[ -f "$1" ]] || fail "Missing file: $1"; }

# Source the same ROS distro/workspace as the manual terminals.
need_file /opt/ros/jazzy/setup.bash
# Some ROS setup scripts require unset shell variables to remain permitted.
source /opt/ros/jazzy/setup.bash
if [[ -n "${ROBOT_OVERLAY_SETUP:-}" ]]; then
  need_file "$ROBOT_OVERLAY_SETUP"
  source "$ROBOT_OVERLAY_SETUP"
fi
need_file "$WS/install/setup.bash"
source "$WS/install/setup.bash"

for cmd in ros2 gz python3 setsid timeout; do
  command -v "$cmd" >/dev/null || fail "Missing command: $cmd"
done
need_file "$WORLD"
need_file "$NAV2_PARAMS"
need_file "$ROOT/launch/nav2_slow.launch.py"
need_file "$ROOT/launch/localization_custom.launch.py"
need_file "$ROOT/config/nav2/a200_localization.yaml"
need_file "$HOME/clearpath/robot.yaml"
need_file "$DEX3_CFG"
need_file "$SCENE"
need_file "$MAP"
need_file "$ROOT/maps/tabletop_nav.pgm"
need_file "$WS/src/mobile_manipulator_moveit_config/launch/move_group_gazebo.launch.py"
need_file "$WS/src/mobile_manipulator_moveit_config/launch/moveit_rviz.launch.py"
need_file "$WS/src/mobile_manipulator_moveit_config/config/mm_camera.rviz"
[[ -d "$KINOVA_PKG/arms/gen3/7dof/meshes" ]] || fail "Missing Kinova mesh directory: $KINOVA_PKG"
ros2 pkg prefix clearpath_gz >/dev/null || fail 'ROS package clearpath_gz not available'
ros2 pkg prefix mobile_manipulator_moveit_config >/dev/null || fail 'MoveIt config package not available'
ros2 pkg prefix clearpath_nav2_demos >/dev/null || fail "Clearpath Nav2 demos not installed"
ros2 pkg prefix nav2_bringup >/dev/null || fail "Nav2 bringup not installed"
# Identical Gazebo mesh resource path used in your working manual terminals.
export GZ_SIM_RESOURCE_PATH="$KINOVA_ROOT${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
# gz-transport on loopback: with host networking it otherwise multicasts on the Wi-Fi
# interface and stops answering when that drops. gz CLI callers need the same value
# (place_cup.py / collect_cup_dataset.py set it for the Python scripts).
export GZ_IP="${GZ_IP:-127.0.0.1}"

# Some Clearpath helper processes parse package://kortex_description paths via
# the ROS ament index, not GZ_SIM_RESOURCE_PATH. Register the existing files
# without changing/rebuilding/installing the Kinova package.
if ! ros2 pkg prefix kortex_description >/dev/null 2>&1; then
  KORTEX_PREFIX="$ROOT/.kortex_ament_prefix"
  mkdir -p "$KORTEX_PREFIX/share/ament_index/resource_index/packages"
  : > "$KORTEX_PREFIX/share/ament_index/resource_index/packages/kortex_description"
  ln -sfn "$KINOVA_PKG" "$KORTEX_PREFIX/share/kortex_description"
  export AMENT_PREFIX_PATH="$KORTEX_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
fi
KORTEX_SHARE="$(ros2 pkg prefix kortex_description)/share/kortex_description"
need_file "$KORTEX_SHARE/arms/gen3/7dof/meshes/bracelet_no_vision_link.STL"

if [[ "${1:-}" == '--check' ]]; then
  message 'Preflight OK; no processes started.'
  printf 'Gazebo mesh resources: %s\n' "$GZ_SIM_RESOURCE_PATH"
  printf 'Kinova ROS package: %s\n' "$(ros2 pkg prefix kortex_description)"
  exit 0
fi
[[ $# -eq 0 ]] || fail 'Usage: ./runRobot.sh [--check]'

# Catch a frequent accidental case before spawning duplicate servers.
# The ros2 CLI daemon caches the node graph; right after a previous session is stopped
# it still lists /move_group, /cup_detector, ... for a while. That used to block a quick
# restart and could make the readiness checks below pass on stale entries. Flush it
# (it restarts on demand) and decide from real processes instead (next check).
# Leftover processes from an earlier session (or a hand-started detector) double the
# CPU load and steal Fast DDS discovery slots. Refuse rather than run two copies.
leftovers="$(pgrep -af 'gz sim (server|gui|-)|python3 [^ ]*cup_detector\.py|nav2_controller/controller_server|component_container_isolated|moveit_ros_move_group/move_group' \
  | grep -v -e pgrep -e "$$" || true)"
if [[ -n "$leftovers" ]]; then
  printf '%s\n' "$leftovers" >&2
  fail 'Processes from a previous session are still running (listed above). Stop them first, e.g.: pkill -f "gz sim"; pkill -f cup_detector.py'
fi
timeout 20 ros2 daemon stop >/dev/null 2>&1 || true

LOG_DIR="$ROOT/logs/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
[[ -w "$LOG_DIR" ]] || fail "Not writable: $LOG_DIR"
message "Logs: $LOG_DIR"

start() {
  local name="$1"; shift
  message "Starting $name: $*"
  setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  printf 'PID %s | log %s\n' "$!" "$LOG_DIR/$name.log"
}

check_children() {
  local i
  for i in "${!PIDS[@]}"; do
    if [[ "${NAMES[$i]}" == "rviz" && "$ALLOW_RVIZ_EXIT" == 1 ]]; then
      continue  # RViz may be closed/restarted without stopping Gazebo or MoveIt.
    fi
    kill -0 "${PIDS[$i]}" 2>/dev/null || fail "${NAMES[$i]} exited unexpectedly"
  done
}

wait_for_line() {
  local label="$1" seconds="$2" expected="$3"; shift 3
  local started=$SECONDS result
  message "Waiting for $label (maximum ${seconds}s)"
  while ((SECONDS - started < seconds)); do
    check_children
    result="$(timeout 25 "$@" 2>/dev/null || true)"
    if grep -Fxq -- "$expected" <<< "$result"; then
      message "Ready: $label"
      return 0
    fi
    sleep 2
  done
  fail "Timed out waiting for $label: $expected"
}

# Start a Nav2 lifecycle launch and wait for its manager to report active. A lifecycle
# service reply is occasionally lost under load ("failed to send response to
# .../change_state"), which leaves the manager waiting forever; restart that launch
# (up to 3 tries) instead of failing the whole session.
#   start_lifecycle <name> <label> <manager> <seconds-per-try> <command...>
start_lifecycle() {
  local name="$1" label="$2" manager="$3" seconds="$4"; shift 4
  local try started log="$LOG_DIR/$name.log" last
  for try in 1 2 3; do
    start "$name" "$@"
    started=$SECONDS
    message "Waiting for $label (try $try/3, maximum ${seconds}s)"
    while ((SECONDS - started < seconds)); do
      check_children
      if grep -aEq "${manager}\]: .*Managed nodes are active" "$log" 2>/dev/null; then
        message "Ready: $label"
        return 0
      fi
      grep -aEq "${manager}\]: .*(Failed to bring up|Aborting bringup)" "$log" 2>/dev/null && break
      sleep 2
    done
    message "$label did not come up on try $try; restarting $name"
    last=$(( ${#PIDS[@]} - 1 ))
    kill -TERM -- "-${PIDS[$last]}" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-${PIDS[$last]}" 2>/dev/null || true
    unset 'PIDS[last]' 'NAMES[last]'
    PIDS=("${PIDS[@]}"); NAMES=("${NAMES[@]}")
    cp "$log" "$LOG_DIR/${name}_try${try}.log" 2>/dev/null || true
    sleep 2
  done
  fail "$label: $manager did not activate after 3 tries (see $LOG_DIR/${name}_try*.log)"
}

# Terminal 1: Gazebo
# Headless by default: the Gazebo GUI alone costs ~1.7 cores and RViz already shows the
# robot, camera, map and plans. MM_GZ_HEADLESS=0 starts the GUI too; or attach a GUI to
# the running server any time with:  GZ_IP=127.0.0.1 gz sim -g
GZ_ARGS=(-r -v 4)
[[ "${MM_GZ_HEADLESS:-1}" == 1 ]] && GZ_ARGS+=(-s)
start gazebo gz sim "${GZ_ARGS[@]}" "$WORLD"
wait_for_line 'Gazebo tabletop world' 120 '/world/tabletop/create' gz service -l

# Sim clock: Gazebo -> ROS. Clearpath's gz_sim.launch.py normally bridges this, but we
# start `gz sim` directly. Without /clock every use_sim_time node sits at t=0:
# joint_states are stamped 0, robot_state_publisher stops updating arm TF, and RViz
# drops camera images it cannot transform into base_link. Start it before the robot.
start clock_bridge ros2 run ros_gz_bridge parameter_bridge \
  '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
wait_for_line 'Simulation clock' 60 '/clock' ros2 topic list

# Terminal 2: same Clearpath spawn command that worked manually
start robot_spawn ros2 launch clearpath_gz robot_spawn.launch.py \
  'world:=tabletop' "setup_path:=$HOME/clearpath"
wait_for_line 'Clearpath controller manager' 180 \
  '/a200_0000/controller_manager/list_controllers' ros2 service list
wait_for_line 'Kinova trajectory action' 120 \
  '/a200_0000/arm_0_joint_trajectory_controller/follow_joint_trajectory' ros2 action list

# The joystick nodes do nothing in simulation (no /dev/input/ps4) but each holds one of
# the 32 Fast DDS discovery slots (LOCALHOST range), and the full stack needs 33.
# Stop them so every stack node stays discoverable. MM_KEEP_JOY=1 keeps them.
if [[ "${MM_KEEP_JOY:-0}" != 1 ]]; then
  pkill -f 'joy_linux/joy_linux_node' 2>/dev/null || true
  pkill -f 'teleop_twist_joy/teleop_node' 2>/dev/null || true
  message 'Stopped unused joystick nodes (frees 2 DDS discovery slots)'
fi

# Cup DetachableJoint (dex3_extras.urdf.xacro). gz-sim 8 ignores <initial_attach>false</...>
# and welds the cup to left_hand_palm_link as soon as the robot spawns, ~0.6 m away. Any
# arm motion then drags the cup off the table. Detach it, confirm in the Gazebo log
# (needs `gz sim -v 4`), then put the cup back where tabletop.sdf / add_scene_objects.py
# expect it. automate_cup.py attaches it again on purpose with /cup_grasp/attach.
GZ_LOG="$LOG_DIR/gazebo.log"
count_in_gz_log() { grep -ac -- "$1" "$GZ_LOG" 2>/dev/null || true; }
cup_free=0
for attempt in $(seq 1 15); do
  attached="$(count_in_gz_log 'Attaching entity')"
  detached="$(count_in_gz_log 'Detaching joint')"
  if (( attached > 0 && detached >= attached )); then cup_free=1; break; fi
  (( attached > 0 )) && gz topic -t /cup_grasp/detach -m gz.msgs.Empty -p ''
  sleep 2
done
(( cup_free )) || fail "Cup is still welded to the palm (no 'Detaching joint' in $GZ_LOG)"
gz service -s /world/tabletop/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean \
  --timeout 3000 --req 'name: "cup", position: {x: 0.95, y: 0.0, z: 0.811}, orientation: {w: 1.0}' \
  >/dev/null || fail 'Could not reset the cup pose after detaching'
message 'Ready: cup detached from palm and reset on the table'

# Wrist RGB-D camera (cup_wrist_rgbd, defined in dex3_extras.urdf.xacro): Gazebo -> ROS.
# The Sensors system is loaded by the robot model, so it must NOT also be in tabletop.sdf
# (two Sensors systems crash OGRE). The gz point cloud is deliberately not bridged: its
# points use x-forward axes but are stamped with the optical frame; RViz builds the
# cloud from depth_image + camera_info instead.
CAM='/cup_wrist_rgbd'
start camera_bridge ros2 run ros_gz_bridge parameter_bridge \
  "$CAM/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "$CAM/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "$CAM/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  --ros-args -p use_sim_time:=true
wait_for_line 'Wrist camera (Gazebo side)' 60 "$CAM/image" gz topic -l
if timeout 15 ros2 topic echo --once "$CAM/camera_info" --field header.frame_id \
     >"$LOG_DIR/camera_check.log" 2>&1; then
  message "Ready: wrist camera on $CAM/{image,depth_image,camera_info}"
else
  message "Warning: no frames on $CAM/camera_info yet; inspect camera_bridge.log"
fi

# YOLO cup detector. Started HERE, before MoveIt/Nav2, on purpose:
# with ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST, Fast DDS only probes participant
# slots 0-31. The full stack fills all 32, so a detector started afterwards gets
# slot >= 32 and is invisible to every later process (pick scripts get no
# detections even though the detector logs them). Set MM_START_DETECTOR=0 to skip.
if [[ "${MM_START_DETECTOR:-1}" == 1 ]]; then
  YOLO_PY="${MM_YOLO_PYTHON:-/opt/yolo_venv/bin/python3}"
  need_file "$YOLO_PY"
  need_file "$ROOT/perception/cup_detector.py"
  # One torch thread and 2 frames/s: it used ~2.2 cores at the defaults. observe_cup
  # needs 10 samples, i.e. ~5 s at 2 Hz.
  start cup_detector env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$YOLO_PY" "$ROOT/perception/cup_detector.py" --threads 1 --max-rate 2 \
    --device "${MM_YOLO_DEVICE:-0}"   # GPU 0 (CUDA is available in the venv); MM_YOLO_DEVICE=cpu to force CPU
  # Do not continue until its DDS participant exists, so it keeps a low slot.
  wait_for_line 'Cup detector node' 120 '/cup_detector' ros2 node list
fi

# Terminal 3: Dex3 controller. The controller manager service may be advertised
# before Clearpath's description / ros2_control configuration has fully settled.
# Also tolerate a controller left active by an earlier session.
DEX3_ACTION='/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory'
controller_active() {
  local output
  output="$(timeout 12 ros2 control list_controllers -c /a200_0000/controller_manager 2>/dev/null || true)"
  grep -Eq '^dex3_joint_trajectory_controller([[:space:]]|$).*active' <<<"$output"
}
action_available() {
  timeout 12 ros2 action list 2>/dev/null | grep -Fxq "$DEX3_ACTION"
}

message 'Checking Dex3 controller status'
if controller_active && action_available; then
  message 'Dex3 controller already active; skipping spawner.'
else
  # Wait for the whole robot (including all hand joints) to be available.
  wait_for_line 'Clearpath joint-state topic' 90 \
    '/a200_0000/platform/joint_states' ros2 topic list
  success=0
  for attempt in 1 2 3 4; do
    message "Activating Dex3 controller (attempt $attempt/4)"
    if timeout 45 ros2 run controller_manager spawner \
       dex3_joint_trajectory_controller \
       -c /a200_0000/controller_manager \
       --param-file "$DEX3_CFG" \
       >"$LOG_DIR/dex3_spawner_attempt_${attempt}.log" 2>&1; then
      if controller_active && action_available; then
        success=1
        break
      fi
    fi
    # Even if spawner exited with an error, controller activation may have
    # completed asynchronously. Check actual state before retrying.
    if controller_active && action_available; then
      success=1
      break
    fi
    message "Dex3 not ready; last spawner output:"
    tail -n 18 "$LOG_DIR/dex3_spawner_attempt_${attempt}.log" || true
    sleep 5
  done
  if (( !success )); then
    message 'Controller manager currently reports:'
    timeout 15 ros2 control list_controllers -c /a200_0000/controller_manager || true
    fail "Dex3 could not be activated. Full logs: $LOG_DIR/dex3_spawner_attempt_*.log"
  fi
fi
message 'Ready: Dex3 trajectory action'

# Terminal 4: MoveIt planning server
start move_group ros2 launch mobile_manipulator_moveit_config move_group_gazebo.launch.py
wait_for_line 'MoveIt planning scene' 120 '/apply_planning_scene' ros2 service list

# Terminal 5: RViz
# MM_RVIZ_GDB=true runs RViz under gdb so a segfault leaves a backtrace in rviz.log.
start rviz ros2 launch mobile_manipulator_moveit_config moveit_rviz.launch.py "gdb:=${MM_RVIZ_GDB:-false}"
wait_for_line 'RViz' 120 '/rviz2' ros2 node list
if ! timeout 10 ros2 param set /rviz2 use_sim_time true >"$LOG_DIR/rviz_clock.log" 2>&1; then
  message 'Warning: Could not enable simulation time in RViz; inspect rviz_clock.log'
fi

# Terminal 6: world collision objects in MoveIt
message 'Adding table and cup to planning scene'
if ! timeout 30 python3 "$SCENE" >"$LOG_DIR/scene.log" 2>&1; then
  fail 'add_scene_objects.py failed'
fi
if ! grep -q 'Table and cup added to MoveIt planning scene' "$LOG_DIR/scene.log"; then
  fail 'Planning scene did not report successful insertion'
fi

# Terminal 7: AMCL localization using saved occupancy map
message 'Starting map server and AMCL'

# Repo wrapper: AMCL tuned in config/nav2/a200_localization.yaml. The map is generated
# from the world by scripts/make_map_from_world.py (map frame == Gazebo world).
start_lifecycle localization 'Map server + AMCL active' lifecycle_manager_localization 30 \
  ros2 launch "$ROOT/launch/localization_custom.launch.py" \
  use_sim_time:=true \
  "setup_path:=$HOME/clearpath" \
  "map:=$MAP" \
  scan_topic:=/a200_0000/sensors/lidar2d_0/scan

# Random-start initialization: distribute AMCL particles across free map space
wait_for_line 'AMCL global-localization service' 60 \
  '/a200_0000/reinitialize_global_localization' ros2 service list

# Simulation: seed AMCL with the robot's true Gazebo pose (map frame == Gazebo world).
# Random global localization left AMCL ~18 cm / 3 deg off until the base had driven
# around, which put stations, drop points and arrival checks in the wrong place.
# MM_AMCL_INIT=global restores the random global start.
if [[ "${MM_AMCL_INIT:-truth}" == global ]]; then
  message 'Initializing AMCL global localization'
  if ! timeout 30 ros2 service call \
      /a200_0000/reinitialize_global_localization \
      std_srvs/srv/Empty '{}' \
      >"$LOG_DIR/amcl_initialization.log" 2>&1; then
    fail 'AMCL global localization initialization failed'
  fi
else
  message 'Seeding AMCL from Gazebo ground truth'
  if ! (cd "$ROOT/scripts" && timeout 120 python3 seed_amcl_from_gazebo.py) \
      >"$LOG_DIR/amcl_initialization.log" 2>&1; then
    cat "$LOG_DIR/amcl_initialization.log" >&2
    fail 'Could not seed AMCL from Gazebo (see amcl_initialization.log)'
  fi
  tail -n 1 "$LOG_DIR/amcl_initialization.log"
fi

# Terminal 8: Nav2 navigation
message 'Starting Nav2 navigation stack'

# Repo wrapper around clearpath_nav2_demos/nav2.launch.py using slowed limits
# from config/nav2/a200_nav2_slow.yaml (fast turns made AMCL jump).
# MM_NAV2_COMPOSITION=true would run all Nav2 servers in one process (~13 fewer
# processes and DDS slots) but the composed MPPI controller segfaults here, so off.
start_lifecycle navigation 'Nav2 stack active' lifecycle_manager_navigation 90 \
  ros2 launch "$ROOT/launch/nav2_slow.launch.py" \
  use_sim_time:=true \
  "params_file:=$NAV2_PARAMS" \
  "setup_path:=$HOME/clearpath" \
  scan_topic:=/a200_0000/sensors/lidar2d_0/scan \
  "use_composition:=${MM_NAV2_COMPOSITION:-false}"

wait_for_line 'NavigateToPose action' 90 \
  '/a200_0000/navigate_to_pose' ros2 action list

message 'Ready: AMCL + Nav2 navigation'

ALLOW_RVIZ_EXIT=1
message 'READY — Gazebo + A200 + Kinova + Dex3 + MoveIt + AMCL + Nav2 + RViz'
message "Navigation map: $MAP"
message 'Verify AMCL convergence before sending navigation goals.'
message 'RViz can now be closed independently; Gazebo and MoveIt will remain running.'
message 'Set RViz Fixed Frame to base_link if the goal marker is not visible.'
message 'Press Ctrl+C to stop launched components.'
while :; do
  check_children
  sleep 3
done
