#!/usr/bin/env python3
"""Shared helpers for the cup scripts (library only; not run directly).

Gazebo model poses, world -> base_link conversion, MoveIt scene/FK/Cartesian
services, action goals with re-send, Gazebo detach with log confirmation, and
the MoveIt detach + Dex3 open used after a placement.

Simulation only. The table is x=1.15 +/- 0.40, y=0 +/- 0.30, tabletop z=0.75 m;
cup height 0.12 m. No TF subscriptions: poses come from Gazebo, palm FK from MoveIt.
"""

import copy
import math
import os
import re
import subprocess
import time

# gz CLI calls must use the same transport interface as the sim (runRobot.sh exports
# GZ_IP=127.0.0.1; Wi-Fi multicast broke gz-transport in long sessions).
os.environ.setdefault('GZ_IP', '127.0.0.1')

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject,
                             PlanningScene, PlanningSceneComponents,
                             MoveItErrorCodes)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetPlanningScene, GetPositionFK)
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

GROUP = 'Kinova_arm'
PALM = 'left_hand_palm_link'
CUP_ID = 'target_cup'
TABLE_TOP = 0.75
CUP_HEIGHT = 0.12
CUP_RADIUS = 0.035
# A relatively open, previously checked Dex3 hand configuration.
HAND_OPEN = {
    'left_hand_thumb_0_joint': 0.0,
    'left_hand_thumb_1_joint': 0.0,
    'left_hand_thumb_2_joint': 0.2,
    'left_hand_middle_0_joint': -0.2,
    'left_hand_middle_1_joint': -0.2,
    'left_hand_index_0_joint': -0.2,
    'left_hand_index_1_joint': -0.2,
}
NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TRIPLE = re.compile(r'\[\s*(' + NUM + r')\s+(' + NUM + r')\s+(' + NUM + r')\s*\]')


def gazebo_pose(model, attempts=5):
    # `gz model --pose` intermittently returns empty output (seen under CPU load);
    # retry instead of aborting a pick/place halfway through.
    cmd = ['gz', 'model', '-m', model, '--pose']
    for attempt in range(attempts):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        except subprocess.TimeoutExpired:
            p = subprocess.CompletedProcess(cmd, 1, '', 'gz model timed out after 15 s')
            continue
        values = [tuple(map(float, m.groups())) for m in TRIPLE.finditer(p.stdout)]
        if not p.returncode and len(values) >= 2:
            return values[0] + values[1]  # world xyz, rpy
        time.sleep(0.3 * (attempt + 1))
    if p.returncode:
        raise RuntimeError(f"Gazebo model lookup failed ({model}): {p.stderr or p.stdout}")
    raise RuntimeError(f"Could not parse Gazebo pose of {model}: {p.stdout}")


# Kinova Gen3 joints 1/3/5/7 are continuous. MoveIt's CheckStartStateBounds rejects a
# start state where they sit outside [-pi, pi] ("Start state out of bounds"), which
# happens after any plan that crossed the wrap. Plan from a normalised copy and then
# shift the returned trajectory back onto the real joint turns: no physical spin.
CONTINUOUS_JOINTS = {'arm_0_joint_1', 'arm_0_joint_3', 'arm_0_joint_5', 'arm_0_joint_7'}


def wrap_pi(a):
    # Keep strictly inside +-pi: a joint resting exactly on -pi (seen after a lift)
    # can round to just below it and still fail CheckStartStateBounds.
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return min(max(a, -math.pi + 1e-6), math.pi - 1e-6)


def normalized_start(state):
    """Copy of a RobotState with continuous arm joints wrapped into [-pi, pi]."""
    out = copy.deepcopy(state)
    pos = list(out.joint_state.position)
    for i, name in enumerate(out.joint_state.name):
        if name in CONTINUOUS_JOINTS:
            pos[i] = wrap_pi(pos[i])
    out.joint_state.position = pos
    return out


def unwrap_to_start(joint_trajectory, start_state):
    """Shift continuous joints by whole turns so the path starts at start_state."""
    now = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
    prev = [now.get(n, 0.0) for n in joint_trajectory.joint_names]
    for pt in joint_trajectory.points:
        fixed = []
        for name, value, ref in zip(joint_trajectory.joint_names, pt.positions, prev):
            if name in CONTINUOUS_JOINTS:
                value += 2 * math.pi * round((ref - value) / (2 * math.pi))
            fixed.append(value)
        pt.positions = fixed
        prev = fixed
    return joint_trajectory


def quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy)


def qmul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X+x*W+y*Z-z*Y,
            w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W,
            w*W-x*X-y*Y-z*Z)


def qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def rotate(q, v):
    return qmul(qmul(q, (v[0], v[1], v[2], 0.0)), qinv(q))[:3]


def cup_pose_in_base(cup, base):
    qc = quat(*cup[3:])
    qb = quat(*base[3:])
    xyz = rotate(qinv(qb), tuple(cup[i] - base[i] for i in range(3)))
    q = qmul(qinv(qb), qc)
    p = Pose()
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def service(node, client, req, title, timeout=20.0):
    if not client.wait_for_service(timeout_sec=10):
        raise RuntimeError(f'{title} service unavailable')
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{title} failed or timed out')
    return future.result()


def get_scene(node, client):
    req = GetPlanningScene.Request()
    req.components.components = (PlanningSceneComponents.ROBOT_STATE |
                                 PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
                                 PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
    return service(node, client, req, 'GetPlanningScene').scene


def apply_scene(node, client, diff):
    req = ApplyPlanningScene.Request()
    req.scene = diff
    if not service(node, client, req, 'ApplyPlanningScene').success:
        raise RuntimeError('MoveIt rejected scene update; inspect move_group log')


def palm_fk(node, client, state):
    req = GetPositionFK.Request()
    req.header.frame_id = 'base_link'
    req.fk_link_names = [PALM]
    req.robot_state = state
    ans = service(node, client, req, 'GetPositionFK')
    if ans.error_code.val != MoveItErrorCodes.SUCCESS or not ans.pose_stamped:
        raise RuntimeError(f'Palm FK failed: MoveIt error {ans.error_code.val}')
    return ans.pose_stamped[0].pose


def waypoint(pose, delta):
    p = copy.deepcopy(pose)
    p.position.x += delta[0]
    p.position.y += delta[1]
    p.position.z += delta[2]
    return p


def plan_place(node, client, state, current_palm, cup, base, args):
    # Cup world displacement is the same as palm world displacement while rigidly attached.
    safe_z = max(args.clearance_z, cup[2], args.release_z + 0.10)
    up = safe_z - cup[2]
    waypoints_world = [
        (0.0, 0.0, up),
        (args.place_x - cup[0], args.place_y - cup[1], up),
        (args.place_x - cup[0], args.place_y - cup[1], args.release_z - cup[2]),
    ]
    qb_inv = qinv(quat(*base[3:]))
    waypoints = [waypoint(current_palm, rotate(qb_inv, delta))
                 for delta in waypoints_world]

    req = GetCartesianPath.Request()
    req.header.frame_id = 'base_link'
    req.start_state = state
    req.group_name = GROUP
    req.link_name = PALM
    req.waypoints = waypoints
    req.max_step = 0.008  # 8 mm Cartesian resolution
    req.jump_threshold = 2.0
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = 0.10
    req.max_acceleration_scaling_factor = 0.10
    result = service(node, client, req, 'GetCartesianPath', timeout=45.0)
    traj = result.solution.joint_trajectory
    arm = {f'arm_0_joint_{i}' for i in range(1, 8)}
    print(f'Cartesian path fraction: {result.fraction:.3f}, MoveIt code: {result.error_code.val}')
    print(f'Trajectory: {len(traj.points)} points; joints: {list(traj.joint_names)}')
    if result.error_code.val != MoveItErrorCodes.SUCCESS or result.fraction < 0.999:
        raise RuntimeError('No complete collision-free Cartesian path. Nothing executed/detached.')
    if set(traj.joint_names) != arm or len(traj.points) < 2:
        raise RuntimeError('Unexpected or empty trajectory. Nothing executed/detached.')
    if traj.points[-1].time_from_start.sec == 0 and traj.points[-1].time_from_start.nanosec == 0:
        raise RuntimeError('Trajectory has zero duration. Nothing executed/detached.')
    # Guard against abrupt IK jumps between Cartesian samples.
    for i in range(1, len(traj.points)):
        biggest = max(abs(b-a) for a,b in zip(traj.points[i-1].positions,
                                               traj.points[i].positions))
        if biggest > 0.35:
            raise RuntimeError(f'Large joint step {biggest:.3f} rad at point {i}; refuse execution')
    return result.solution


ARM_CONTROLLER_ACTION = '/a200_0000/arm_0_joint_trajectory_controller/follow_joint_trajectory'


def settle_to(node, joint_names, positions, seconds=2.0):
    """Re-send the final point of a finished trajectory straight to the arm controller.

    In Gazebo the arm has stopped a few degrees short of a trajectory's end (seen on
    wrist joint 7 after a loaded lift: -0.069 rad, constant for minutes) while the
    controller still held the right target. Re-commanding the same point closed it to
    <0.001 rad. This only re-asserts where the plan already ended; it adds no motion.
    """
    from control_msgs.action import FollowJointTrajectory
    from builtin_interfaces.msg import Duration as Dur
    client = ActionClient(node, FollowJointTrajectory, ARM_CONTROLLER_ACTION)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in positions]
    pt.time_from_start = Dur(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
    goal.trajectory.points = [pt]
    run_action(node, client, goal, 'settle to trajectory end', timeout=30)
    time.sleep(1.0)


def run_action(node, client, goal, title, timeout=150):
    if not client.wait_for_server(timeout_sec=10):
        raise RuntimeError(f'{title} action server unavailable')
    # A freshly started process can see the server before the server has matched this
    # client; a goal sent in that window is silently dropped and never acknowledged
    # (seen as "Dex3 open goal not acknowledged" after 45 s). Let discovery settle,
    # then re-send if there is no acknowledgement.
    settle = time.monotonic() + 1.0
    while time.monotonic() < settle:
        rclpy.spin_once(node, timeout_sec=0.1)
    future = None
    for attempt in range(1, 4):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20)
        if future.done() and future.result() is not None:
            break
        print(f'{title}: goal not acknowledged after 20 s (attempt {attempt}/3); re-sending',
              flush=True)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{title} goal not acknowledged')
    handle = future.result()
    if not handle.accepted:
        raise RuntimeError(f'{title} goal rejected')
    future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{title} timed out; check arm before doing anything else')
    result = future.result()
    if result.status != GoalStatus.STATUS_SUCCEEDED:
        raise RuntimeError(f'{title} action failed with status {result.status}')
    return result.result


def _newest_gazebo_log():
    import glob
    logs = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs',
                                  'run_*', 'gazebo.log'))
    return max(logs, key=os.path.getmtime) if logs else None


def _count_in_log(path, needle):
    try:
        with open(path, 'rb') as f:
            return f.read().count(needle.encode())
    except (OSError, TypeError):
        return 0


def gazebo_detach():
    # One-shot `gz topic -p` messages are sometimes lost under load. Gazebo (-v 4) logs
    # "Detaching joint" when the weld is really removed: confirm and re-publish.
    cmd = ['gz', 'topic', '-t', '/cup_grasp/detach', '-m', 'gz.msgs.Empty', '-p', '']
    log = _newest_gazebo_log()
    before = _count_in_log(log, 'Detaching joint')
    for attempt in range(1, 6):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        if p.returncode != 0:
            raise RuntimeError('Gazebo detach command failed: ' + (p.stderr or p.stdout))
        if log is None:
            print('Gazebo detach command published (no gazebo.log found to confirm it).')
            return
        end = time.time() + 2.5
        while time.time() < end:
            if _count_in_log(log, 'Detaching joint') > before:
                print(f'Gazebo confirmed the detach (request {attempt}).')
                return
            time.sleep(0.25)
    # Not an error by itself: the joint may already have been detached.
    print('Gazebo detach published 5 times without a new "Detaching joint" log line '
          '(already detached?).')


def detach_in_moveit(node, get_client, apply_client, base, physical_cup):
    # Remove attached collision object, then restore the cup in the world scene.
    diff = PlanningScene()
    diff.is_diff = True
    diff.robot_state.is_diff = True
    attached = AttachedCollisionObject()
    attached.link_name = PALM
    attached.object.id = CUP_ID
    attached.object.operation = CollisionObject.REMOVE
    diff.robot_state.attached_collision_objects.append(attached)
    apply_scene(node, apply_client, diff)

    after = get_scene(node, get_client)
    if any(x.object.id == CUP_ID for x in after.robot_state.attached_collision_objects):
        raise RuntimeError('MoveIt did not detach target_cup; do not continue blindly.')

    co = CollisionObject()
    co.header.frame_id = 'base_link'
    co.id = CUP_ID
    co.operation = CollisionObject.ADD
    co.pose.orientation.w = 1.0
    geom = SolidPrimitive()
    geom.type = SolidPrimitive.CYLINDER
    geom.dimensions = [CUP_HEIGHT, CUP_RADIUS]
    co.primitives = [geom]
    co.primitive_poses = [cup_pose_in_base(physical_cup, base)]
    world_diff = PlanningScene()
    world_diff.is_diff = True
    world_diff.robot_state.is_diff = True
    world_diff.world.collision_objects = [co]
    apply_scene(node, apply_client, world_diff)
    checked = get_scene(node, get_client)
    if not any(x.id == CUP_ID for x in checked.world.collision_objects):
        raise RuntimeError('MoveIt cup was detached but not restored to world geometry.')
    print('MoveIt verification: target_cup detached and added back to world.')


def open_dex3(node, client):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(HAND_OPEN)
    p = JointTrajectoryPoint()
    p.positions = list(HAND_OPEN.values())
    p.time_from_start = Duration(sec=3, nanosec=0)
    goal.trajectory.points = [p]
    outcome = run_action(node, client, goal, 'Dex3 open', timeout=30)
    if outcome.error_code != 0:
        raise RuntimeError(f'Dex3 controller returned error {outcome.error_code}')
    print('Dex3 opened to hand_ready.')
