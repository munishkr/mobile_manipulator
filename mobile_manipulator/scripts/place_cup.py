#!/usr/bin/env python3
"""Place an already-grasped cup using MoveIt 2 + Gazebo Harmonic.

ASSUMPTIONS (verified for the user's robot):
  * Gazebo DetachableJoint is CURRENTLY ATTACHED: /cup_grasp/detach.
  * MoveIt has target_cup attached to left_hand_palm_link (scene sync already run).
  * Robot base is stationary and Gazebo's model origin coincides with base_link.
  * The cup remains approximately upright. The table is x=1.15 +/- 0.40,
    y=0 +/- 0.30, tabletop z=0.75 m; cup height 0.12 m.

Default is PLAN ONLY. Nothing moves or detaches without --execute.
Only use in the user's tabletop simulation, not on physical equipment.

Examples:
  python3 place_cup.py
  python3 place_cup.py --place-x 1.10 --place-y 0.16
  python3 place_cup.py --place-x 1.10 --place-y 0.16 --execute

No TF topic subscriptions: pose is read from Gazebo, and palm FK from MoveIt.
"""

import argparse
import copy
import math
import os
import re
import subprocess
import sys
import time

# gz CLI calls must use the same transport interface as the sim (runRobot.sh exports
# GZ_IP=127.0.0.1; Wi-Fi multicast broke gz-transport in long sessions).
os.environ.setdefault('GZ_IP', '127.0.0.1')

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject,
                             PlanningScene, PlanningSceneComponents,
                             MoveItErrorCodes)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetPlanningScene, GetPositionFK)
from rclpy.action import ActionClient
from rclpy.node import Node
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
    future = client.send_goal_async(goal)
    # 45 s: under heavy sim load goal acceptance has taken >15 s (Dex3 open at release)
    rclpy.spin_until_future_complete(node, future, timeout_sec=45)
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


def gazebo_detach():
    cmd = ['gz', 'topic', '-t', '/cup_grasp/detach', '-m', 'gz.msgs.Empty', '-p', '']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    if p.returncode != 0:
        raise RuntimeError('Gazebo detach command failed: ' + (p.stderr or p.stdout))
    print('Gazebo detach command published (transport delivery cannot be verified by this command).')


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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--place-x', type=float, default=1.10, help='World-frame destination X (m)')
    ap.add_argument('--place-y', type=float, default=0.16, help='World-frame destination Y (m)')
    ap.add_argument('--release-z', type=float, default=0.83,
                    help='Cup CENTER Z for release; table top 0.75, cup half-height 0.06')
    ap.add_argument('--clearance-z', type=float, default=0.96,
                    help='Cup center height during lateral translation')
    ap.add_argument('--robot-model', default='a200_0000/robot', help='Gazebo robot model name')
    ap.add_argument('--execute', action='store_true', help='EXECUTE arm motion, detach, and open hand')
    ap.add_argument('--skip-open', action='store_true', help='Leave Dex3 closed after detachment')
    return ap.parse_args()


def main():
    args = parse_args()
    if not (0.80 <= args.place_x <= 1.50 and -0.25 <= args.place_y <= 0.25):
        raise RuntimeError('Placement must remain inside table surface with margin: x 0.80..1.50, y -0.25..0.25')
    if not (0.82 <= args.release_z <= 0.90):
        raise RuntimeError('Unsafe release-z; expected cup center about 0.83 m')
    if args.clearance_z < args.release_z + 0.07:
        raise RuntimeError('clearance-z should exceed release-z by at least 7 cm')

    cup = gazebo_pose('cup')
    base = gazebo_pose(args.robot_model)
    # Keep the cup's world vertical axis upright during the fixed-orientation transfer.
    vertical = math.cos(cup[3])*math.cos(cup[4])
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, vertical))))
    print(f'Gazebo cup XYZ = {tuple(round(x, 4) for x in cup[:3])}, tilt = {tilt_deg:.1f} deg')
    print(f'Place target = ({args.place_x:.3f}, {args.place_y:.3f}, {args.release_z:.3f}) m WORLD')
    if tilt_deg > 20.0:
        raise RuntimeError('Cup is too tilted for a fixed-orientation placement. Nothing moved.')

    rclpy.init()
    node = Node('place_cup_moveit_gazebo')
    try:
        gsc = node.create_client(GetPlanningScene, '/get_planning_scene')
        asc = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        fk = node.create_client(GetPositionFK, '/compute_fk')
        cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        scene = get_scene(node, gsc)
        attached = [a for a in scene.robot_state.attached_collision_objects
                    if a.object.id == CUP_ID and a.link_name == PALM]
        if len(attached) != 1:
            raise RuntimeError('MoveIt target_cup must already be attached to palm. Re-run working scene sync.')
        if any(co.id == CUP_ID for co in scene.world.collision_objects):
            raise RuntimeError('MoveIt also has a stationary cup. Resolve duplicate before planning.')
        palm = palm_fk(node, fk, scene.robot_state)
        print('MoveIt palm in base_link:', tuple(round(v, 4) for v in
              (palm.position.x, palm.position.y, palm.position.z)))
        trajectory = plan_place(node, cart, scene.robot_state, palm, cup, base, args)
        if not args.execute:
            print('\nPLAN-ONLY success. No Gazebo, robot, or MoveIt state changed.')
            print('To run the placement from the CURRENT held state, add --execute.')
            return 0

        print('Executing collision-checked Kinova Cartesian placement...')
        exe = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        outcome = run_action(node, exe, goal, 'Kinova placement')
        if outcome.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'Kinova execution error: {outcome.error_code.val}. Cup stays attached.')

        current_cup = gazebo_pose('cup')
        err = math.dist(current_cup[:3], (args.place_x, args.place_y, args.release_z))
        print('Gazebo physical cup at:', tuple(round(x, 4) for x in current_cup[:3]),
              f'(target error {err*1000:.0f} mm)')
        if err > 0.05:
            raise RuntimeError('Gazebo cup did NOT follow arm to destination; refuse detach!')

        gazebo_detach()
        # Let the cylinder settle on the table before rebuilding the world obstacle.
        time.sleep(1.5)
        current_cup = gazebo_pose('cup')
        if math.hypot(current_cup[0] - args.place_x,
                      current_cup[1] - args.place_y) > 0.08:
            raise RuntimeError('Cup moved away from placement after detach; MoveIt scene left unchanged.')
        if current_cup[2] < TABLE_TOP - 0.02:
            raise RuntimeError('Cup fell below tabletop; MoveIt scene left unchanged.')
        detach_in_moveit(node, gsc, asc, base, current_cup)
        if not args.skip_open:
            gripper = ActionClient(node, FollowJointTrajectory,
                                   '/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory')
            open_dex3(node, gripper)
        print('PLACEMENT sequence finished. Check the Gazebo cup settles on the table.')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
