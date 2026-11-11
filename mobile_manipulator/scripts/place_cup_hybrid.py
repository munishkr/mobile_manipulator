#!/usr/bin/env python3
"""Hybrid placement: OMPL lift/transfer + orientation-locked Cartesian lowering.

Requires place_cup.py in the same directory for the verified Gazebo/MoveIt
interfaces. Default: plan only. With --execute, move through the three stages
but KEEP the cup attached. Add --release to detach and open the fingers after
reaching / verifying the placement pose. Alternatively, --release-only
performs a guarded release later without commanding any arm motion.

The third stage uses a straight lowering motion with the palm orientation
held constant, instead of accepting a random OMPL wrist-twisting trajectory.

Unlike v2, this script executes the EXACT trajectories it accepted during
preflight; it never silently replans during execution. It compares actual arm
joint states with planned trajectory starts / endpoints before continuing.

This is for the Gazebo tabletop simulation only. Monitor the robot in Gazebo.
"""

import argparse
import copy
import math
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, MoveItErrorCodes, OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionFK, ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from control_msgs.action import FollowJointTrajectory

import place_cup as pc

ARM = [f'arm_0_joint_{i}' for i in range(1, 8)]


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--place-x', type=float, default=1.10)
    ap.add_argument('--place-y', type=float, default=0.16)
    ap.add_argument('--release-z', type=float, default=0.88,
                    help='Cup center Z in the world at release (m); default .88 gives cup bottom 7 cm above table')
    ap.add_argument('--clearance-z', type=float, default=1.10,
                    help='Cup center world Z during lateral transfer (m)')
    ap.add_argument('--robot-model', default='a200_0000/robot')
    ap.add_argument('--pipeline', default='',
                    help='MoveIt pipeline ID (empty = configured default)')
    ap.add_argument('--planner', default='',
                    help='Planner ID (empty = configured default; try RRTConnectkConfigDefault if needed)')
    ap.add_argument('--constrain-path', action='store_true',
                    help='Require orientation limits along the full path (can make OMPL planning harder)')
    ap.add_argument('--max-palm-rotation-deg', type=float, default=20.0,
                    help='Reject planned trajectories with large palm rotation; checked with MoveIt FK')
    ap.add_argument('--planning-time', type=float, default=15.0)
    ap.add_argument('--attempts', type=int, default=8)
    ap.add_argument('--execute', action='store_true',
                    help='Execute the accepted trajectories; keep the cup attached unless --release is also set')
    ap.add_argument('--release', action='store_true',
                    help='After successful execution and pose verification, detach the cup and open the fingers')
    ap.add_argument('--release-only', action='store_true',
                    help='Release a cup ALREADY at the placement pose; does not move the arm')
    ap.add_argument('--rotation-retries', type=int, default=5,
                    help='Max independent plans per segment to find one within palm rotation limit')
    ap.add_argument('--joint-tolerance', type=float, default=0.01,
                    help='Maximum joint start/end deviation (radians), default MoveIt start tolerance')
    ap.add_argument('--skip-open', action='store_true')
    return ap.parse_args()


def create_goal_constraints(target, position_radius=0.015, orientation_tol=0.12):
    """Pose goal region at the palm; OMPL samples joint states satisfying it."""
    c = Constraints()
    pcst = PositionConstraint()
    pcst.header.frame_id = 'base_link'
    pcst.link_name = pc.PALM
    pcst.weight = 1.0
    pcst.constraint_region.primitives = [SolidPrimitive(
        type=SolidPrimitive.SPHERE,
        dimensions=[position_radius],
    )]
    region_pose = Pose()
    region_pose.position = copy.deepcopy(target.position)
    region_pose.orientation.w = 1.0
    pcst.constraint_region.primitive_poses = [region_pose]
    c.position_constraints = [pcst]

    oc = OrientationConstraint()
    oc.header.frame_id = 'base_link'
    oc.link_name = pc.PALM
    oc.orientation = copy.deepcopy(target.orientation)
    oc.absolute_x_axis_tolerance = orientation_tol
    oc.absolute_y_axis_tolerance = orientation_tol
    oc.absolute_z_axis_tolerance = orientation_tol
    oc.weight = 1.0
    c.orientation_constraints = [oc]
    return c


def create_upright_path_constraints(start_palm):
    """Limit wrist rotation along paths, not merely at endpoints."""
    c = Constraints()
    oc = OrientationConstraint()
    oc.header.frame_id = 'base_link'
    oc.link_name = pc.PALM
    oc.orientation = copy.deepcopy(start_palm.orientation)
    oc.absolute_x_axis_tolerance = 0.35   # about 20 degrees
    oc.absolute_y_axis_tolerance = 0.35
    oc.absolute_z_axis_tolerance = 0.65   # yaw freedom about 37 degrees
    oc.weight = 1.0
    c.orientation_constraints = [oc]
    return c


def plan_segment(node, plan_client, start_state, target_palm, upright_constraints,
                 args, label):
    req = GetMotionPlan.Request()
    m = req.motion_plan_request
    m.group_name = pc.GROUP
    m.pipeline_id = args.pipeline
    m.planner_id = args.planner
    m.num_planning_attempts = args.attempts
    m.allowed_planning_time = args.planning_time
    m.max_velocity_scaling_factor = 0.08
    m.max_acceleration_scaling_factor = 0.08
    m.start_state = pc.normalized_start(start_state)
    m.start_state.is_diff = False
    m.goal_constraints = [create_goal_constraints(target_palm)]
    if args.constrain_path:
        m.path_constraints = copy.deepcopy(upright_constraints)

    result = pc.service(node, plan_client, req, 'GetMotionPlan',
                        timeout=max(30.0, args.planning_time * args.attempts + 10.0))
    response = result.motion_plan_response
    traj = pc.unwrap_to_start(response.trajectory.joint_trajectory, start_state)
    print(f'{label}: code={response.error_code.val}, '
          f'points={len(traj.points)}, planning_time={response.planning_time:.2f}s')
    if hasattr(response.error_code, 'message') and response.error_code.message:
        print('  MoveIt message:', response.error_code.message)
    if hasattr(response.error_code, 'source') and response.error_code.source:
        print('  MoveIt source:', response.error_code.source)
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f'{label}: planning failed with code {response.error_code.val}; '
                           'cup remains attached. No release performed. '
                           'If code=99999 and planning_time=0, inspect move_group logs for '
                           'pipeline resolution or exceptions.')
    if set(traj.joint_names) != set(ARM) or len(traj.points) < 2:
        raise RuntimeError(f'{label}: unexpected trajectory joints / empty path: '
                           f'{list(traj.joint_names)}')
    t = traj.points[-1].time_from_start
    if t.sec == 0 and t.nanosec == 0:
        raise RuntimeError(f'{label}: zero-duration trajectory')
    return response.trajectory


LAST_CARTESIAN_FRACTION = [1.0]


def plan_vertical_lowering(node, cart_client, fk_client, start_state,
                           base_world, lowering_m, label, min_fraction=0.999):
    """Plan a straight downward move from the Stage-2 ending *actual FK*.

    Preserving this particular quaternion avoids asking for another wrist
    orientation or KDL branch during the lower. Collision checking stays ON.
    """
    start_palm = pc.palm_fk(node, fk_client, start_state)
    inv_base_q = pc.qinv(pc.quat(*base_world[3:]))
    local_translation = pc.rotate(inv_base_q, (0.0, 0.0, lowering_m))
    goal_palm = pc.waypoint(start_palm, local_translation)
    # waypoint(...) copies orientation unchanged.

    req = GetCartesianPath.Request()
    req.header.frame_id = 'base_link'
    req.start_state = pc.normalized_start(start_state)
    req.start_state.is_diff = False
    req.group_name = pc.GROUP
    req.link_name = pc.PALM
    req.waypoints = [goal_palm]
    req.max_step = 0.004
    req.jump_threshold = 0.0  # use absolute joint jump threshold below
    req.revolute_jump_threshold = 0.20
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = 0.05
    req.max_acceleration_scaling_factor = 0.05

    ans = pc.service(node, cart_client, req, 'GetCartesianPath', timeout=60.0)
    traj = pc.unwrap_to_start(ans.solution.joint_trajectory, start_state)
    LAST_CARTESIAN_FRACTION[0] = ans.fraction
    print(f'{label}: Cartesian fraction={ans.fraction:.3f}, '
          f'code={ans.error_code.val}, points={len(traj.points)}')
    if (ans.error_code.val != MoveItErrorCodes.SUCCESS or
            ans.fraction < min_fraction or len(traj.points) < 2):
        raise RuntimeError(f'{label}: no complete collision-checked '
                           'orientation-locked descent. Nothing moved/released. '
                           'A higher --release-z (e.g. 0.90) is one safer test; '
                           'do not disable collision checks.')
    if set(traj.joint_names) != set(ARM):
        raise RuntimeError(f'{label}: unexpected trajectory joints: {traj.joint_names}')
    end_time = traj.points[-1].time_from_start
    if not (end_time.sec or end_time.nanosec):
        raise RuntimeError(f'{label}: invalid zero-duration trajectory')
    for i in range(1, len(traj.points)):
        delta = max(abs(b-a) for a,b in zip(traj.points[i-1].positions,
                                             traj.points[i].positions))
        if delta > 0.20:
            raise RuntimeError(f'{label}: adjacent joint jump {delta:.3f} rad '
                               f'at point {i}; refusing execution')
    return ans.solution


def state_at_point(start_state, trajectory, point):
    """Copy a robot state and set the arm joints from one trajectory point."""
    state = copy.deepcopy(start_state)
    lookup = {name: i for i, name in enumerate(state.joint_state.name)}
    for name, value in zip(trajectory.joint_trajectory.joint_names, point.positions):
        if name not in lookup:
            raise RuntimeError(f'Planning scene state missing joint {name}')
        state.joint_state.position[lookup[name]] = value
    state.is_diff = False
    return state


def check_palm_rotation(node, fk_client, start_state, trajectory,
                        reference_quaternion, degrees_limit, label):
    """Sample trajectory FK; reject unplanned large cup/palm tilting.

    This compares palm orientation to the initial grasp pose; it does not
    require wrist orientation constraints inside OMPL.
    """
    ref = reference_quaternion
    ref_norm = math.sqrt(sum(x*x for x in ref))
    if ref_norm < 1e-10:
        raise RuntimeError('Invalid reference palm quaternion')
    ref = tuple(x/ref_norm for x in ref)
    pts = trajectory.joint_trajectory.points
    step = max(1, len(pts) // 50)
    samples = list(range(0, len(pts), step))
    if samples[-1] != len(pts)-1:
        samples.append(len(pts)-1)
    biggest = 0.0
    for i in samples:
        state = state_at_point(start_state, trajectory, pts[i])
        qmsg = pc.palm_fk(node, fk_client, state).orientation
        q = (qmsg.x, qmsg.y, qmsg.z, qmsg.w)
        norm = math.sqrt(sum(x*x for x in q))
        if norm < 1e-10:
            raise RuntimeError(f'{label}: invalid FK quaternion at waypoint {i}')
        dot = abs(sum(x*y for x, y in zip(ref, (v/norm for v in q))))
        degrees = math.degrees(2 * math.acos(min(1.0, max(0.0, dot))))
        biggest = max(biggest, degrees)
    print(f'  {label}: sampled maximum palm rotation={biggest:.1f} deg '
          f'(limit={degrees_limit:.1f} deg)')
    if biggest > degrees_limit:
        raise RuntimeError(f'{label}: planned palm rotation exceeds configured limit; '
                           'no motion or detachment performed. Try an intermediate '
                           'joint-space waypoint or --constrain-path.')


def state_after(start_state, trajectory):
    """Hypothetical next robot state for read-only planning of later stages."""
    state = copy.deepcopy(start_state)
    last = trajectory.joint_trajectory.points[-1]
    indices = {name: i for i, name in enumerate(state.joint_state.name)}
    for name, pos in zip(trajectory.joint_trajectory.joint_names, last.positions):
        if name not in indices:
            raise RuntimeError(f'Planning scene missing joint: {name}')
        state.joint_state.position[indices[name]] = pos
    state.is_diff = False
    return state


def make_stages(cup, args):
    z_high = max(args.clearance_z, cup[2], args.release_z + 0.10)
    return [
        ('1/3 lift to clearance', (cup[0], cup[1], z_high)),
        ('2/3 transfer above target', (args.place_x, args.place_y, z_high)),
        ('3/3 lower for release', (args.place_x, args.place_y, args.release_z)),
    ]


def palm_target_for_cup_target(palm, cup_world, target_world, base_world):
    world_delta = tuple(target_world[i] - cup_world[i] for i in range(3))
    inv_base_q = pc.qinv(pc.quat(*base_world[3:]))
    base_delta = pc.rotate(inv_base_q, world_delta)
    return pc.waypoint(palm, base_delta)


def check_setup(scene):
    attached = [a for a in scene.robot_state.attached_collision_objects
                if a.object.id == pc.CUP_ID and a.link_name == pc.PALM]
    if len(attached) != 1:
        raise RuntimeError('MoveIt target_cup must be attached to left_hand_palm_link first')
    if any(obj.id == pc.CUP_ID for obj in scene.world.collision_objects):
        raise RuntimeError('target_cup is also present in world collision objects')


def validate_args(args):
    if not (0.80 <= args.place_x <= 1.50 and -0.25 <= args.place_y <= 0.25):
        raise RuntimeError('Placement must be inside table margins')
    if not (0.82 <= args.release_z <= 0.90):
        raise RuntimeError('release-z must be 0.82..0.90 m')
    if args.clearance_z < args.release_z + 0.07:
        raise RuntimeError('clearance-z must exceed release-z by 0.07 m')



def get_arm_joint_positions(state):
    joints = dict(zip(state.joint_state.name, state.joint_state.position))
    missing = [j for j in ARM if j not in joints]
    if missing:
        raise RuntimeError(f'Missing arm joints in robot state: {missing}')
    return joints


def check_joint_match(actual_state, trajectory, point, tol, phase, per_joint_tol=None):
    measured = get_arm_joint_positions(actual_state)
    diffs = {name: abs(measured[name] - value) for name, value in
             zip(trajectory.joint_trajectory.joint_names, point.positions)}
    joint = max(diffs, key=diffs.get)
    error = diffs[joint]
    print(f'  {phase}: maximum joint deviation={error:.6f} rad ({joint})')
    limits = {name: (per_joint_tol or {}).get(name, tol) for name in diffs}
    over = {n: d for n, d in diffs.items() if d > limits[n]}
    if over:
        joint, error = max(over.items(), key=lambda kv: kv[1])
        tol = limits[joint]
    if over:
        raise RuntimeError(f'{phase}: joint deviation exceeds {tol:.5f} rad; '
                           'robot stays in its current state; cup stays attached')


def safe_plan_segment(node, plan_client, fk_client, start_state, target,
                      upright, args, reference_q, title):
    for attempt in range(1, args.rotation_retries + 1):
        print(f'  Planning {title}: attempt {attempt}/{args.rotation_retries}')
        try:
            trajectory = plan_segment(node, plan_client, start_state, target,
                                      upright, args, title)
            check_palm_rotation(node, fk_client, start_state, trajectory,
                                reference_q, args.max_palm_rotation_deg, title)
            return trajectory
        except RuntimeError as exc:
            print(f'  Rejected plan: {exc}')
    raise RuntimeError(f'{title}: no acceptable trajectory after '
                       f'{args.rotation_retries} attempts; nothing executed/detached')


def release_cup(node, scene_client, apply_client, args):
    # Called only after position checks and verified MoveIt attachment.
    print('Releasing cup in Gazebo ...')
    pc.gazebo_detach()
    time.sleep(1.5)
    settled_cup = pc.gazebo_pose('cup')
    if math.hypot(settled_cup[0]-args.place_x,
                  settled_cup[1]-args.place_y) > 0.08:
        raise RuntimeError('Cup left placement region; MoveIt still attached')
    if settled_cup[2] < pc.TABLE_TOP - 0.02:
        raise RuntimeError('Cup below tabletop; MoveIt still attached')
    pc.detach_in_moveit(node, scene_client, apply_client,
                        pc.gazebo_pose(args.robot_model), settled_cup)
    if not args.skip_open:
        gripper = ActionClient(node, FollowJointTrajectory,
            '/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory')
        pc.open_dex3(node, gripper)
    print('PLACEMENT COMPLETE. Verify cup in Gazebo.')


def main():
    args = get_args()
    validate_args(args)
    if args.release and not args.execute:
        raise RuntimeError('--release requires --execute')
    if args.release_only and (args.release or args.execute):
        raise RuntimeError('--release-only cannot be combined with --execute or --release')
    if args.rotation_retries < 1 or not (0 < args.joint_tolerance <= 0.05):
        raise RuntimeError('Invalid retries or joint tolerance')

    cup = pc.gazebo_pose('cup')
    base = pc.gazebo_pose(args.robot_model)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0,
                           math.cos(cup[3]) * math.cos(cup[4])))))
    print(f'Gazebo cup: xyz={tuple(round(v, 4) for v in cup[:3])}; tilt={tilt:.1f} deg')
    if tilt > 20.0:
        raise RuntimeError('Cup significantly tilted; refusing placement')
    stages = make_stages(cup, args)
    print('Destination:', stages[-1][1], 'm in Gazebo world frame')
    print('Mode:', 'RELEASE ONLY' if args.release_only else
          ('EXECUTE + RELEASE' if args.release else
          ('EXECUTE (NO RELEASE)' if args.execute else 'PLAN ONLY')))
    print('Strategy: OMPL for lift/transfer; fixed-orientation Cartesian descent')
    print('Planner selection:', 'pipeline=', repr(args.pipeline),
          'planner=', repr(args.planner), 'path constraint=', args.constrain_path)

    rclpy.init()
    node = Node('place_cup_hybrid')
    try:
        scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
        apply_client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        fk_client = node.create_client(GetPositionFK, '/compute_fk')
        plan_client = node.create_client(GetMotionPlan, '/plan_kinematic_path')
        cart_client = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        scene = pc.get_scene(node, scene_client)
        check_setup(scene)
        if args.release_only:
            if math.dist(cup[:3], stages[-1][1]) > 0.04:
                raise RuntimeError('Cup is not within 40 mm of the requested '
                                   'placement target; refusing release-only')
            release_cup(node, scene_client, apply_client, args)
            return 0
        palm_start = pc.palm_fk(node, fk_client, scene.robot_state)
        upright = create_upright_path_constraints(palm_start)
        initial_q = (palm_start.orientation.x, palm_start.orientation.y,
                     palm_start.orientation.z, palm_start.orientation.w)

        # IMPORTANT: save the exact prevalidated plans for execution.
        plans = []
        virtual_state = copy.deepcopy(scene.robot_state)
        virtual_cup = cup[:3]
        for title, target_cup in stages:
            if math.dist(virtual_cup, target_cup) < 0.005:
                print(f'{title}: already at target; no motion needed')
                plans.append((title, target_cup, None))
                virtual_cup = target_cup
                continue
            if title.startswith('1/3') and abs(target_cup[0] - virtual_cup[0]) < 0.005 \
                    and abs(target_cup[1] - virtual_cup[1]) < 0.005:
                # Lift straight up with the wrist orientation locked. OMPL lifts swung
                # the palm >20 deg on all 5 attempts in an unattended run (cup would tip).
                try:
                    # Partial is fine: the Kinova reaches a limit ~10 cm up from this
                    # grasp. Stage 2 (OMPL) then only covers the short remaining rise.
                    rise = target_cup[2] - virtual_cup[2]
                    traj = plan_vertical_lowering(
                        node, cart_client, fk_client, virtual_state, base,
                        rise, title, min_fraction=0.4)
                    check_palm_rotation(node, fk_client, virtual_state, traj,
                                        initial_q, args.max_palm_rotation_deg, title)
                    reached = virtual_cup[2] + rise * LAST_CARTESIAN_FRACTION[0]
                    target_cup = (target_cup[0], target_cup[1], reached)
                    print(f'  straight lift reaches cup z={reached:.3f}')
                except RuntimeError as exc:
                    print(f'  straight lift unavailable ({exc}); falling back to OMPL')
                    target_palm = palm_target_for_cup_target(palm_start, cup, target_cup, base)
                    traj = safe_plan_segment(node, plan_client, fk_client, virtual_state,
                                             target_palm, upright, args, initial_q, title)
            elif title.startswith('3/3'):
                # Stage 3 starts at the actual hypothetical end-of-Stage-2
                # rather than assuming the pose-goal tolerance is zero.
                delta_z = target_cup[2] - virtual_cup[2]
                traj = plan_vertical_lowering(
                    node, cart_client, fk_client, virtual_state, base,
                    delta_z, title)
                check_palm_rotation(node, fk_client, virtual_state, traj,
                                    initial_q, args.max_palm_rotation_deg,
                                    title)
            else:
                target_palm = palm_target_for_cup_target(
                    palm_start, cup, target_cup, base)
                traj = safe_plan_segment(
                    node, plan_client, fk_client, virtual_state, target_palm,
                    upright, args, initial_q, title)
            plans.append((title, target_cup, traj))
            virtual_state = state_after(virtual_state, traj)
            virtual_cup = target_cup

        print('\nHybrid plan validated: OMPL lift/transfer + Cartesian descent.')
        if not args.execute:
            print('PLAN ONLY: no motion, detach, or scene changes.')
            return 0

        executor = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        for title, target_cup, traj in plans:
            if traj is None:
                print(f'EXEC {title}: skipped (validated no-op)')
                continue

            # Confirm that previous stage reached the exact start of THIS plan.
            # Never replan behind the operator's back.
            measured = pc.get_scene(node, scene_client)
            check_setup(measured)
            check_joint_match(measured.robot_state, traj,
                              traj.joint_trajectory.points[0],
                              args.joint_tolerance, f'{title} start')
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj
            print(f'EXEC {title}: executing prevalidated trajectory '
                  f'({len(traj.joint_trajectory.points)} points)')
            planned_s = Duration.from_msg(
                traj.joint_trajectory.points[-1].time_from_start).nanoseconds / 1e9
            t_exec = time.time()
            outcome = pc.run_action(node, executor, goal, title, timeout=180)
            took = time.time() - t_exec
            # Wall time vs planned (sim) time; RTF < 1 alone gives ~1.2x. A stalled arm
            # shows up here long before MoveIt's own timeout cancels the trajectory.
            print(f'  {title}: executed in {took:.1f} s for {planned_s:.1f} s planned '
                  f'({took / max(planned_s, 1e-3):.2f}x)')
            if outcome.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f'{title}: execution error {outcome.error_code.val}; '
                                   'cup NOT detached')

            # ROS state can lag the Gazebo controller: poll briefly. If the arm has
            # stopped short of the end (seen: wrist joint 7 stuck ~0.07 rad off), re-send
            # the final point once (pc.settle_to), then apply the normal strict check.
            end_pt = traj.joint_trajectory.points[-1]
            settle_end = time.time() + 3.0
            settled_once = False
            while True:
                time.sleep(0.3)
                post = pc.get_scene(node, scene_client)
                try:
                    check_joint_match(post.robot_state, traj, end_pt,
                                      args.joint_tolerance, f'{title} end')
                    break
                except RuntimeError:
                    if time.time() <= settle_end:
                        continue
                    if settled_once:
                        raise
                    print(f'  {title}: arm stopped short of the end; re-sending final point')
                    pc.settle_to(node, traj.joint_trajectory.joint_names, end_pt.positions)
                    settled_once = True
                    settle_end = time.time() + 2.0
            physical_cup = pc.gazebo_pose('cup')
            err = math.dist(physical_cup[:3], target_cup)
            print(f'  Gazebo cup error: {err * 1000:.0f} mm')
            if err > 0.06:
                raise RuntimeError(f'{title}: cup did not follow intended motion; '
                                   'cup NOT detached')

        physical_cup = pc.gazebo_pose('cup')
        if math.dist(physical_cup[:3], stages[-1][1]) > 0.06:
            raise RuntimeError('Cup not near placement pose; refusing release')
        if not args.release:
            print('TRANSFER COMPLETE. Cup remains attached in Gazebo and MoveIt.')
            print('Inspect cup and fingers in Gazebo before requesting release.')
            return 0

        release_cup(node, scene_client, apply_client, args)
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
