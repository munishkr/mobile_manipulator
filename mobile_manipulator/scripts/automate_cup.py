#!/usr/bin/env python3
"""Cup grasping for A200 + Kinova + Dex3 (Gazebo Harmonic), used as a library.

station_pick_and_place.py imports this module for the pick half of a cycle: look pose,
vision, grasp search, approach, confirmed Gazebo attach, MoveIt sync, hand retreat.

Run directly it only has two maintenance modes:
  --teach   Read-only capture of the palm-to-cup pose; writes the calibration JSON.
            Teach once, at a cup pose that is known to be graspable.
  --reset   Detach the cup, put it back at its start pose, update the MoveIt scene.

Do NOT run this on a physical robot. It assumes the known Gazebo tabletop world.
"""

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (PlanningScene, CollisionObject, MoveItErrorCodes,
                             JointConstraint, Constraints, RobotState)
from moveit_msgs.srv import (GetPlanningScene, ApplyPlanningScene,
                             GetPositionFK, GetPositionIK, GetStateValidity,
                             GetMotionPlan, GetCartesianPath)
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import place_cup as pc
import place_cup_hybrid as hybrid

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION = ROOT / 'config' / 'saved_states' / 'cup_pick_calibration.json'
DEFAULT_LOOK_POSE = ROOT / 'config' / 'saved_states' / 'cup_look_pose.json'
ARM = hybrid.ARM
# These are *recorded finger targets*, not proof of a stable friction-only grasp.
# A Gazebo detachable joint is used for the physical transfer.
GRASP = {
    'left_hand_thumb_0_joint': -5.3667897431966955e-05,
    'left_hand_thumb_1_joint': 0.22089482904032778,
    'left_hand_thumb_2_joint': 0.39889607535064037,
    'left_hand_middle_0_joint': -0.6183193447646054,
    'left_hand_middle_1_joint': -0.3999681739034592,
    'left_hand_index_0_joint': -0.6582562698075487,
    'left_hand_index_1_joint': -0.40002847383921947,
}
TABLE_START = (0.95, 0.0, 0.811)


def xyz(pose):
    return (pose.position.x, pose.position.y, pose.position.z)


def quaternion(pose):
    q = pose.orientation
    return (q.x, q.y, q.z, q.w)


def normalized(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-10:
        raise RuntimeError('Invalid zero-length quaternion')
    return tuple(v / n for v in q)


def set_pose(p, position, orientation):
    p.position.x, p.position.y, p.position.z = position
    q = normalized(orientation)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def relative_pose(child, parent):
    """Pose of child in the parent frame (both in same reference frame)."""
    parent_q = normalized(quaternion(parent))
    inv = pc.qinv(parent_q)
    delta = tuple(a-b for a, b in zip(xyz(child), xyz(parent)))
    trans = pc.rotate(inv, delta)
    rot = pc.qmul(inv, normalized(quaternion(child)))
    return {'position': list(trans), 'quaternion': list(normalized(rot))}


def orientation_error_deg(q1, q2):
    a, b = normalized(q1), normalized(q2)
    return math.degrees(2*math.acos(min(1.0, abs(sum(x*y for x,y in zip(a,b))))))


def current_scene(node):
    client = node.create_client(GetPlanningScene, '/get_planning_scene')
    return pc.get_scene(node, client)


def require_world_cup(scene):
    if any(obj.object.id == pc.CUP_ID for obj in scene.robot_state.attached_collision_objects):
        raise RuntimeError('MoveIt cup is ATTACHED; run only from a detached cup-on-table state')
    if not any(obj.id == pc.CUP_ID for obj in scene.world.collision_objects):
        raise RuntimeError('MoveIt world cup is missing; run your existing add_scene_objects.py')


def check_table_cup(cup):
    if abs(cup[2] - TABLE_START[2]) > .045:
        raise RuntimeError(f'Gazebo cup is not on table (Z={cup[2]:.3f}); reset it first')
    tilt = math.degrees(math.acos(max(-1.0,min(1.0,math.cos(cup[3])*math.cos(cup[4])))))
    if tilt > 15:
        raise RuntimeError(f'Gazebo cup tilted {tilt:.1f} degrees; correct before pickup')


def get_fk(node, state):
    client = node.create_client(GetPositionFK, '/compute_fk')
    return pc.palm_fk(node, client, state)


def capture(node, args):
    cup = pc.gazebo_pose('cup')
    base = pc.gazebo_pose(args.robot_model)
    check_table_cup(cup)
    scene = current_scene(node)
    require_world_cup(scene)
    palm_base = get_fk(node, scene.robot_state)
    if args.teach_lower_m:
        # Estimate the palm's former grasp-height pose using the settled
        # post-release arm pose. Only the saved calibration is shifted;
        # the actual arm is NOT commanded.
        base_q_inv = pc.qinv(pc.quat(*base[3:]))
        delta = pc.rotate(base_q_inv, (0.0, 0.0, -args.teach_lower_m))
        palm_base = copy.deepcopy(palm_base)
        palm_base.position.x += delta[0]
        palm_base.position.y += delta[1]
        palm_base.position.z += delta[2]
        print(f'Teach estimate: virtual palm lowered {args.teach_lower_m*1000:.0f} mm, no actual motion')
    cup_base = pc.cup_pose_in_base(cup, base)
    rel = relative_pose(palm_base, cup_base)
    qb = pc.quat(*base[3:])
    world_offset = pc.rotate(qb, tuple(a-b for a,b in zip(xyz(palm_base), xyz(cup_base))))
    palm_world_q = normalized(pc.qmul(qb, quaternion(palm_base)))
    dist = math.dist(xyz(palm_base), xyz(cup_base))
    print('Grasp teach: palm-to-cup center distance:', round(dist*1000), 'mm')
    if not (.055 <= dist <= .17):
        raise RuntimeError('Palm is too close/far for a plausible cup grasp calibration. '
                           'Teach with OPEN fingers surrounding cup, before gripping.')
    joints = dict(zip(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position))
    missing = [j for j in ARM if j not in joints or not math.isfinite(joints[j])]
    if missing:
        raise RuntimeError('Cannot record Kinova seed joints: ' + str(missing))
    data = {
        'format_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'description': 'Taught open-finger palm pose relative to cup; verify once in Gazebo.',
        'palm_in_cup_frame': rel,
        # For this symmetric cylinder with a stationary base, preserve the
        # approach direction in world coordinates regardless of cup yaw.
        'world_fixed_grasp': {
            'palm_minus_cup_xyz_m': list(world_offset),
            'palm_quaternion_xyzw': list(palm_world_q),
        },
        'seed_arm_joints_rad': {j: float(joints[j]) for j in ARM},
        'grasp_finger_joints_rad': GRASP,
        'taught_cup_xyz_world': list(cup[:3]),
        'taught_palm_to_cup_distance_m': dist,
        'virtual_release_height_correction_m': args.teach_lower_m,
    }
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    if args.calibration.exists() and not args.overwrite:
        raise RuntimeError(f'{args.calibration} exists: use --overwrite to replace calibration')
    args.calibration.write_text(json.dumps(data, indent=2) + '\n')
    print('Saved grasp calibration:', args.calibration)
    print('This is an UNVALIDATED grasp reference, not proof of a collision-free approach.')
    print('Run the plan-only mode next. Inspect both hand and cup in Gazebo before executing.')


def load_calibration(path):
    if not path.is_file():
        raise RuntimeError(f'Missing {path}. Teach palm-to-cup geometry ONCE with --teach.')
    data = json.loads(path.read_text())
    if data.get('format_version') != 1:
        raise RuntimeError('Unsupported calibration format')
    rel = data['palm_in_cup_frame']
    if len(rel['position']) != 3 or len(rel['quaternion']) != 4:
        raise RuntimeError('Invalid calibrated transform')
    if any(not math.isfinite(float(v)) for v in rel['position']+rel['quaternion']):
        raise RuntimeError('Non-finite calibration value')
    if 'world_fixed_grasp' not in data:
        raise RuntimeError('Calibration lacks world-fixed palm pose; re-teach it')
    for j in ARM:
        if j not in data['seed_arm_joints_rad']:
            raise RuntimeError(f'Calibration missing {j}')
    return data


def virtual_open_hand(state):
    state = copy.deepcopy(state)
    lookup = {j:i for i,j in enumerate(state.joint_state.name)}
    for j,v in pc.HAND_OPEN.items():
        if j not in lookup:
            raise RuntimeError('Missing Dex3 joint '+j)
        state.joint_state.position[lookup[j]] = float(v)
    state.is_diff = False
    return state


def arm_seed(state, config):
    state = copy.deepcopy(state)
    lookup = {j:i for i,j in enumerate(state.joint_state.name)}
    for j,v in config.items():
        if j in lookup:
            state.joint_state.position[lookup[j]] = float(v)
    state.is_diff = False
    return state


def mix_arm_state(start_state, solved_state):
    state = copy.deepcopy(start_state)
    lookup = dict(zip(solved_state.joint_state.name, solved_state.joint_state.position))
    indices = {j:i for i,j in enumerate(state.joint_state.name)}
    for j in ARM:
        if j not in lookup or j not in indices:
            raise RuntimeError('IK response missing arm joint '+j)
        state.joint_state.position[indices[j]] = lookup[j]
    state.is_diff = False
    return state


def get_validity(node, client, state, label):
    req = GetStateValidity.Request()
    req.robot_state = state
    req.group_name = pc.GROUP
    result = pc.service(node, client, req, 'GetStateValidity', timeout=15)
    if not result.valid:
        pairs = [(c.contact_body_1, c.contact_body_2) for c in result.contacts[:8]]
        raise RuntimeError(f'{label} robot state invalid; contact pairs: {pairs}')


def ik_to_pose(node, ik_client, valid_client, start_state, target, taught_seed, label):
    seeds = [arm_seed(start_state, taught_seed), start_state]   # chosen posture first
    for index, seed in enumerate(seeds, 1):
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = pc.GROUP
        ik.robot_state = copy.deepcopy(seed)
        ik.robot_state.is_diff = False
        ik.ik_link_name = pc.PALM
        ik.pose_stamped.header.frame_id = 'base_link'
        ik.pose_stamped.pose = target
        ik.avoid_collisions = True
        ik.timeout = Duration(sec=2, nanosec=0)
        ans = pc.service(node, ik_client, req, 'GetPositionIK', timeout=12)
        print(f'{label} IK seed {index}: code={ans.error_code.val}')
        if ans.error_code.val != MoveItErrorCodes.SUCCESS:
            continue
        solved = mix_arm_state(start_state, ans.solution)
        try:
            get_validity(node, valid_client, solved, label)
            return solved
        except RuntimeError as exc:
            print('  Rejected IK:', exc)
    raise RuntimeError(f'{label}: no collision-free IK found. No robot motion requested.')


def plan_joint_target(node, planner, start, target, label, planning_time):
    req = GetMotionPlan.Request()
    m = req.motion_plan_request
    m.group_name = pc.GROUP
    m.pipeline_id = ''
    m.planner_id = ''
    m.num_planning_attempts = 4
    m.allowed_planning_time = planning_time
    m.max_velocity_scaling_factor = 0.07
    m.max_acceleration_scaling_factor = 0.07
    m.start_state = pc.normalized_start(start)
    m.start_state.is_diff = False
    positions = dict(zip(target.joint_state.name,target.joint_state.position))
    constraint = Constraints()
    for name in ARM:
        c = JointConstraint()
        c.joint_name = name
        c.position = float(pc.wrap_pi(positions[name]) if name in pc.CONTINUOUS_JOINTS
                           else positions[name])
        c.tolerance_above = 0.005
        c.tolerance_below = 0.005
        c.weight = 1.0
        constraint.joint_constraints.append(c)
    m.goal_constraints = [constraint]
    # OMPL checks collisions at coarse steps along a path, so it sometimes returns a
    # path that brushes the cup; MoveIt's ValidateSolution then fails the whole request
    # (code 99999). A fresh plan almost always avoids it: retry a few times.
    for attempt in range(1, 4):
        result = pc.service(node, planner, req, 'GetMotionPlan',
                            timeout=max(35, 4*planning_time+12))
        response = result.motion_plan_response
        traj = response.trajectory
        print(f'{label}: code={response.error_code.val}, '
              f'points={len(traj.joint_trajectory.points)}, '
              f'planning_time={response.planning_time:.2f}s'
              + (f' (attempt {attempt}/3)' if attempt > 1 else ''), flush=True)
        if response.error_code.val == MoveItErrorCodes.SUCCESS:
            break
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f'{label}: RRTConnect/OMPL returned {response.error_code.val}')
    pc.unwrap_to_start(traj.joint_trajectory, start)
    if set(traj.joint_trajectory.joint_names) != set(ARM):
        raise RuntimeError(f'{label}: unexpected trajectory joint names')
    if len(traj.joint_trajectory.points) < 2:
        raise RuntimeError(f'{label}: no motion trajectory')
    end = traj.joint_trajectory.points[-1].time_from_start
    if not (end.sec or end.nanosec):
        raise RuntimeError(f'{label}: zero-duration plan')
    return traj


# Non-continuous Kinova joints and their URDF limits (rad): margin to these is
# what the grasp search maximises.
LAST_GRASP_YAW = ROOT / 'config' / 'saved_states' / 'last_grasp_yaw.json'


def last_grasp_yaw():
    try:
        return float(json.loads(LAST_GRASP_YAW.read_text())['grasp_yaw_rad'])
    except (OSError, ValueError, KeyError):
        return 0.0


KINOVA_LIMITS = {'arm_0_joint_2': 2.41, 'arm_0_joint_4': 2.66, 'arm_0_joint_6': 2.23}


def rotate_fixed_grasp(fixed, yaw):
    """The cup is a cylinder: rotating the taught palm offset/orientation about the
    world vertical through the cup gives an equally valid grasp."""
    qz = pc.quat(0.0, 0.0, yaw)
    off = pc.rotate(qz, tuple(fixed['palm_minus_cup_xyz_m']))
    q = pc.qmul(qz, normalized(fixed['palm_quaternion_xyzw']))
    return {'palm_minus_cup_xyz_m': list(off), 'palm_quaternion_xyzw': list(q)}


def grasp_pose_for(cup, base, fixed):
    world_xyz = tuple(cup[i] + fixed['palm_minus_cup_xyz_m'][i] for i in range(3))
    qb_inv = pc.qinv(pc.quat(*base[3:]))
    local_xyz = pc.rotate(qb_inv, tuple(world_xyz[i]-base[i] for i in range(3)))
    local_quat = pc.qmul(qb_inv, normalized(fixed['palm_quaternion_xyzw']))
    return set_pose(Pose(), local_xyz, local_quat)


def select_grasp(node, scene, args, data):
    """Pick the grasp yaw and arm posture with the most joint-limit margin that still
    allows a full straight lift. Replaces data['world_fixed_grasp'] and the IK seed."""
    import random
    cup = args.vision_cup if getattr(args, 'vision_cup', None) is not None else pc.gazebo_pose('cup')
    base = pc.gazebo_pose(args.robot_model)
    ik = node.create_client(GetPositionIK, '/compute_ik')
    validity = node.create_client(GetStateValidity, '/check_state_validity')
    fk = node.create_client(GetPositionFK, '/compute_fk')
    cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
    taught, seed0 = data['world_fixed_grasp'], data['seed_arm_joints_rad']
    opening = virtual_open_hand(scene.robot_state)
    lift_dz = max(args.clearance_z, args.release_z + 0.10) - cup[2]
    lift_base = pc.rotate(pc.qinv(pc.quat(*base[3:])), (0.0, 0.0, lift_dz))
    rng = random.Random(0)
    found = []
    for yaw_deg in args.grasp_yaws:
        fixed = rotate_fixed_grasp(taught, math.radians(yaw_deg))
        target = grasp_pose_for(cup, base, fixed)
        seeds = [seed0] + [{j: v + rng.uniform(-0.6, 0.6) for j, v in seed0.items()}
                           for _ in range(args.grasp_seeds)]
        best = None
        for seed in seeds:
            req = GetPositionIK.Request()
            r = req.ik_request
            r.group_name = pc.GROUP
            r.robot_state = arm_seed(opening, seed)
            r.robot_state.is_diff = False
            r.ik_link_name = pc.PALM
            r.pose_stamped.header.frame_id = 'base_link'
            r.pose_stamped.pose = target
            r.avoid_collisions = True
            r.timeout = Duration(sec=1, nanosec=0)
            ans = pc.service(node, ik, req, 'GetPositionIK', timeout=10)
            if ans.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            solved = mix_arm_state(opening, ans.solution)
            vreq = GetStateValidity.Request()
            vreq.robot_state = solved
            vreq.group_name = pc.GROUP
            if not pc.service(node, validity, vreq, 'GetStateValidity', timeout=10).valid:
                continue
            joints = dict(zip(solved.joint_state.name, solved.joint_state.position))
            margin = min(lim - abs(joints[j]) for j, lim in KINOVA_LIMITS.items())
            creq = GetCartesianPath.Request()
            creq.header.frame_id = 'base_link'
            creq.start_state = pc.normalized_start(solved)
            creq.start_state.is_diff = False
            creq.group_name = pc.GROUP
            creq.link_name = pc.PALM
            creq.waypoints = [pc.waypoint(pc.palm_fk(node, fk, solved), lift_base)]
            creq.max_step = 0.01
            creq.jump_threshold = 0.0
            creq.revolute_jump_threshold = 0.20
            creq.avoid_collisions = False      # kinematic reach only; cup is held then
            frac = pc.service(node, cart, creq, 'GetCartesianPath', timeout=30).fraction
            cand = (frac >= 0.999, margin, frac, yaw_deg, {j: joints[j] for j in ARM}, fixed)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best:
            found.append(best)
            print(f'  grasp yaw {yaw_deg:+4.0f} deg: straight lift {best[2]*100:5.1f}%, '
                  f'joint-limit margin {best[1]:.2f} rad')
        else:
            print(f'  grasp yaw {yaw_deg:+4.0f} deg: no collision-free IK')
    if not found:
        raise RuntimeError('Grasp search: no reachable grasp at any yaw; nothing moved')
    full, margin, frac, yaw, joints, fixed = max(found, key=lambda c: c[:2])
    print(f'Grasp search: chose yaw {yaw:+.0f} deg (lift {frac*100:.0f}%, margin {margin:.2f} rad)')
    if not full:
        print('  WARNING: no candidate allows a full straight lift; hybrid lift will fall back')
    data['world_fixed_grasp'] = fixed
    data['seed_arm_joints_rad'] = joints
    args.grasp_yaw = math.radians(yaw)
    # The next cycle's post-release retreat runs before its own search: remember this.
    LAST_GRASP_YAW.write_text(json.dumps({'grasp_yaw_rad': args.grasp_yaw}))


def calculate_approach(node, scene, args, data):
    vision = getattr(args, 'vision_cup', None)
    cup = vision if vision is not None else pc.gazebo_pose('cup')
    base = pc.gazebo_pose(args.robot_model)
    check_table_cup(cup)
    require_world_cup(scene)
    if vision is None and args.require_start_cup and math.dist(cup[:3],TABLE_START) > .04:
        raise RuntimeError('Cup is not near original start pose; run --reset first')
    # Cup is axially symmetric; its yaw after placement may be arbitrary.
    # Use the taught WORLD approach direction while the A200 base is stationary.
    fixed = data['world_fixed_grasp']
    grasp = grasp_pose_for(cup, base, fixed)
    pregrasp = copy.deepcopy(grasp)
    if getattr(args, 'backoff_dir', 'palm') == 'palm':
        # Retreat along the horizontal cup->palm line, so the final approach slides
        # the cup straight into the open hand. A pure world -X retreat swept the open
        # index finger through the cup (taught palm sits diagonally from the cup).
        off = fixed['palm_minus_cup_xyz_m']
        norm = math.hypot(off[0], off[1])
        backoff_world = (args.backoff*off[0]/norm, args.backoff*off[1]/norm, args.pregrasp_raise)
    else:
        backoff_world = (-args.backoff, 0.0, args.pregrasp_raise)
    backoff_base = pc.rotate(pc.qinv(pc.quat(*base[3:])),backoff_world)
    pregrasp.position.x += backoff_base[0]
    pregrasp.position.y += backoff_base[1]
    pregrasp.position.z += backoff_base[2]
    print('Vision cup XYZ [world]:' if vision is not None else 'Gazebo cup XYZ:',
          [round(v,4) for v in cup[:3]])
    print('Pregrasp palm XYZ [base_link]:', [round(v,4) for v in xyz(pregrasp)])
    print('Grasp palm XYZ [base_link]:', [round(v,4) for v in xyz(grasp)])
    print('Backoff in world:', tuple(round(v,4) for v in backoff_world))
    return pregrasp, grasp, cup


def plan_cartesian_approach(node, fk, start_state, grasp, label):
    """Straight-line palm move pregrasp -> grasp, orientation locked, collisions on.

    A joint-space (OMPL) approach does not keep the hand on a line, and the open
    index finger swept through the cup in ~2/3 of plans (ValidateSolution failure).
    """
    start_palm = pc.palm_fk(node, fk, start_state)
    goal = copy.deepcopy(grasp)
    goal.orientation = start_palm.orientation   # keep the pregrasp wrist orientation
    cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
    req = GetCartesianPath.Request()
    req.header.frame_id = 'base_link'
    req.start_state = pc.normalized_start(start_state)
    req.start_state.is_diff = False
    req.group_name = pc.GROUP
    req.link_name = pc.PALM
    req.waypoints = [goal]
    req.max_step = 0.004
    req.jump_threshold = 0.0
    req.revolute_jump_threshold = 0.20
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = 0.05
    req.max_acceleration_scaling_factor = 0.05
    ans = pc.service(node, cart, req, 'GetCartesianPath', timeout=60.0)
    traj = ans.solution.joint_trajectory
    print(f'{label}: Cartesian fraction={ans.fraction:.3f}, code={ans.error_code.val}, '
          f'points={len(traj.points)}')
    if ans.error_code.val != MoveItErrorCodes.SUCCESS or ans.fraction < 0.999 or len(traj.points) < 2:
        raise RuntimeError(f'{label}: no complete collision-free straight approach '
                           f'(fraction {ans.fraction:.3f}); try --approach ompl or another --backoff')
    if set(traj.joint_names) != set(ARM):
        raise RuntimeError(f'{label}: unexpected trajectory joints {traj.joint_names}')
    # Continuous joints come back wrapped to [-pi, pi]; put them on the real turn.
    pc.unwrap_to_start(traj, start_state)
    now = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
    first_dev = max(abs(a - now[n]) for n, a in zip(traj.joint_names, traj.points[0].positions))
    if first_dev > 0.012:
        raise RuntimeError(f'{label}: Cartesian path starts {first_dev:.3f} rad from the start state')
    for i in range(1, len(traj.points)):
        jump = max(abs(b - a) for a, b in zip(traj.points[i-1].positions, traj.points[i].positions))
        if jump > 0.20:
            raise RuntimeError(f'{label}: joint jump {jump:.3f} rad at point {i}')
    return ans.solution


def build_approach(node, scene, args, data):
    ik = node.create_client(GetPositionIK, '/compute_ik')
    validity = node.create_client(GetStateValidity, '/check_state_validity')
    planner = node.create_client(GetMotionPlan, '/plan_kinematic_path')
    fk = node.create_client(GetPositionFK, '/compute_fk')
    pregrasp, grasp, initial_cup = calculate_approach(node, scene, args, data)
    opening = virtual_open_hand(scene.robot_state)
    pre_goal = ik_to_pose(node, ik, validity, opening, pregrasp,
                          data['seed_arm_joints_rad'], 'pregrasp')
    first = None
    for attempt in range(1, 4):   # OMPL is randomised; an occasional 99999 is not final
        try:
            first = plan_joint_target(node, planner, opening, pre_goal, '1/2 pregrasp',
                                      args.planning_time)
            break
        except RuntimeError as exc:
            if attempt == 3:
                raise
            print(f'  pregrasp plan attempt {attempt} failed ({exc}); retrying')
    theoretical = hybrid.state_after(opening, first)
    if getattr(args, 'approach', 'cartesian') == 'cartesian':
        second = plan_cartesian_approach(node, fk, theoretical, grasp, '2/2 approach')
    else:
        goal = ik_to_pose(node, ik, validity, theoretical, grasp,
                          data['seed_arm_joints_rad'], 'grasp approach')
        second = plan_joint_target(node, planner, theoretical, goal, '2/2 approach',
                                   args.planning_time)
    for title, start, traj, target_pose in (
            ('pregrasp', opening, first, pregrasp),
            ('grasp approach', theoretical, second, grasp)):
        finish = hybrid.state_after(start,traj)
        actual = pc.palm_fk(node, fk, finish)
        dist = math.dist(xyz(actual), xyz(target_pose))
        turn = orientation_error_deg(quaternion(actual),quaternion(target_pose))
        print(f'  {title} FK endpoint: position_error={dist*1000:.1f}mm, '
              f'orientation_error={turn:.1f}deg')
        if dist > .025 or turn > 8.0:
            raise RuntimeError(f'{title} endpoint not sufficiently close to taught target')
    return (("pregrasp",first),("grasp approach",second)), initial_cup


def command_hand(node, target, label):
    hand = ActionClient(node, FollowJointTrajectory,
        '/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory')
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(target)
    p = JointTrajectoryPoint()
    p.positions = [float(target[j]) for j in target]
    p.time_from_start = Duration(sec=3,nanosec=0)
    goal.trajectory.points = [p]
    result = pc.run_action(node,hand,goal,label,timeout=40)
    if result.error_code != 0:
        raise RuntimeError(f'{label}: controller returned {result.error_code}')
    print(label,'complete')


def relax_fingers_to_contact(node, closed_target):
    """Re-command the fingers to where they actually stopped on the welded cup.

    With the cup welded first, fingers that stall on it keep pushing with the controller's
    full error (target is inside the cup) and that force loads the arm through the weld.
    Holding the measured contact pose keeps the look of the grasp without the squeeze."""
    state = current_scene(node).robot_state.joint_state
    measured = dict(zip(state.name, state.position))
    missing = [j for j in closed_target if j not in measured]
    if missing:
        print(f'  finger relax skipped: no joint state for {missing}')
        return
    gaps = {j: closed_target[j] - measured[j] for j in closed_target}
    worst = max(gaps, key=lambda j: abs(gaps[j]))
    print(f'  finger squeeze before relax: max gap {abs(gaps[worst]):.3f} rad ({worst})')
    command_hand(node, {j: measured[j] for j in closed_target}, 'Dex3 relax to contact')


def execute_approach(node, args, data):
    print('Opening Dex3 before approach (cup must be detached) ...')
    command_hand(node, pc.HAND_OPEN, 'Dex3 open')
    time.sleep(.35)
    scene = current_scene(node)
    plans, original_cup = build_approach(node, scene, args, data)
    execution = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
    for label, trajectory in plans:
        current = current_scene(node)
        hybrid.check_joint_match(current.robot_state,trajectory,
                                 trajectory.joint_trajectory.points[0], .012,
                                 label+' start')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory=trajectory
        print('EXEC',label)
        outcome=pc.run_action(node,execution,goal,label,timeout=180)
        if outcome.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{label}: execution error={outcome.error_code.val}')
        time.sleep(.35)
        current = current_scene(node)
        end_pt = trajectory.joint_trajectory.points[-1]
        try:
            hybrid.check_joint_match(current.robot_state,trajectory,end_pt,.012,label+' end')
        except RuntimeError:
            # Arm stopped short of the end: re-send the final point once, re-check strictly.
            print(f'  {label}: arm stopped short of the end; re-sending final point')
            pc.settle_to(node, trajectory.joint_trajectory.joint_names, end_pt.positions)
            current = current_scene(node)
            hybrid.check_joint_match(current.robot_state,trajectory,end_pt,.012,label+' end')
    before_close=pc.gazebo_pose('cup')
    if math.dist(before_close[:3],original_cup[:3])>.035:
        raise RuntimeError('Cup moved significantly during open-finger approach; refusing attach')
    current = current_scene(node)
    palm = get_fk(node,current.robot_state)
    base = pc.gazebo_pose(args.robot_model)
    fixed = data['world_fixed_grasp']
    qb = pc.quat(*base[3:])
    actual_world_palm = tuple(base[i] + pc.rotate(qb, xyz(palm))[i] for i in range(3))
    actual_offset = tuple(actual_world_palm[i] - before_close[i] for i in range(3))
    actual_world_q = pc.qmul(qb, quaternion(palm))
    distance = math.dist(actual_offset, fixed['palm_minus_cup_xyz_m'])
    angle = orientation_error_deg(actual_world_q,fixed['palm_quaternion_xyzw'])
    print(f'Grasp alignment error: {distance*1000:.1f} mm; {angle:.1f} deg')
    if distance>.035 or angle>10:
        raise RuntimeError('Grasp alignment outside tolerance; refusing Gazebo attachment')
    def gazebo_attach():
        # `gz topic -p` is a one-shot publisher: under load the message is sometimes
        # lost, the cup is never welded, and the arm then lifts away without it and
        # knocks it over. Gazebo (-v 4) logs "Attaching entity" when the weld really
        # happens, so confirm against the newest runRobot gazebo.log and re-publish.
        log = newest_gazebo_log()
        before = count_in_file(log, 'Attaching entity')
        for attempt in range(1, 6):
            print(f'Publishing Gazebo attachment request (attempt {attempt}/5) ...', flush=True)
            proc=subprocess.run(['gz','topic','-t','/cup_grasp/attach','-m','gz.msgs.Empty','-p',''],
                                capture_output=True,text=True,timeout=20)
            if proc.returncode != 0:
                raise RuntimeError('Gazebo attach command failed: '+(proc.stderr or proc.stdout))
            if log is None:
                print('WARN: no gazebo.log found; attachment cannot be confirmed')
                return
            end = time.time() + 2.5
            while time.time() < end:
                if count_in_file(log, 'Attaching entity') > before:
                    print('Gazebo confirmed the attachment.')
                    return
                time.sleep(0.25)
        raise RuntimeError('Gazebo never confirmed the cup attachment after 5 requests; '
                           'fingers not closed, cup left where it stands')
    if not args.close_before_attach:
        # Weld the cup where it stands, then close: closing first pushes it ~6 mm and
        # tilts it ~7 deg, the weld keeps that, and every place lands ~1-2 cm short.
        gazebo_attach()
        time.sleep(.6)
    print('Closing Dex3 with taught finger targets ...')
    command_hand(node,data['grasp_finger_joints_rad'],'Dex3 close')
    time.sleep(.3)
    if not args.close_before_attach and not args.no_finger_relax:
        relax_fingers_to_contact(node, data['grasp_finger_joints_rad'])
    cup_after=pc.gazebo_pose('cup')
    tilt = math.degrees(math.acos(max(-1.0, min(1.0,
        math.cos(cup_after[3]) * math.cos(cup_after[4])))))
    print(f'Cup after close: moved {1000*math.dist(cup_after[:3], original_cup[:3]):.1f} mm, '
          f'tilt {tilt:.1f} deg')
    if math.dist(cup_after[:3], original_cup[:3])>.055:
        raise RuntimeError('Cup displaced during finger closing; refusing attachment')
    if tilt > 25:
        raise RuntimeError('Cup tipped during finger closing; refusing attachment')
    if args.close_before_attach:
        gazebo_attach()
    time.sleep(.6)
    # Sync the physical cup into MoveIt; this subprocess owns its own ROS node.
    run_existing('sync_gazebo_cup_to_moveit.py', ['--robot-model',args.robot_model])
    print('Gazebo attach command sent; MoveIt attached scene verified by sync script.')
    print('Physical attachment is finally verified by cup following the first lift stage.')


def execute_planned(node, trajectory, label):
    execution = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
    current = current_scene(node)
    hybrid.check_joint_match(current.robot_state, trajectory,
                             trajectory.joint_trajectory.points[0], .012, label+' start')
    goal = ExecuteTrajectory.Goal()
    goal.trajectory = trajectory
    print('EXEC', label)
    outcome = pc.run_action(node, execution, goal, label, timeout=180)
    if outcome.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f'{label}: execution error={outcome.error_code.val}')
    time.sleep(.35)


def newest_gazebo_log():
    """gazebo.log of the most recent runRobot.sh session, or None."""
    logs = sorted((pc.ROOT / 'logs').glob('run_*/gazebo.log'), key=lambda f: f.stat().st_mtime) \
        if hasattr(pc, 'ROOT') else []
    if not logs:
        logs = sorted((Path(__file__).resolve().parent.parent / 'logs').glob('run_*/gazebo.log'),
                      key=lambda f: f.stat().st_mtime)
    return logs[-1] if logs else None


def count_in_file(path, needle):
    if path is None:
        return 0
    try:
        return path.read_bytes().count(needle.encode())
    except OSError:
        return 0


def retreat_from_cup_if_touching(node, args):
    """After a release MoveIt sees the open palm touching the cup, so every plan from
    that state fails CheckStartStateCollision. If the ONLY contacts are hand<->cup,
    back the palm off in a straight line along cup->palm (+ up), then re-check."""
    validity = node.create_client(GetStateValidity, '/check_state_validity')
    fk = node.create_client(GetPositionFK, '/compute_fk')
    scene = current_scene(node)
    cup_obj = next((o for o in scene.world.collision_objects if o.id == pc.CUP_ID), None)
    if cup_obj is None:
        return
    cp = cup_obj.primitive_poses[0].position if cup_obj.primitive_poses else cup_obj.pose.position
    if cup_obj.primitive_poses and (cup_obj.pose.position.x or cup_obj.pose.position.y):
        cp = cup_obj.pose.position   # object-level pose used instead of primitive pose
    # Decide from the actual collision check. (A palm-to-cup distance pre-filter
    # compared base_link FK with the odom-frame cup, which only matched while the
    # base sat at the odom origin; with a mobile base it silently skipped retreats.)
    req = GetStateValidity.Request()
    req.robot_state = scene.robot_state
    req.group_name = pc.GROUP
    res = pc.service(node, validity, req, 'GetStateValidity', timeout=15)
    bodies = {b for c in res.contacts for b in (c.contact_body_1, c.contact_body_2)}
    fkreq = GetPositionFK.Request()
    fkreq.header.frame_id = cup_obj.header.frame_id      # same frame as the cup object
    fkreq.fk_link_names = [pc.PALM]
    fkreq.robot_state = scene.robot_state
    fkans = pc.service(node, fk, fkreq, 'GetPositionFK')
    palm = fkans.pose_stamped[0].pose.position if fkans.pose_stamped else None
    gap = math.dist((palm.x, palm.y, palm.z), (cp.x, cp.y, cp.z)) if palm else 9.9
    touching = (not res.valid) and pc.CUP_ID in bodies
    # Also retreat when the open hand still surrounds the cup without touching it
    # (right after a release): plans from there clip the cup with the thumb and fail
    # ValidateSolution even though the start state itself is valid.
    if not touching and gap > 0.14:
        return
    others = {b for b in bodies if b != pc.CUP_ID and not b.startswith('left_hand_')}
    if others:
        raise RuntimeError(f'Start state in collision with {sorted(bodies)}; not retreating')
    # Retreat along the actual cup->palm direction (+ up). The stored grasp yaw is in
    # world coordinates from the LAST grasp; after driving to another station the base
    # has turned, so that direction no longer points away from the cup.
    if palm is None:
        raise RuntimeError('retreat from cup: palm FK unavailable')
    off = (palm.x - cp.x, palm.y - cp.y)
    norm = math.hypot(*off)
    if norm < 1e-3:
        raise RuntimeError('retreat from cup: palm is on the cup axis; direction undefined')
    world_delta = (0.08 * off[0] / norm, 0.08 * off[1] / norm, 0.04)   # cup-object frame
    mdof = scene.robot_state.multi_dof_joint_state
    byaw = 0.0
    if 'mobile_base_joint' in mdof.joint_names:
        q = mdof.transforms[list(mdof.joint_names).index('mobile_base_joint')].rotation
        byaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(-byaw), math.sin(-byaw)
    delta = (c * world_delta[0] - s * world_delta[1], s * world_delta[0] + c * world_delta[1],
             world_delta[2])
    start = scene.robot_state
    goal = pc.waypoint(pc.palm_fk(node, fk, start), delta)
    cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
    creq = GetCartesianPath.Request()
    creq.header.frame_id = 'base_link'
    creq.start_state = pc.normalized_start(start)
    creq.start_state.is_diff = False
    creq.group_name = pc.GROUP
    creq.link_name = pc.PALM
    creq.waypoints = [goal]
    creq.max_step = 0.004
    creq.jump_threshold = 0.0
    creq.revolute_jump_threshold = 0.20
    creq.avoid_collisions = False      # start is touching the cup; we move away from it
    creq.max_velocity_scaling_factor = 0.05
    creq.max_acceleration_scaling_factor = 0.05
    ans = pc.service(node, cart, creq, 'GetCartesianPath', timeout=60.0)
    traj = ans.solution.joint_trajectory
    if ans.error_code.val != MoveItErrorCodes.SUCCESS or ans.fraction < 0.999 or len(traj.points) < 2:
        raise RuntimeError(f'retreat from cup: Cartesian fraction {ans.fraction:.3f}')
    pc.unwrap_to_start(traj, start)
    end = hybrid.state_after(start, ans.solution)
    req.robot_state = end
    if not pc.service(node, validity, req, 'GetStateValidity', timeout=15).valid:
        raise RuntimeError('retreat end state still in collision; not moving')
    print(f'Palm {gap*100:.0f} cm from the cup: retreating {[round(v,3) for v in world_delta]} m')
    execute_planned(node, ans.solution, 'retreat from cup')


def move_to_look_pose(node, args):
    """Plan + execute to the wrist-camera look pose (whole table in view)."""
    if not args.look_pose.is_file():
        raise RuntimeError(f'No look pose at {args.look_pose}; move the arm so the cup is '
                           'visible in RViz "Wrist RGB" and save its arm joints there')
    joints = json.loads(args.look_pose.read_text())['arm_joints_rad']
    retreat_from_cup_if_touching(node, args)
    scene = current_scene(node)
    now = dict(zip(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position))
    if max(abs(now[j] - joints[j]) for j in ARM) < .01:
        print('Already at look pose')
        return
    target = RobotState()
    target.joint_state.name = list(ARM)
    target.joint_state.position = [float(joints[j]) for j in ARM]
    planner = node.create_client(GetMotionPlan, '/plan_kinematic_path')
    traj = plan_joint_target(node, planner, scene.robot_state, target, 'look pose',
                             args.planning_time)
    execute_planned(node, traj, 'look pose')


def observe_cup(node, args):
    """Median of N /cup_detector/pose samples -> world (x, y, z, 0, 0, 0)."""
    samples = []
    ready_after = time.time() + 1.0          # ignore frames taken while the arm settled
    def on_pose(msg):
        if time.time() >= ready_after:
            if msg.header.frame_id != 'base_link':
                print('Ignoring cup pose in frame', msg.header.frame_id)
                return
            p = msg.pose.position
            samples.append((p.x, p.y, p.z))
    sub = node.create_subscription(PoseStamped, '/cup_detector/pose', on_pose, 10)
    end = time.time() + args.vision_timeout
    while len(samples) < args.vision_samples and time.time() < end:
        rclpy.spin_once(node, timeout_sec=.1)
    node.destroy_subscription(sub)
    if len(samples) < args.vision_samples:
        raise RuntimeError(f'Only {len(samples)}/{args.vision_samples} cup detections in '
                           f'{args.vision_timeout:.0f} s. Is cup_detector.py running and the '
                           'cup in view (RViz /cup_detector/image)?')
    med = [sorted(s[i] for s in samples)[len(samples)//2] for i in range(3)]
    spread = max(math.dist(s, med) for s in samples)
    print(f'Vision: {len(samples)} detections, median base_link '
          f'{[round(v,4) for v in med]}, spread {spread*1000:.1f} mm')
    if spread > args.vision_max_spread:
        raise RuntimeError(f'Cup detections disagree by {spread*1000:.1f} mm '
                           f'(> {args.vision_max_spread*1000:.0f} mm); refusing to plan')
    base = pc.gazebo_pose(args.robot_model)
    world = tuple(base[i] + pc.rotate(pc.quat(*base[3:]), med)[i] for i in range(3))
    cup = world + (0.0, 0.0, 0.0)
    # Diagnostic only: how far the camera estimate is from simulator truth.
    truth = pc.gazebo_pose('cup')
    print(f'Vision cup [world] {[round(v,4) for v in world]} | Gazebo truth '
          f'{[round(v,4) for v in truth[:3]]} | error {math.dist(world, truth[:3])*1000:.1f} mm')
    return cup, base


def run_existing(script, params):
    path = Path(__file__).with_name(script)
    if not path.is_file():
        raise RuntimeError(f'Missing known-working dependency: {path}')
    print(f'Running: {script}',params,flush=True)
    result=subprocess.run([sys.executable,str(path)]+list(params), timeout=600)
    if result.returncode:
        raise RuntimeError(f'{script} returned {result.returncode}; no automatic release')


def update_world_scene(node, cup, base):
    client=node.create_client(GetPlanningScene,'/get_planning_scene')
    apply=node.create_client(ApplyPlanningScene,'/apply_planning_scene')
    scene=pc.get_scene(node,client)
    if any(a.object.id==pc.CUP_ID for a in scene.robot_state.attached_collision_objects):
        pc.detach_in_moveit(node,client,apply,base,cup)
        return
    co=CollisionObject()
    co.header.frame_id='base_link'
    co.id=pc.CUP_ID
    co.operation=CollisionObject.ADD
    co.pose.orientation.w=1.0
    from shape_msgs.msg import SolidPrimitive
    primitive=SolidPrimitive()
    primitive.type=SolidPrimitive.CYLINDER
    primitive.dimensions=[pc.CUP_HEIGHT, pc.CUP_RADIUS]
    co.primitives=[primitive]
    co.primitive_poses=[pc.cup_pose_in_base(cup,base)]
    diff=PlanningScene()
    diff.is_diff=True
    diff.robot_state.is_diff=True
    diff.world.collision_objects=[co]
    pc.apply_scene(node,apply,diff)
    verified=pc.get_scene(node,client)
    require_world_cup(verified)


def reset_cup(node,args):
    print('Explicit reset requested: detach then restore Gazebo cup pose')
    pc.gazebo_detach()
    time.sleep(.5)
    request=(f'name: "cup", position: {{x: {TABLE_START[0]}, y: {TABLE_START[1]}, '
             f'z: {TABLE_START[2]}}}, orientation: {{w: 1.0}}')
    for attempt in range(1, 4):       # the service occasionally times out under load
        result=subprocess.run(['gz','service','-s','/world/tabletop/set_pose',
             '--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean',
             '--timeout','5000','--req',request],capture_output=True,text=True,timeout=20)
        if not result.returncode and 'data: true' in result.stdout:
            break
        print(f'Gazebo set_pose attempt {attempt}/3 failed; retrying', flush=True)
        time.sleep(1.0)
    if result.returncode or 'data: true' not in result.stdout:
        raise RuntimeError('Gazebo set_pose failed: '+result.stdout+' '+result.stderr)
    time.sleep(1.0)
    cup=pc.gazebo_pose('cup')
    for _ in range(3):
        # The Gazebo detach is asynchronous: if it lands after the teleport, the cup
        # falls with the hand's motion. Teleport again until it stays put.
        if math.dist(cup[:3],TABLE_START)<=.035 and abs(cup[3])<.1 and abs(cup[4])<.1:
            break
        time.sleep(1.0)
        subprocess.run(['gz','service','-s','/world/tabletop/set_pose',
             '--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean',
             '--timeout','3000','--req',request],capture_output=True,text=True,timeout=15)
        time.sleep(1.0)
        cup=pc.gazebo_pose('cup')
    print('Reset Gazebo cup:',tuple(round(x,4) for x in cup[:3]))
    if math.dist(cup[:3],TABLE_START)>.035:
        raise RuntimeError('Gazebo cup did not remain at reset pose; inspect plugin/robot proximity')
    check_table_cup(cup)
    base=pc.gazebo_pose(args.robot_model)
    update_world_scene(node,cup,base)
    print('RESET COMPLETE: cup detached and restored in Gazebo + MoveIt world')


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--teach',action='store_true',help='Store open-hand grasp pose relative to cup once')
    modes.add_argument('--reset',action='store_true',help='Detach and restore cup, including MoveIt world pose')
    p.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    p.add_argument('--overwrite',action='store_true',help='Allow replacing grasp calibration')
    p.add_argument('--teach-lower-m',type=float,default=0.0,
        help='When teaching from POST-RELEASE pose, virtually lower palm in world Z by this many meters; does not move robot')
    p.add_argument('--robot-model',default='a200_0000/robot')
    return p.parse_args()


def main():
    args=arguments()
    if not (0.0 <= args.teach_lower_m <= 0.10):
        raise RuntimeError('--teach-lower-m must be 0..0.10 m')
    rclpy.init()
    node=Node('automate_cup')
    try:
        if args.reset:
            reset_cup(node,args)
        else:
            capture(node,args)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr)
        print('If something already moved, inspect Gazebo before retrying; '
              'this script never auto-detaches on failure.',file=sys.stderr)
        sys.exit(1)
