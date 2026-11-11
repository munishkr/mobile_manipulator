#!/usr/bin/env python3
"""Sync Gazebo's current cup pose into MoveIt as an attached collision object.

Changes the MoveIt planning scene ONLY. Does not publish a Gazebo attach/detach,
move joints, or change the Gazebo physics simulation.
"""

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from place_cup import gazebo_pose  # retries flaky `gz model --pose` reads

OBJECT_ID = 'target_cup'
LINK = 'left_hand_palm_link'
TOUCH_LINKS = [
    'left_hand_palm_link',
    *[f'left_hand_thumb_{i}_link' for i in range(3)],
    *[f'left_hand_middle_{i}_link' for i in range(2)],
    *[f'left_hand_index_{i}_link' for i in range(2)],
]


def quaternion(r, p, y):
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return (sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy,
            cr*cp*cy+sr*sp*sy)


def mul(a, b):
    x,y,z,w = a
    X,Y,Z,W = b
    return (w*X+x*W+y*Z-z*Y,
            w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W,
            w*W-x*X-y*Y-z*Z)


def inv(q):
    return (-q[0], -q[1], -q[2], q[3])


def rotate(q, v):
    t = mul(mul(q, (v[0],v[1],v[2],0.0)), inv(q))
    return t[:3]


def base_frame_pose(cup, base):
    # both are Gazebo-world XYZ RPY; output cup pose in MoveIt's base_link frame
    qc = quaternion(*cup[3:])
    qb_inv = inv(quaternion(*base[3:]))
    delta = tuple(cup[i]-base[i] for i in range(3))
    xyz = rotate(qb_inv, delta)
    q = mul(qb_inv, qc)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def call(node, client, request, label):
    if not client.wait_for_service(timeout_sec=10):
        raise RuntimeError(f'{label} service not available')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=12)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{label} did not return successfully')
    return future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-pose', type=float, nargs=6, metavar=('X','Y','Z','R','P','YAW'),
                        help='Exact world pose of base_link, overriding Gazebo robot lookup')
    parser.add_argument('--robot-model', default='a200_0000/robot',
                        help='Gazebo robot model to query (default: a200_0000/robot)')
    parser.add_argument('--fallback-base-pose', type=float, nargs=6,
                        default=[0,0,0.132279,0,0,0],
                        metavar=('X','Y','Z','R','P','YAW'),
                        help='Fallback base_link pose when robot model lookup fails; prior observed pose')
    parser.add_argument('--table-clearance', type=float, default=0.005,
                        help='Raise the attached MoveIt cylinder by this much (m). The grasp '
                             'tilts the cup a few degrees, so its exact Gazebo pose touches '
                             'work_table and MoveIt rejects every lift (CheckStartStateCollision).')
    args = parser.parse_args()

    cup = gazebo_pose('cup')
    if args.base_pose:
        base = args.base_pose
        source = 'explicit --base-pose'
    else:
        try:
            base = gazebo_pose(args.robot_model)
            source = f'Gazebo model {args.robot_model}'
        except RuntimeError as exc:
            base = args.fallback_base_pose
            source = 'FALLBACK — previously observed base_link pose; verify it is still correct'
            print(f'WARNING: robot pose lookup failed: {exc}', file=sys.stderr)
    pose = base_frame_pose(cup, base)
    pose.position.z += args.table_clearance   # MoveIt model only; Gazebo is untouched
    print('Gazebo cup XYZ RPY:', [round(v,6) for v in cup])
    print(f'Base pose from {source}:', [round(v,6) for v in base])
    print('Cup position in base_link:', [round(pose.position.x,5), round(pose.position.y,5), round(pose.position.z,5)])

    rclpy.init()
    node = Node('sync_gazebo_cup_to_moveit')
    try:
        get_scene = node.create_client(GetPlanningScene, '/get_planning_scene')
        apply_scene = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
                                     PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        before = call(node, get_scene, req, 'GetPlanningScene').scene
        world = next((o for o in before.world.collision_objects if o.id == OBJECT_ID), None)
        attached_before = next((o for o in before.robot_state.attached_collision_objects
                                if o.object.id == OBJECT_ID), None)
        if world is None and attached_before is None:
            raise RuntimeError('target_cup missing from MoveIt. Run add_scene_objects.py first.')
        if attached_before is not None:
            print('MoveIt already has target_cup attached; replacing its pose to match Gazebo.')

        co = CollisionObject()
        co.header.frame_id = 'base_link'
        co.id = OBJECT_ID
        co.operation = CollisionObject.ADD
        co.pose.orientation.w = 1.0
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [0.12, 0.035]  # height, radius used in tabletop world
        co.primitives = [primitive]
        co.primitive_poses = [pose]
        aco = AttachedCollisionObject()
        aco.link_name = LINK
        aco.object = co
        aco.touch_links = TOUCH_LINKS
        diff = PlanningScene()
        diff.is_diff = True
        diff.robot_state.is_diff = True
        diff.robot_state.attached_collision_objects = [aco]
        apply_req = ApplyPlanningScene.Request()
        apply_req.scene = diff
        if not call(node, apply_scene, apply_req, 'ApplyPlanningScene').success:
            raise RuntimeError('MoveIt rejected the attachment; inspect latest move_group log')

        after = call(node, get_scene, req, 'GetPlanningScene').scene
        in_world = any(o.id == OBJECT_ID for o in after.world.collision_objects)
        attached = any(o.object.id == OBJECT_ID and o.link_name == LINK
                       for o in after.robot_state.attached_collision_objects)
        print(f'MoveIt verification: attached_to_palm={attached}, in_world={in_world}')
        if not attached or in_world:
            raise RuntimeError('Attachment verification failed')
        print('SUCCESS: MoveIt cup is attached using Gazebo cup pose. Gazebo physics unchanged.')
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
