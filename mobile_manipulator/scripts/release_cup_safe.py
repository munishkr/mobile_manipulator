#!/usr/bin/env python3
"""Safely release a Gazebo-held cup at a verified placement pose.

Requires place_cup.py in the same directory. Does not command the arm.
Default is read-only preview; --execute opens Dex3 while Gazebo still holds
cup, detaches in Gazebo, then synchronizes the MoveIt world cup from Gazebo.
Designed for the specific tabletop assessment world, not a real robot.
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

import place_cup as pc


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--place-x', type=float, default=0.90)
    p.add_argument('--place-y', type=float, default=0.16)
    p.add_argument('--release-z', type=float, default=0.88)
    p.add_argument('--robot-model', default='a200_0000/robot')
    p.add_argument('--execute', action='store_true', help='OPEN fingers, DETACH in Gazebo, update MoveIt scene')
    return p


def main():
    args = parser().parse_args()
    cup = pc.gazebo_pose('cup')
    distance = math.dist(cup[:3], (args.place_x, args.place_y, args.release_z))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(cup[3]) * math.cos(cup[4])))))
    print(f'Gazebo cup xyz: {[round(v, 4) for v in cup[:3]]}')
    print(f'Placement error: {distance*1000:.1f} mm; tilt: {tilt:.1f} deg')

    if distance > 0.04:
        raise RuntimeError('Cup is more than 40 mm from placement pose; refusing release')
    if tilt > 20:
        raise RuntimeError('Cup is tilted more than 20 degrees; refusing release')
    if not (0.8 <= args.place_x <= 1.5 and -0.25 <= args.place_y <= 0.25):
        raise RuntimeError('Placement coordinates outside known table footprint')

    rclpy.init()
    node = Node('release_cup_safe')
    try:
        scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
        apply_client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        scene = pc.get_scene(node, scene_client)
        attached = [a for a in scene.robot_state.attached_collision_objects
                    if a.object.id == pc.CUP_ID and a.link_name == pc.PALM]
        if len(attached) != 1:
            raise RuntimeError('MoveIt target_cup is not attached to palm; refusing release')
        if any(obj.id == pc.CUP_ID for obj in scene.world.collision_objects):
            raise RuntimeError('MoveIt has duplicate stationary cup; refusing release')

        if not args.execute:
            print('PREVIEW OK: Cup position and MoveIt attachment verified.')
            print('No actions taken. Add --execute to open fingers and release cup.')
            return 0

        # Important: leave Gazebo joint attached as the fingers open so cup
        # does not become trapped between the fingers as it falls.
        print('Opening Dex3 while Gazebo supports the cup ...')
        gripper = ActionClient(node, FollowJointTrajectory,
                               '/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory')
        pc.open_dex3(node, gripper)

        print('Detaching Gazebo joint ...')
        pc.gazebo_detach()
        time.sleep(2.0)
        released = pc.gazebo_pose('cup')
        print(f'Gazebo cup after detach: {[round(v, 4) for v in released[:3]]}')

        # The cup bottom should settle near tabletop z=0.75 (center~0.81).
        # Avoid silently claiming success if it remains suspended.
        if released[2] > 0.86:
            raise RuntimeError('Cup is still unexpectedly high; inspect Gazebo before MoveIt scene update')
        if released[2] < pc.TABLE_TOP - 0.02:
            raise RuntimeError('Cup appears below tabletop; inspect Gazebo before MoveIt scene update')
        if math.hypot(released[0] - args.place_x, released[1] - args.place_y) > 0.08:
            raise RuntimeError('Cup left the placement area; inspect Gazebo before updating MoveIt')

        # Synchronize MoveIt to ACTUAL pose after release and finger opening.
        base = pc.gazebo_pose(args.robot_model)
        pc.detach_in_moveit(node, scene_client, apply_client, base, released)
        print('RELEASE COMPLETE: Gazebo cup detached and MoveIt world object updated.')
        print('Check that the cup is stable on the table in Gazebo.')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
