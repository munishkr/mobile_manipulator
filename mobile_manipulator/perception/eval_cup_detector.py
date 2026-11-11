#!/usr/bin/env python3
"""Measure cup_detector.py 3D accuracy against Gazebo truth at random table spots.

Needs cup_detector.py running and the arm in the look pose. Teleports the cup
(same table bounds as the dataset), takes the median of --samples fresh
/cup_detector/pose messages, and compares with the Gazebo pose in base_link.
The cup is put back at the start pose at the end.

    python3 eval_cup_detector.py --trials 20
"""

import argparse
import math
import random
import statistics
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

import collect_cup_dataset as cc
from collect_cup_dataset import pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rclpy.init()
    node = Node('cup_detector_eval')
    got = []
    node.create_subscription(PoseStamped, '/cup_detector/pose', got.append, 10)
    base = cc.gazebo_pose(cc.ROBOT_MODEL)
    errors, xy_errors, misses = [], [], 0
    try:
        for i in range(args.trials):
            spot = (0.95, 0.0) if i == 0 else cc.sample_table_xy(rng)
            cc.set_cup_pose(*spot, cc.CUP_Z, yaw=rng.uniform(-math.pi, math.pi))
            time.sleep(1.0)
            got.clear()
            end = time.time() + 8.0
            while time.time() < 1.0 + end - 8.0 or (len(got) < args.samples + 2 and time.time() < end):
                rclpy.spin_once(node, timeout_sec=0.05)
            fresh = got[2:]                       # drop frames that may predate the teleport
            truth = pc.cup_pose_in_base(cc.gazebo_pose('cup'), base).position
            if len(fresh) < args.samples:
                misses += 1
                print(f'{i:2d} spot ({spot[0]:.3f}, {spot[1]:.3f}): NO DETECTION ({len(fresh)})')
                continue
            med = [statistics.median(getattr(m.pose.position, k) for m in fresh) for k in 'xyz']
            err = math.dist(med, (truth.x, truth.y, truth.z))
            exy = math.dist(med[:2], (truth.x, truth.y))
            errors.append(err)
            xy_errors.append(exy)
            print(f'{i:2d} truth ({truth.x:.3f}, {truth.y:.3f}, {truth.z:.3f}) '
                  f'vision ({med[0]:.3f}, {med[1]:.3f}, {med[2]:.3f}) '
                  f'err {err*1000:5.1f} mm (xy {exy*1000:4.1f})', flush=True)
    finally:
        cc.set_cup_pose(*cc.CUP_START)
        node.destroy_node()
        rclpy.shutdown()
    if errors:
        print(f'\n{len(errors)}/{args.trials} detected, {misses} missed. 3D error mean '
              f'{statistics.mean(errors)*1000:.1f} mm, max {max(errors)*1000:.1f} mm; '
              f'xy mean {statistics.mean(xy_errors)*1000:.1f} mm, max {max(xy_errors)*1000:.1f} mm. '
              f'Cup reset to {cc.CUP_START}.')


if __name__ == '__main__':
    main()
