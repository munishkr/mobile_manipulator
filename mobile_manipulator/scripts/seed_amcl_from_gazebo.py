#!/usr/bin/env python3
"""Gazebo only: set the AMCL pose to the robot's true Gazebo pose, then verify it.

Global localization (random particles over the whole map) leaves AMCL tens of
centimetres off until the robot has driven around. In simulation the true pose
is known, so seed AMCL with it. The map frame is assumed to equal the Gazebo
world frame, since maps/tabletop_nav.yaml was built from that world.

  python3 seed_amcl_from_gazebo.py            # seed + verify (exit 1 if still off)
  python3 seed_amcl_from_gazebo.py --check    # only report the AMCL error

runRobot.sh runs this at startup. Run it again whenever AMCL drifts.
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

import place_cup as pc

NS = '/a200_0000'


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='Report only; do not seed')
    ap.add_argument('--xy-tol', type=float, default=0.05)
    ap.add_argument('--yaw-tol-deg', type=float, default=3.0)
    ap.add_argument('--attempts', type=int, default=3)
    a = ap.parse_args()

    rclpy.init(args=['--ros-args', '-r', f'/tf:={NS}/tf', '-r', f'/tf_static:={NS}/tf_static'])
    node = Node('seed_amcl_from_gazebo',
                parameter_overrides=[Parameter('use_sim_time', value=True)])
    tf_buf = Buffer()
    TransformListener(tf_buf, node)
    pub = node.create_publisher(PoseWithCovarianceStamped, f'{NS}/initialpose', 10)
    nomotion = node.create_client(Empty, f'{NS}/request_nomotion_update')

    def spin(sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def amcl_error(wait=10.0):
        g = pc.gazebo_pose('a200_0000/robot')
        end = time.monotonic() + wait
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                tr = tf_buf.lookup_transform('map', 'base_link', Time()).transform
                break
            except Exception as exc:  # noqa: BLE001
                if time.monotonic() > end:
                    raise RuntimeError(f'No map->base_link TF: {exc}') from exc
        d = math.hypot(tr.translation.x - g[0], tr.translation.y - g[1])
        dyaw = math.degrees(math.atan2(math.sin(yaw_of(tr.rotation) - g[5]),
                                       math.cos(yaw_of(tr.rotation) - g[5])))
        return d, dyaw, g

    try:
        try:
            d, dyaw, _ = amcl_error(wait=5.0)
            print(f'AMCL vs Gazebo before: {d:.3f} m, {dyaw:+.1f} deg')
        except RuntimeError:
            # Fresh AMCL publishes no map frame until it gets an initial pose.
            if a.check:
                print('AMCL has no pose yet (no map frame)')
                return 1
            print('AMCL has no pose yet; seeding it')
        if a.check:
            return 0 if d <= a.xy_tol and abs(dyaw) <= a.yaw_tol_deg else 1

        end = time.monotonic() + 15
        while pub.get_subscription_count() == 0 and time.monotonic() < end:
            spin(0.2)
        if pub.get_subscription_count() == 0:
            raise RuntimeError(f'AMCL is not subscribed to {NS}/initialpose')

        for attempt in range(1, a.attempts + 1):
            g = pc.gazebo_pose('a200_0000/robot')
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.pose.pose.position.x, msg.pose.pose.position.y = g[0], g[1]
            msg.pose.pose.orientation.z = math.sin(g[5] / 2)
            msg.pose.pose.orientation.w = math.cos(g[5] / 2)
            msg.pose.covariance[0] = msg.pose.covariance[7] = 0.02 ** 2
            msg.pose.covariance[35] = math.radians(2.0) ** 2
            pub.publish(msg)
            spin(1.0)
            # A few in-place scan updates so AMCL refines and republishes map->odom.
            if nomotion.wait_for_service(timeout_sec=3.0):
                for _ in range(3):
                    fut = nomotion.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
                    spin(0.5)
            spin(1.0)
            d, dyaw, _ = amcl_error()
            print(f'Attempt {attempt}: AMCL vs Gazebo {d:.3f} m, {dyaw:+.1f} deg')
            if d <= a.xy_tol and abs(dyaw) <= a.yaw_tol_deg:
                print('AMCL seeded from Gazebo ground truth.')
                return 0
        print('AMCL still disagrees with Gazebo; check that the map frame matches the world.',
              file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
