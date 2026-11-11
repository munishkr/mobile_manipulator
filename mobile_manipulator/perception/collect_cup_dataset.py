#!/usr/bin/env python3
"""Auto-labelled YOLO dataset of the tabletop cup, from the wrist RGB-D camera.

Gazebo knows exactly where the cup is, so nobody has to draw boxes: for every
sample the cup is teleported to a random spot on the table, the cylinder
(radius/height from place_cup.py) is projected into the camera with the TF pose
and camera_info intrinsics, and the box is written in YOLO format. The depth
image is used to reject samples where something (the Dex3 fingers) hides the cup.

The arm is NOT moved. Put it in the look pose first (the cup must be visible in
RViz "Wrist RGB"), e.g. with `look_pose.py --execute`.

    python3 collect_cup_dataset.py --count 600            # -> perception/datasets/cup
    python3 collect_cup_dataset.py --count 20 --preview   # also writes boxed previews

At the end the cup is put back at the world start pose (0.95, 0, 0.811).
Simulation only.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

# gz CLI calls must use the same transport interface as the sim (runRobot.sh exports
# GZ_IP=127.0.0.1; Wi-Fi multicast broke gz-transport in long sessions).
os.environ.setdefault('GZ_IP', '127.0.0.1')
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import place_cup as pc  # noqa: E402  (gazebo_pose, quaternion helpers, cup size)

CAM = '/cup_wrist_rgbd'
CAMERA_FRAME = 'cup_wrist_camera_optical_frame'
BASE_FRAME = 'base_link'
ROBOT_MODEL = 'a200_0000/robot'
CUP_START = (0.95, 0.0, 0.811)
# Table (tabletop.sdf): centre x=1.15, y=0, top 0.8 x 0.6 at z=0.75.
TABLE_X = (0.75, 1.55)
TABLE_Y = (-0.30, 0.30)
EDGE_MARGIN = 0.06
CUP_Z = pc.TABLE_TOP + pc.CUP_HEIGHT / 2 + 0.001
OFF_VIEW = (0.0, -1.6, pc.CUP_HEIGHT / 2 + 0.001)   # on the floor, out of view


def gazebo_pose(model, attempts=4):
    """pc.gazebo_pose with retries: `gz model` sometimes returns before the state."""
    for k in range(attempts):
        try:
            return pc.gazebo_pose(model)
        except RuntimeError:
            if k == attempts - 1:
                raise
            time.sleep(0.5)


def set_cup_pose(x, y, z, yaw=0.0):
    q = pc.quat(0.0, 0.0, yaw)
    req = (f'name: "cup", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}')
    for attempt in range(5):  # gz service calls time out now and then under load
        r = subprocess.run(['gz', 'service', '-s', '/world/tabletop/set_pose',
                            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                            '--timeout', '5000', '--req', req],
                           capture_output=True, text=True, timeout=15)
        if 'data: true' in r.stdout:
            return
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f'set_pose failed 5 times: {r.stdout} {r.stderr}')


def cylinder_points(cup_xyz, cup_rpy, n=32):
    """Points on the top and bottom rims of the cup, in world."""
    q = pc.quat(*cup_rpy)
    h, r = pc.CUP_HEIGHT / 2, pc.CUP_RADIUS
    pts = []
    for z in (-h, h):
        for k in range(n):
            a = 2 * math.pi * k / n
            local = (r * math.cos(a), r * math.sin(a), z)
            w = pc.rotate(q, local)
            pts.append([cup_xyz[i] + w[i] for i in range(3)])
    return np.array(pts)


def tf_matrix(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, (t.x, t.y, t.z)
    return M


def world_to_base_matrix(base):
    """4x4 transform taking world points into base_link (from Gazebo model pose)."""
    q = pc.quat(*base[3:])
    x, y, z, w = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    M = np.eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ np.array(base[:3])
    return M


class Collector(Node):
    def __init__(self):
        super().__init__('cup_dataset_collector',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.bridge = CvBridge()
        self.rgb = self.depth = self.info = None
        self.create_subscription(Image, f'{CAM}/image', self._rgb, 2)
        self.create_subscription(Image, f'{CAM}/depth_image', self._depth, 2)
        self.create_subscription(CameraInfo, f'{CAM}/camera_info', self._info, 2)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

    def _rgb(self, m): self.rgb = m
    def _depth(self, m): self.depth = m
    def _info(self, m): self.info = m

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def fresh_frames(self, after: Time, timeout=10.0):
        """RGB + depth both stamped after `after` (sim time)."""
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rgb is None or self.depth is None or self.info is None:
                continue
            if (Time.from_msg(self.rgb.header.stamp) > after and
                    Time.from_msg(self.depth.header.stamp) > after):
                return self.rgb, self.depth
        raise RuntimeError('No fresh camera frames; is runRobot.sh up and the bridge running?')

    def camera_from_base(self, timeout=15.0):
        # Spin while waiting: lookup_transform's own timeout does not spin this
        # single-threaded node, so a late /tf_static would never arrive.
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf.can_transform(CAMERA_FRAME, BASE_FRAME, Time()):
                return tf_matrix(self.tf.lookup_transform(CAMERA_FRAME, BASE_FRAME, Time()))
        raise RuntimeError(f'No TF {BASE_FRAME} -> {CAMERA_FRAME} after {timeout:.0f} s '
                           '(robot_state_publisher on /a200_0000/tf?)')


def project(points_cam, K):
    z = points_cam[:, 2]
    if np.any(z <= 0.05):
        return None
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return u, v


def label_for(cup_world, base, cam_from_base, K, W, H, depth, min_visible=0.85):
    """Return (yolo_box, info) or (None, reason)."""
    pts = cylinder_points(cup_world[:3], cup_world[3:])
    hom = np.c_[pts, np.ones(len(pts))]
    cam = (cam_from_base @ world_to_base_matrix(base) @ hom.T).T[:, :3]
    uv = project(cam, K)
    if uv is None:
        return None, 'behind camera'
    u, v = uv
    x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max()
    full = (x1 - x0) * (y1 - y0)
    cx0, cx1 = max(0.0, x0), min(W - 1.0, x1)
    cy0, cy1 = max(0.0, y0), min(H - 1.0, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, 'outside image'
    if (cx1 - cx0) * (cy1 - cy0) < min_visible * full:
        return None, 'cut by image border'
    if (cx1 - cx0) < 6 or (cy1 - cy0) < 6:
        return None, 'too small'
    # Occlusion check: depth at the box centre should be about the distance to the
    # cup's near surface. A much smaller depth means fingers/hand in front of it.
    centre_cam = (cam_from_base @ world_to_base_matrix(base) @
                  np.r_[cup_world[:3], 1.0])[:3]
    u0, v0 = int((cx0 + cx1) / 2), int((cy0 + cy1) / 2)
    pw, ph = max(2, int((cx1 - cx0) * 0.15)), max(2, int((cy1 - cy0) * 0.15))
    patch = depth[max(0, v0-ph):v0+ph+1, max(0, u0-pw):u0+pw+1]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return None, 'no depth at cup'
    measured = float(np.median(patch))
    expected = float(centre_cam[2]) - pc.CUP_RADIUS
    if measured < expected - 0.04:
        return None, f'occluded (depth {measured:.3f} < {expected:.3f})'
    box = ((cx0 + cx1) / 2 / W, (cy0 + cy1) / 2 / H, (cx1 - cx0) / W, (cy1 - cy0) / H)
    return box, {'pixel_box': [cx0, cy0, cx1, cy1], 'depth_check': [measured, expected]}


def sample_table_xy(rng):
    return (rng.uniform(TABLE_X[0] + EDGE_MARGIN, TABLE_X[1] - EDGE_MARGIN),
            rng.uniform(TABLE_Y[0] + EDGE_MARGIN, TABLE_Y[1] - EDGE_MARGIN))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--count', type=int, default=600, help='labelled samples to keep')
    ap.add_argument('--out', type=Path, default=HERE / 'datasets' / 'cup')
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--negatives', type=float, default=0.05,
                    help='fraction of extra frames with no cup in view')
    ap.add_argument('--settle', type=float, default=0.6, help='wall seconds after teleport')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--preview', action='store_true', help='also save images with boxes drawn')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rclpy.init(args=['--ros-args', '-r', '/tf:=/a200_0000/tf',
                     '-r', '/tf_static:=/a200_0000/tf_static'])
    node = Collector()
    for split in ('train', 'val'):
        (args.out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (args.out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (args.out / 'preview').mkdir(parents=True, exist_ok=True)

    node.spin_for(2.0)
    base = gazebo_pose(ROBOT_MODEL)
    cam_from_base = node.camera_from_base()
    start = len(list((args.out / 'images').rglob('*.png')))
    meta_path = args.out / 'meta.jsonl'
    kept = rejected = negatives = 0
    reasons = {}
    n_neg = int(round(args.count * args.negatives))
    plan = ['cup'] * args.count + ['neg'] * n_neg
    rng.shuffle(plan)
    try:
        with meta_path.open('a') as meta:
            i = 0
            attempts = 0
            while i < len(plan) and attempts < 6 * len(plan):
                attempts += 1
                kind = plan[i]
                if kind == 'neg':
                    target = OFF_VIEW
                else:
                    target = (*sample_table_xy(rng), CUP_Z)
                yaw = rng.uniform(-math.pi, math.pi)
                set_cup_pose(*target, yaw=yaw)
                node.spin_for(args.settle)
                after = node.get_clock().now()
                rgb_msg, depth_msg = node.fresh_frames(after)
                cup = gazebo_pose('cup')
                K = np.array(node.info.k).reshape(3, 3)
                W, H = node.info.width, node.info.height
                rgb = node.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
                depth = node.bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)

                if kind == 'cup':
                    tilt = math.degrees(math.acos(max(-1, min(1, math.cos(cup[3]) * math.cos(cup[4])))))
                    if tilt > 10:
                        rejected += 1
                        reasons['tipped over'] = reasons.get('tipped over', 0) + 1
                        continue
                    box, info = label_for(cup, base, cam_from_base, K, W, H, depth)
                    if box is None:
                        rejected += 1
                        reasons[info.split(' (')[0]] = reasons.get(info.split(' (')[0], 0) + 1
                        continue
                    line = '0 ' + ' '.join(f'{v:.6f}' for v in box) + '\n'
                else:
                    box, info, line = None, {}, ''
                    negatives += 1

                idx = start + i
                split = 'val' if rng.random() < args.val_fraction else 'train'
                name = f'cup_{idx:05d}'
                cv2.imwrite(str(args.out / 'images' / split / f'{name}.png'), rgb)
                (args.out / 'labels' / split / f'{name}.txt').write_text(line)
                cup_base = pc.cup_pose_in_base(cup, base).position
                meta.write(json.dumps({
                    'image': f'images/{split}/{name}.png', 'has_cup': box is not None,
                    'cup_world': cup[:3], 'cup_base': [cup_base.x, cup_base.y, cup_base.z],
                    'stamp': [rgb_msg.header.stamp.sec, rgb_msg.header.stamp.nanosec],
                    **({k: v for k, v in info.items()} if isinstance(info, dict) else {}),
                }) + '\n')
                if args.preview and box is not None:
                    x0, y0, x1, y1 = map(int, info['pixel_box'])
                    prev = rgb.copy()
                    cv2.rectangle(prev, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.imwrite(str(args.out / 'preview' / f'{name}.png'), prev)
                kept += box is not None
                i += 1
                if i % 25 == 0:
                    print(f'{i}/{len(plan)} saved ({kept} cups, {negatives} empty, '
                          f'{rejected} rejected: {reasons})', flush=True)
    finally:
        set_cup_pose(*CUP_START)
        node.destroy_node()
        rclpy.shutdown()

    (args.out / 'cup.yaml').write_text(
        f'path: {args.out}\ntrain: images/train\nval: images/val\nnames:\n  0: cup\n')
    print(f'Done: {kept} labelled cups, {negatives} empty frames, {rejected} rejected '
          f'{reasons}. Dataset: {args.out / "cup.yaml"}. Cup reset to {CUP_START}.')


if __name__ == '__main__':
    main()
