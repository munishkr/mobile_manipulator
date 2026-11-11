#!/opt/yolo_venv/bin/python3
"""YOLO cup detector on the wrist RGB-D camera -> cup pose in base_link.

Subscribes  /cup_wrist_rgbd/{image,depth_image,camera_info}  (TF from /a200_0000/tf)
Publishes   /cup_detector/pose        geometry_msgs/PoseStamped, cup CENTRE in base_link
            /cup_detector/detections  vision_msgs/Detection2DArray (pixel boxes)
            /cup_detector/image       sensor_msgs/Image, annotated (add it in RViz)

3D estimate: depth pixels inside the YOLO box are back-projected; the cup's front
surface is the nearest depth cluster (table/background behind it are dropped). The
surface centroid is pushed one cup radius further along the horizontal viewing
direction to reach the axis, and z is the top rim minus half the cup height.

    /opt/yolo_venv/bin/python3 cup_detector.py                    # runs continuously
    /opt/yolo_venv/bin/python3 cup_detector.py --compare-gazebo   # also logs error vs truth

Simulation only (cup size comes from place_cup.py).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import place_cup as pc  # noqa: E402

CAM = '/cup_wrist_rgbd'
BASE_FRAME = 'base_link'
DEFAULT_MODEL = HERE / 'models' / 'cup_yolov8n.pt'


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class CupDetector(Node):
    def __init__(self, args):
        super().__init__('cup_detector',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.args = args
        self.bridge = CvBridge()
        self.model = YOLO(str(args.model))
        self.device = args.device
        self.model.predict(np.zeros((480, 640, 3), np.uint8), device=self.device,
                           imgsz=args.imgsz, verbose=False)  # warm-up (CUDA init) before the first frame
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pose_pub = self.create_publisher(PoseStamped, '/cup_detector/pose', 10)
        self.det_pub = self.create_publisher(Detection2DArray, '/cup_detector/detections', 10)
        self.img_pub = self.create_publisher(Image, '/cup_detector/image', 2)
        subs = [Subscriber(self, Image, f'{CAM}/image'),
                Subscriber(self, Image, f'{CAM}/depth_image'),
                Subscriber(self, CameraInfo, f'{CAM}/camera_info')]
        # The camera bridge drops frames under load (long runs: ~40% of RGB, ~70% of
        # depth lost), so exact-stamp pairs got rare (0.15 Hz). The arm is still while
        # the detector is used, so pairing with the neighbouring frame is harmless.
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=args.sync_slop)
        self.sync.registerCallback(self.on_frames)
        self.last_log = 0.0
        self.last_run = 0.0
        self.get_logger().info(f'Loaded {args.model} on device {self.device}; '
                               f'conf >= {args.conf}')

    # ------------------------------------------------------------------ 3D
    def cup_centre(self, box, depth, K):
        """Cup axis point in the camera optical frame, or (None, reason)."""
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        # Inner part of the box: avoids the silhouette edge where depth mixes with
        # the table behind the cup.
        u0, u1 = int(x0 + 0.2 * w), int(math.ceil(x1 - 0.2 * w))
        v0, v1 = int(y0 + 0.1 * h), int(math.ceil(y1 - 0.1 * h))
        patch = depth[v0:v1, u0:u1]
        vv, uu = np.mgrid[v0:v1, u0:u1]
        ok = np.isfinite(patch) & (patch > 0.05)
        if ok.sum() < 20:
            return None, 'too few depth pixels'
        z = patch[ok]
        near = np.percentile(z, 10)
        keep = z < near + 2.2 * pc.CUP_RADIUS          # front surface of the cylinder
        z, u, v = z[keep], uu[ok][keep], vv[ok][keep]
        X = (u - K[0, 2]) * z / K[0, 0]
        Y = (v - K[1, 2]) * z / K[1, 1]
        return np.c_[X, Y, z], None

    def on_frames(self, rgb_msg, depth_msg, info_msg):
        # Rate cap: CPU YOLO on every 10 Hz frame used ~4 cores and slowed Gazebo to
        # ~0.2x real time, which made arm tracking lag. 3 Hz is plenty for picking.
        now = time.monotonic()
        if now - self.last_run < 1.0 / self.args.max_rate:
            return
        self.last_run = now
        K = np.array(info_msg.k).reshape(3, 3)
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
        res = self.model.predict(rgb, conf=self.args.conf, device=self.device,
                                 imgsz=self.args.imgsz, verbose=False)[0]
        dets = Detection2DArray()
        dets.header = rgb_msg.header
        annotated = rgb.copy()
        best = None
        for b in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            conf = float(b.conf[0])
            d = Detection2D()
            d.header = rgb_msg.header
            d.bbox.center.position.x, d.bbox.center.position.y = (x0 + x1) / 2, (y0 + y1) / 2
            d.bbox.size_x, d.bbox.size_y = x1 - x0, y1 - y0
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = 'cup', conf
            d.results.append(hyp)
            dets.detections.append(d)
            cv2.rectangle(annotated, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(annotated, f'cup {conf:.2f}', (int(x0), max(12, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if best is None or conf > best[1]:
                best = ((x0, y0, x1, y1), conf)
        self.det_pub.publish(dets)

        if best is not None:
            pose, why = self.estimate_pose(best[0], depth, K, rgb_msg)
            if pose is not None:
                self.pose_pub.publish(pose)
                p = pose.pose.position
                cv2.putText(annotated, f'base_link ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})',
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                self.maybe_log(pose, best[1])
            else:
                self.get_logger().warn(f'Cup seen but no 3D pose: {why}',
                                       throttle_duration_sec=2.0)
        out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out.header = rgb_msg.header
        self.img_pub.publish(out)

    def estimate_pose(self, box, depth, K, rgb_msg):
        pts, why = self.cup_centre(box, depth, K)
        if pts is None:
            return None, why
        frame = rgb_msg.header.frame_id
        stamp = Time.from_msg(rgb_msg.header.stamp)
        if not self.tf.can_transform(BASE_FRAME, frame, stamp):
            stamp = Time()  # arm static during look; fall back to latest
            if not self.tf.can_transform(BASE_FRAME, frame, stamp):
                return None, f'no TF {BASE_FRAME} <- {frame}'
        tr = self.tf.lookup_transform(BASE_FRAME, frame, stamp)
        R = quat_to_R(tr.transform.rotation)
        t = np.array([tr.transform.translation.x, tr.transform.translation.y,
                      tr.transform.translation.z])
        P = pts @ R.T + t                                 # surface points in base_link
        top = np.percentile(P[:, 2], 97)
        centre_z = top - pc.CUP_HEIGHT / 2
        # x/y from the side wall only: a downward-looking camera also sees the top
        # face, whose points already sit over the axis and must not be pushed out.
        side = P[P[:, 2] < top - 0.015]
        if len(side) < 10:
            side = P
        surf = np.median(side, axis=0)
        ray = surf[:2] - t[:2]                           # horizontal viewing direction
        ray /= max(np.linalg.norm(ray), 1e-6)
        centre_xy = surf[:2] + pc.CUP_RADIUS * ray          # visible face -> axis
        msg = PoseStamped()
        msg.header.stamp = rgb_msg.header.stamp
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x, msg.pose.position.y = float(centre_xy[0]), float(centre_xy[1])
        msg.pose.position.z = float(centre_z)
        msg.pose.orientation.w = 1.0                     # cylinder: yaw is meaningless
        return msg, None

    def maybe_log(self, pose, conf):
        now = time.time()
        if now - self.last_log < self.args.log_period:
            return
        self.last_log = now
        p = pose.pose.position
        text = f'cup conf {conf:.2f} at base_link ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
        if self.args.compare_gazebo:
            try:
                truth = pc.cup_pose_in_base(pc.gazebo_pose('cup'),
                                            pc.gazebo_pose(self.args.robot_model)).position
                err = math.dist((p.x, p.y, p.z), (truth.x, truth.y, truth.z))
                text += (f' | gazebo ({truth.x:.3f}, {truth.y:.3f}, {truth.z:.3f})'
                         f' error {1000 * err:.1f} mm')
            except Exception as e:  # noqa: BLE001
                text += f' | gazebo compare failed: {e}'
        self.get_logger().info(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--device', default='cpu',
                    help="'cpu' (~70 ms/frame, enough for the 10 Hz camera) or a CUDA index, e.g. 0")
    ap.add_argument('--imgsz', type=int, default=416, help='must match training')
    ap.add_argument('--compare-gazebo', action='store_true')
    ap.add_argument('--robot-model', default='a200_0000/robot')
    ap.add_argument('--log-period', type=float, default=1.0)
    ap.add_argument('--sync-slop', type=float, default=0.12,
                    help='max RGB/depth/camera_info stamp difference (s) to pair a frame')
    ap.add_argument('--max-rate', type=float, default=3.0, help='max frames processed per second')
    ap.add_argument('--threads', type=int, default=2, help='CPU torch threads')
    args, ros_args = ap.parse_known_args()
    import torch
    torch.set_num_threads(args.threads)
    if not args.model.exists():
        sys.exit(f'No weights at {args.model}; run train_cup_yolo.py first')
    rclpy.init(args=['--ros-args', '-r', '/tf:=/a200_0000/tf',
                     '-r', '/tf_static:=/a200_0000/tf_static'] + ros_args)
    node = CupDetector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
