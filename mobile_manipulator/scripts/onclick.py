import cv2
import rclpy
from cv_bridge import CvBridge
import numpy as np
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.optimize import least_squares
from visualization_msgs.msg import Marker



class ImageClickListener(Node):
    def __init__(self):
        super().__init__('image_click_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        
        # 1. Create the OpenCV window and attach the callback
        cv2.namedWindow("RGB")
        cv2.setMouseCallback("RGB", self.on_mouse)
        
        # 2. Subscribe to your camera topic
        self.subscription = self.create_subscription(Image,'/cup_wrist_rgbd/image', self.image_callback,10)

        # 3. Subscribe to the aligned Depth image topic
        self.depth_sub = self.create_subscription(Image,'/cup_wrist_rgbd/depth_image', self.depth_callback, 10)

    def depth_callback(self, msg):
        # Convert depth image using 'passthrough' to preserve exact depth units (mm or meters)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_header = msg.header 

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info(f'Image clicked at pixel: ({x}, {y})')
            depth_value = self.latest_depth[y, x]
            self.get_logger().info(f'Clicked ({x}, {y}) -> Depth: {depth_value:.3f} m')

            point_camera = PointStamped()
            point_camera.header.frame_id = "cup_wrist_camera_optical_frame"
            point_camera.header.stamp = self.depth_header.stamp

            fx, fy = 467.743, 467.743
            cx, cy = 320.0, 240.0
            point_camera.point.x = (x - cx) * depth_value / fx
            point_camera.point.y = (y - cy) * depth_value / fy
            point_camera.point.z = depth_value
            self.get_logger().info(f'Clicked in camera frame ({point_camera.point.x}, {point_camera.point.y}, {point_camera.point.z})') 

            point_base = PointStamped()
            point_base.header.frame_id = "base_link"
            point_base.header.stamp = self.depth_header.stamp
            point_base = self.tf_buffer.transform(point_camera,"base_link")

            self.get_logger().info(f'Clicked in base frame ({point_base.point.x}, {point_base.point.y}, {point_base.point.z})') 


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("RGB", cv_image)
        cv2.waitKey(1)

class ROIPointCloud(Node):
    def __init__(self):
        super().__init__("roi_pointcloud")

        self.tf_wait_count = 0
        self.pending_roi = None
        self.create_timer(0.05, self.try_process_roi)
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_msg = None
        self.depth = None
        self.camera_info = None

        self.drag_start = None
        self.drag_end = None

        self.cloud_pub = self.create_publisher(PointCloud2, "/cup_roi_cloud", 10)
        self.marker_pub = self.create_publisher(Marker, "/cylinder_center_marker", 10)

        self.create_subscription(CameraInfo,"/cup_wrist_rgbd/camera_info",self.info_callback,qos_profile_sensor_data)

        rgb_sub = Subscriber(self, Image, "/cup_wrist_rgbd/image", qos_profile=qos_profile_sensor_data)

        depth_sub = Subscriber(self, Image, "/cup_wrist_rgbd/depth_image", qos_profile=qos_profile_sensor_data)


        # Pair RGB and depth frames by timestamp.
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.image_callback)

        cv2.namedWindow("RGB")
        cv2.setMouseCallback("RGB", self.on_mouse)

    def try_process_roi(self):
        if self.pending_roi is None:
            return

        x0, y0, x1, y1, depth, stamp, frame_id, encoding = \
            self.pending_roi

        image_time = Time.from_msg(stamp)

        available, reason = self.tf_buffer.can_transform(
            "base_link",
            frame_id,
            image_time,
            return_debug_tuple=True
        )

        if not available:
            self.tf_wait_count += 1

            # Print approximately once per second
            if self.tf_wait_count % 20 == 0:
                self.get_logger().warn(
                    f"Waiting for TF at {stamp.sec}.{stamp.nanosec:09d}: "
                    f"{reason}"
                )
            return

        self.get_logger().info("TF available! Processing ROI")

        self.pending_roi = None
        self.tf_wait_count = 0

        self.process_roi(
            x0, y0, x1, y1,
            depth, stamp, frame_id, encoding
        )

    def info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, rgb_msg, depth_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

        self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.depth_msg = depth_msg

        display = rgb.copy()

        if self.drag_start and self.drag_end:
            cv2.rectangle(
                display, self.drag_start, self.drag_end,
                (0, 255, 0), 2
            )

        cv2.imshow("RGB", display)
        cv2.waitKey(1)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0 = min(self.drag_start[0], x)
            x1 = max(self.drag_start[0], x)
            y0 = min(self.drag_start[1], y)
            y1 = max(self.drag_start[1], y)

            self.drag_start = None
            self.drag_end = None

            if x1 > x0 and y1 > y0:
                if self.depth is None or self.depth_msg is None:
                    self.get_logger().warn("No depth image yet")
                    return

                self.pending_roi = (
                    x0, y0, x1, y1,
                    self.depth.copy(),
                    self.depth_msg.header.stamp,
                    self.depth_msg.header.frame_id,
                    self.depth_msg.encoding
                )

                self.get_logger().info(
                    f"ROI selected: ({x0},{y0}) to ({x1},{y1})"
                )

    def process_roi(self, x0, y0, x1, y1,
    depth, stamp, frame_id, encoding):
        if self.depth is None or self.camera_info is None:
            self.get_logger().warn("Waiting for camera data")
            return

        # 1. Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # 2. Extract the depth region
        depth_roi = depth[y0:y1, x0:x1].astype(np.float32)
        if self.depth_msg.encoding == "16UC1":
            depth_roi /= 1000.0
        elif self.depth_msg.encoding != "32FC1":
            self.get_logger().error("Unsupported depth encoding")
            return

        # 3. Pixel coordinate grids
        u, v = np.meshgrid(
            np.arange(x0, x1),
            np.arange(y0, y1)
        )

        # 4. Reject invalid depths
        valid = np.isfinite(depth_roi) & (depth_roi > 0)

        Z = depth_roi[valid]
        u = u[valid]
        v = v[valid]

        if Z.size == 0:
            self.get_logger().warn("No valid depth points")
            return

        # 5. Back-project into camera optical frame
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_camera = np.column_stack((X, Y, Z))

        # 6. TF at the depth image timestamp
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link",
                frame_id,
                Time.from_msg(stamp)
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF unavailable: {exc}")
            return

        # 7. Quaternion -> rotation matrix
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w

        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])

        t = tf.transform.translation
        translation = np.array([t.x, t.y, t.z])

        # Transform all points at once
        points_base = points_camera @ R.T + translation

        self.get_logger().info(
            f"Valid points: {len(points_base)}, "
            f"Z range in base_link: "
            f"{points_base[:, 2].min():.3f} to "
            f"{points_base[:, 2].max():.3f} m"
        )

        # 8. Optional height filter.
        # Set these after examining the Z range in base_link.
        Z_MIN_BASE = 0.65
        Z_MAX_BASE = 0.7

        if Z_MIN_BASE is not None and Z_MAX_BASE is not None:
            mask = (
                (points_base[:, 2] >= Z_MIN_BASE) &
                (points_base[:, 2] <= Z_MAX_BASE)
            )
            points_base = points_base[mask]

        if len(points_base) == 0:
            self.get_logger().warn("No points after filtering")
            return

        # 8. Cirle fit
        points_xy = points_base[:, :2]

        # Known cylinder radius
        radius = 0.03

        # Initial guess: mean of observed points
        initial_center = np.mean(points_xy, axis=0)
        # Residual function
        def residuals(center):
            distances = np.linalg.norm(
                points_xy - center, axis=1
            )
            return distances - radius

        # Optimize circle center
        result = least_squares(
            residuals,
            initial_center,
            loss="soft_l1",
            f_scale=0.003
        )

        cx, cy = result.x

        self.get_logger().info(f"Estimated cylinder center: "f"X={cx:.4f}, Y={cy:.4f}")

        # 9. Publish to RViz
        header = Header()
        header.frame_id = "base_link"
        header.stamp = stamp

        cloud = point_cloud2.create_cloud_xyz32(
            header, points_base.astype(np.float32)
        )

        self.cloud_pub.publish(cloud)
        self.get_logger().info(
            f"Published {len(points_base)} 3D points"
        )

        marker = Marker() #center marker
        
        # Set the frame ID and timestamp
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "cylinder_center"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = points_base[:, 2].mean()
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Diameter of the sphere
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0  
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    
    # node1 = ImageClickListener()
    node2 = ROIPointCloud()
    try:
        # rclpy.spin(node1)
        rclpy.spin(node2)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        # node1.destroy_node()
        node2.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
