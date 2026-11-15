import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


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

def main(args=None):
    rclpy.init(args=args)
    node = ImageClickListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
