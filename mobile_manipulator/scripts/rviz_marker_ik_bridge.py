#!/usr/bin/env python3
"""Work around MoveIt 2 RViz goal marker snap-back on ROS 2 Jazzy.

After a goal interactive marker MOUSE_UP, request IK and publish the resulting
RobotState to RViz's custom goal-state input. Does NOT execute trajectories.
"""

import copy
import threading

import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarkerFeedback


class MarkerIkBridge(Node):
    def __init__(self):
        super().__init__('rviz_marker_ik_bridge')
        self.declare_parameter('group', 'Kinova_arm')
        self.declare_parameter('tip_link', 'left_hand_palm_link')
        self.declare_parameter('joint_states_topic', '/a200_0000/platform/joint_states')
        self.declare_parameter('feedback_topic',
            '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback')
        self.declare_parameter('goal_topic', '/rviz/moveit/update_custom_goal_state')
        self.declare_parameter('ik_timeout', 0.25)
        self.group = self.get_parameter('group').value
        self.tip = self.get_parameter('tip_link').value
        self._lock = threading.Lock()
        self._joints = None
        self._pending = None
        self._busy = False
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.publisher = self.create_publisher(RobotState, self.get_parameter('goal_topic').value, 10)
        js_qos = QoSProfile(depth=10,
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(JointState, self.get_parameter('joint_states_topic').value,
                                 self._on_joint_state, js_qos)
        self.create_subscription(InteractiveMarkerFeedback,
                                 self.get_parameter('feedback_topic').value,
                                 self._on_feedback, 50)
        self.get_logger().info('Goal marker IK bridge ready for group=%s tip=%s' % (self.group, self.tip))
        self.get_logger().info('Requires /compute_ik and live joint states. Does not move robot automatically.')

    def _on_joint_state(self, msg):
        with self._lock:
            self._joints = copy.deepcopy(msg)

    def _on_feedback(self, msg):
        if msg.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        if not msg.marker_name.startswith('EE:goal_'):
            return
        with self._lock:
            self._pending = copy.deepcopy(msg)
            if self._busy:
                return
            if self._joints is None:
                self.get_logger().warning('No joint states received yet')
                return
            self._send_locked()

    def _send_locked(self):
        if self._pending is None or self._joints is None or not self.ik.service_is_ready():
            if not self.ik.service_is_ready():
                self.get_logger().warning('/compute_ik service is not ready')
            return
        fb = self._pending
        self._pending = None
        self._busy = True
        req = GetPositionIK.Request()
        req.ik_request.group_name = str(self.group)
        req.ik_request.ik_link_name = str(self.tip)
        req.ik_request.pose_stamped.header = fb.header
        req.ik_request.pose_stamped.pose = fb.pose
        req.ik_request.robot_state.joint_state = copy.deepcopy(self._joints)
        # Match the published workaround: solve pose IK independently of scene
        # collisions. MoveIt Plan MUST validate collisions before any Execute.
        req.ik_request.avoid_collisions = False
        timeout = float(self.get_parameter('ik_timeout').value)
        whole = int(timeout)
        req.ik_request.timeout = Duration(sec=whole, nanosec=int((timeout-whole)*1e9))
        future = self.ik.call_async(req)
        future.add_done_callback(self._on_ik_result)

    def _on_ik_result(self, future):
        try:
            response = future.result()
            if response and response.error_code.val == MoveItErrorCodes.SUCCESS:
                response.solution.is_diff = False
                self.publisher.publish(response.solution)
                self.get_logger().info('IK succeeded: published goal state to RViz')
            else:
                code = response.error_code.val if response else 'no response'
                self.get_logger().warning(f'IK rejected goal pose (code {code})')
        except Exception as exc:
            self.get_logger().error(f'IK request error: {exc}')
        finally:
            with self._lock:
                self._busy = False
                if self._pending is not None:
                    self._send_locked()


def main():
    rclpy.init()
    node = MarkerIkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
