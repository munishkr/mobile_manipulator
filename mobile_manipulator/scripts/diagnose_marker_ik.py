#!/usr/bin/env python3
"""Read-only MoveIt/RViz interactive goal IK diagnostic for the user's robot.

Does not publish robot commands or alter any planning-scene objects.
"""
import sys
import rclpy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, GetPositionIK
from visualization_msgs.srv import GetInteractiveMarkers

SCENE_SERVICE = '/get_planning_scene'
IK_SERVICE = '/compute_ik'
MARKER_SERVICE = ('/rviz_moveit_motion_planning_display/'
                  'robot_interaction_interactive_marker_topic/get_interactive_markers')
GOAL_MARKER = 'EE:goal_left_hand_palm_link'


def request(node, service_type, name, req):
    client = node.create_client(service_type, name)
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f'Service missing: {name}')
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'Timed out or failed: {name}')
    return future.result()


def main():
    rclpy.init()
    node = rclpy.create_node('diagnose_kinova_rviz_marker_ik')
    try:
        marker_resp = request(node, GetInteractiveMarkers, MARKER_SERVICE,
                              GetInteractiveMarkers.Request())
        marker = next((m for m in marker_resp.markers if m.name == GOAL_MARKER), None)
        if marker is None:
            names = [m.name for m in marker_resp.markers]
            raise RuntimeError(f'Goal marker not present; current markers: {names}')
        print(f'Goal marker frame: {marker.header.frame_id!r}')
        print(f'Goal marker position: {marker.pose.position.x:.4f}, '
              f'{marker.pose.position.y:.4f}, {marker.pose.position.z:.4f}')
        if not marker.header.frame_id:
            raise RuntimeError('Goal marker frame is empty')

        scene_req = GetPlanningScene.Request()
        scene_req.components.components = PlanningSceneComponents.ROBOT_STATE
        scene_resp = request(node, GetPlanningScene, SCENE_SERVICE, scene_req)
        joint_names = scene_resp.scene.robot_state.joint_state.name
        arm_names = [f'arm_0_joint_{i}' for i in range(1, 8)]
        missing = sorted(set(arm_names) - set(joint_names))
        print('Current scene has all seven arm joint states:', not missing)
        if missing:
            print('Missing:', missing)
            return 2

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'Kinova_arm'
        ik_req.ik_request.ik_link_name = 'left_hand_palm_link'
        ik_req.ik_request.robot_state = scene_resp.scene.robot_state
        ik_req.ik_request.pose_stamped.header.frame_id = marker.header.frame_id
        ik_req.ik_request.pose_stamped.pose = marker.pose
        ik_req.ik_request.pose_stamped.pose.position.z += 0.005  # Only 5 mm, no execution
        ik_req.ik_request.timeout.sec = 2
        ik_req.ik_request.avoid_collisions = False

        ik_resp = request(node, GetPositionIK, IK_SERVICE, ik_req)
        code = ik_resp.error_code.val
        print(f'5-mm upward IK test result code: {code}')
        if code == 1:
            print('IK succeeded. Focus next on RViz goal-state update/interactive plugin.')
            solved = dict(zip(ik_resp.solution.joint_state.name,
                              ik_resp.solution.joint_state.position))
            print('Solved Kinova joints:', [round(solved[j], 4) for j in arm_names])
        else:
            print('IK did not succeed. Focus next on IK model, seed state, and constraints.')
        return 0
    except Exception as exc:
        print(f'DIAGNOSTIC ERROR: {exc}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
