#!/usr/bin/env python3
"""Check current Dex3 state validity against MoveIt's active planning scene.

Read-only: does not move the robot or modify the planning scene.
"""

import sys
import rclpy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, GetStateValidity


def wait_for_result(node, future, seconds=10.0):
    rclpy.spin_until_future_complete(node, future, timeout_sec=seconds)
    if not future.done():
        raise TimeoutError('MoveIt service request timed out')
    result = future.result()
    if result is None:
        raise RuntimeError(f'MoveIt service request failed: {future.exception()}')
    return result


def main():
    rclpy.init()
    node = rclpy.create_node('inspect_dex3_collisions')
    try:
        scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
        validity_client = node.create_client(GetStateValidity, '/check_state_validity')
        for name, client in [('/get_planning_scene', scene_client),
                             ('/check_state_validity', validity_client)]:
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f'{name} is unavailable')

        scene_req = GetPlanningScene.Request()
        scene_req.components.components = PlanningSceneComponents.ROBOT_STATE
        scene_response = wait_for_result(node, scene_client.call_async(scene_req))
        state = scene_response.scene.robot_state
        state.is_diff = False

        available = set(state.joint_state.name)
        dex3 = [n for n in available if n.startswith('left_hand_') and n.endswith('_joint')]
        print(f'Dex3 joints in retrieved robot state: {len(dex3)}')
        if not dex3:
            print('WARNING: No Dex3 joints in the returned state; results may be incomplete.')

        validity_req = GetStateValidity.Request()
        validity_req.robot_state = state
        validity_req.group_name = 'dex3_hand'
        # Jazzy GetStateValidity request has no contacts/max_contacts fields.
        validity_response = wait_for_result(
            node, validity_client.call_async(validity_req)
        )

        print(f'Current state valid for dex3_hand: {validity_response.valid}')
        contacts = getattr(validity_response, 'contacts', [])
        pairs = sorted({tuple(sorted((c.contact_body_1, c.contact_body_2)))
                        for c in contacts})
        if pairs:
            print('Reported collision pairs:')
            for a, b in pairs:
                print(f'  {a} <--> {b}')
        else:
            print('No collision contacts returned.')
            if not validity_response.valid:
                print('NOTE: State may be invalid due to constraints/bounds, or contacts may not be reported.')

        for item in getattr(validity_response, 'constraint_result', []):
            if not item.result:
                print(f'Unsatisfied constraint: {item}')

    except (RuntimeError, TimeoutError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
