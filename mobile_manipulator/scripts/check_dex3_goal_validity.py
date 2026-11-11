#!/usr/bin/env python3
"""Read-only validity checks for Dex3 current, ready, flexed and partial poses."""
import copy
import sys
import rclpy
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from moveit_msgs.msg import PlanningSceneComponents

JOINTS = [
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
    'left_hand_index_0_joint', 'left_hand_index_1_joint',
]
POSES = {
    'hand_ready': [0.0, 0.0, 0.2, -0.2, -0.2, -0.2, -0.2],
    'hand_flexed': [0.0, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4],
    'partial_closure': [0.0, 0.2, 0.3, -0.3, -0.3, -0.3, -0.3],
}

def call(node, client, request, label):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{label} did not complete')
    return future.result()

def main():
    rclpy.init()
    node = rclpy.create_node('check_dex3_goal_validity')
    try:
        scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
        validity_client = node.create_client(GetStateValidity, '/check_state_validity')
        for client, name in [(scene_client,'/get_planning_scene'), (validity_client,'/check_state_validity')]:
            if not client.wait_for_service(timeout_sec=8.0):
                raise RuntimeError(f'Service unavailable: {name}')
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ROBOT_STATE
        response = call(node, scene_client, request, 'scene request')
        state = response.scene.robot_state
        state.is_diff = False
        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)
        missing = sorted(set(JOINTS) - set(names))
        if missing:
            raise RuntimeError('Missing joints: ' + ', '.join(missing))
        indices = {n:names.index(n) for n in JOINTS}
        print('Current Dex3 positions (rad):')
        for j in JOINTS:
            print(f'  {j}: {positions[indices[j]]:+.4f}')
        for label, targets in [('current', None), *POSES.items()]:
            candidate = copy.deepcopy(state)
            if targets is not None:
                p = list(candidate.joint_state.position)
                for name, value in zip(JOINTS, targets):
                    p[indices[name]] = value
                candidate.joint_state.position = p
                # Avoid carrying stale joint velocities, if provided.
                candidate.joint_state.velocity = []
                candidate.joint_state.effort = []
            check = GetStateValidity.Request()
            check.robot_state = candidate
            check.group_name = 'dex3_hand'
            result = call(node, validity_client, check, f'{label} validity')
            print(f'\n{label}: valid={result.valid}')
            if result.contacts:
                pairs = sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in result.contacts})
                for a, b in pairs:
                    print(f'  COLLISION: {a} <--> {b}')
            if result.cost_sources:
                print(f'  cost_sources={len(result.cost_sources)}')
            if result.constraint_result:
                failed = [c for c in result.constraint_result if not c.result]
                for c in failed:
                    print(f'  FAILED CONSTRAINT: {c.distance:.4g}')
        print('\nRead-only: no robot motions or scene updates were requested.')
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
