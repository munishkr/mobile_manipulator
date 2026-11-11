
import math
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

NS = "/a200_0000/arm_0_joint_trajectory_controller"
URDF = "/tmp/sim_robot.urdf"

STEP = 0.3
DURATION = 4
TOLERANCE = 0.05

JOINTS = [f"arm_0_joint_{i}" for i in range(1, 8)]


class ArmTester(Node):
    def __init__(self):
        super().__init__("kinova_all_joint_test")

        self.state = None

        self.create_subscription(
            JointTrajectoryControllerState,
            NS + "/controller_state",
            self.state_callback,
            10
        )

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            NS + "/follow_joint_trajectory"
        )

    def state_callback(self, msg):
        self.state = msg

    def wait_state(self):
        deadline = time.monotonic() + 10

        while self.state is None:
            if time.monotonic() > deadline:
                raise RuntimeError("No controller state")
            rclpy.spin_once(self, timeout_sec=0.1)

    def get_positions(self):
        return dict(zip(
            self.state.joint_names,
            self.state.feedback.positions
        ))

    def move(self, targets):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS

        point = JointTrajectoryPoint()
        point.positions = [targets[j] for j in JOINTS]
        point.time_from_start = Duration(sec=DURATION)

        goal.trajectory.points = [point]

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=10
        )

        if not future.done():
            raise RuntimeError("Goal submission timed out")

        handle = future.result()

        if not handle.accepted:
            raise RuntimeError("Trajectory rejected")

        result_future = handle.get_result_async()

        # Allow for slower-than-real-time simulation.
        deadline = time.monotonic() + 60

        while not result_future.done():
            if time.monotonic() > deadline:
                handle.cancel_goal_async()
                raise RuntimeError(
                    "Trajectory timed out; check Gazebo pause/RTF"
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        result = result_future.result().result

        if result.error_code != 0:
            raise RuntimeError(
                f"Trajectory error: {result.error_code} "
                f"{result.error_string}"
            )

        rclpy.spin_once(self, timeout_sec=0.1)


def read_limits():
    root = ET.parse(URDF).getroot()
    limits = {}

    for name in JOINTS:
        joint = root.find(f"./joint[@name='{name}']")

        if joint is None:
            raise RuntimeError(f"Missing joint: {name}")

        limit = joint.find("limit")

        if joint.get("type") == "continuous":
            limits[name] = None
        elif limit is not None:
            limits[name] = (
                float(limit.get("lower")),
                float(limit.get("upper"))
            )
        else:
            raise RuntimeError(f"Missing limits for {name}")

    return limits


def choose_target(current, limits):
    if limits is None:
        return current + STEP

    lower, upper = limits
    margin = 0.05

    if current + STEP <= upper - margin:
        return current + STEP

    if current - STEP >= lower + margin:
        return current - STEP

    raise RuntimeError("Not enough travel within joint limits")


def main():
    rclpy.init()
    node = ArmTester()

    try:
        node.wait_state()

        if not node.client.wait_for_server(timeout_sec=10):
            raise RuntimeError("Arm action server unavailable")

        limits = read_limits()

        initial = node.get_positions()

        if not all(j in initial for j in JOINTS):
            raise RuntimeError("Missing arm joint state")

        print("\nKinova Gen3 - 7 Joint Test")
        print("=" * 45)

        for name in JOINTS:
            # Preserve the other joints' current positions
            start_positions = node.get_positions()
            current = start_positions[name]

            if not math.isfinite(current):
                raise RuntimeError(f"Invalid position: {name}")

            target = choose_target(current, limits[name])

            command = start_positions.copy()
            command[name] = target

            print(f"\nTesting {name}")
            print(f"  Start:  {current:.4f} rad")
            print(f"  Target: {target:.4f} rad")

            node.move(command)

            actual = node.get_positions()[name]
            error = abs(actual - target)

            print(f"  Actual: {actual:.4f} rad")
            print(f"  Error:  {error:.4f} rad")

            if error > TOLERANCE:
                raise RuntimeError(
                    f"{name} did not reach its target"
                )

            # Return to the starting configuration
            print("  Returning to starting position...")
            node.move(start_positions)

            returned = node.get_positions()[name]

            if abs(returned - current) > TOLERANCE:
                raise RuntimeError(
                    f"{name} failed to return"
                )

            print(f"  PASS: {name}")

        print("\n" + "=" * 45)
        print("ALL SEVEN KINOVA JOINTS PASSED")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
