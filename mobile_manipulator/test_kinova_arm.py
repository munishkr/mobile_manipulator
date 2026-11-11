
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
TEST_JOINT = "arm_0_joint_2"
STEP = 0.20  # radians

rclpy.init()
node = Node("kinova_arm_test")

latest = [None]

def state_callback(msg):
    latest[0] = msg

subscription = node.create_subscription(
    JointTrajectoryControllerState,
    NS + "/controller_state",
    state_callback,
    10
)

client = ActionClient(
    node,
    FollowJointTrajectory,
    NS + "/follow_joint_trajectory"
)

print("Waiting for Kinova controller state...")

deadline = time.monotonic() + 10

while latest[0] is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)

if latest[0] is None:
    raise RuntimeError("No arm controller state received")

state = latest[0]
names = list(state.joint_names)
initial = list(state.feedback.positions)

if len(names) != 7 or len(initial) != 7:
    raise RuntimeError("Expected seven Kinova joints")

index = names.index(TEST_JOINT)
current = initial[index]

# Read joint limits
root = ET.parse("/tmp/sim_robot.urdf").getroot()
joint = root.find(f"./joint[@name='{TEST_JOINT}']")
limit = joint.find("limit")

lower = float(limit.get("lower"))
upper = float(limit.get("upper"))

margin = 0.05

if current + STEP <= upper - margin:
    target = current + STEP
elif current - STEP >= lower + margin:
    target = current - STEP
else:
    raise RuntimeError("Insufficient room within joint limits")

positions = initial.copy()
positions[index] = target

print(f"Joint:   {TEST_JOINT}")
print(f"Current: {current:.4f} rad")
print(f"Target:  {target:.4f} rad")

if not client.wait_for_server(timeout_sec=10):
    raise RuntimeError("Arm trajectory action server unavailable")

# Command all seven joints; move only joint 2
goal = FollowJointTrajectory.Goal()
goal.trajectory.joint_names = names

point = JointTrajectoryPoint()
point.positions = positions
point.time_from_start = Duration(sec=5)

goal.trajectory.points = [point]

future = client.send_goal_async(goal)

rclpy.spin_until_future_complete(
    node, future, timeout_sec=10
)

if not future.done():
    raise RuntimeError("Timed out submitting trajectory")

handle = future.result()

if not handle.accepted:
    raise RuntimeError("Arm controller rejected trajectory")

print("\nTrajectory accepted! Monitoring arm movement...")

result_future = handle.get_result_async()
start = time.monotonic()
next_print = 0

while time.monotonic() - start < 25:
    rclpy.spin_once(node, timeout_sec=0.2)

    elapsed = time.monotonic() - start

    if elapsed >= next_print:
        state = latest[0]

        desired = state.reference.positions[index]
        actual = state.feedback.positions[index]

        print(
            f"{elapsed:5.1f}s | "
            f"desired={desired:.4f} | "
            f"actual={actual:.4f}"
        )

        next_print += 2

    if result_future.done():
        print(
            "Action result code:",
            result_future.result().result.error_code
        )
        break

print("\nFinal position:",
      latest[0].feedback.positions[index])

node.destroy_node()
rclpy.shutdown()
