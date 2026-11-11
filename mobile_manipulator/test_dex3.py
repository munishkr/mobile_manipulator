
import time
import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

STATE_TOPIC = "/a200_0000/dex3_joint_trajectory_controller/controller_state"
COMMAND_TOPIC = "/a200_0000/dex3_joint_trajectory_controller/joint_trajectory"
TEST_JOINT = "left_hand_thumb_0_joint"

rclpy.init()
node = Node("dex3_movement_test")

latest = [None]

def state_callback(msg):
    latest[0] = msg

sub = node.create_subscription(
    JointTrajectoryControllerState,
    STATE_TOPIC,
    state_callback,
    10,
)

print("Waiting for Dex3 controller state...")

deadline = time.monotonic() + 10
while latest[0] is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)

if latest[0] is None:
    raise RuntimeError("No Dex3 controller state received")

state = latest[0]
names = list(state.joint_names)
positions = list(state.feedback.positions)

if TEST_JOINT not in names or len(positions) != len(names):
    raise RuntimeError("Unexpected controller joint state")

root = ET.parse("/tmp/sim_robot.urdf").getroot()
joint = root.find(f"./joint[@name='{TEST_JOINT}']")
limit = joint.find("limit") if joint is not None else None

if limit is None:
    raise RuntimeError("Joint limits not found in URDF")

lower = float(limit.get("lower"))
upper = float(limit.get("upper"))

index = names.index(TEST_JOINT)
current = positions[index]

if not math.isfinite(current):
    raise RuntimeError("Invalid joint position")

if current + 0.7 <= upper:
    target = current + 0.7
else:
    target = max(lower, current - 0.7)

if abs(target - current) < 0.05:
    raise RuntimeError("Insufficient joint travel for test")

print(f"Joint:   {TEST_JOINT}")
print(f"Limits:  [{lower:.3f}, {upper:.3f}] rad")
print(f"Current: {current:.3f} rad")
print(f"Target:  {target:.3f} rad")

positions[index] = target

pub = node.create_publisher(
    JointTrajectory,
    COMMAND_TOPIC,
    10,
)

deadline = time.monotonic() + 5
while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)

if pub.get_subscription_count() == 0:
    raise RuntimeError("No trajectory controller subscriber")

trajectory = JointTrajectory()
trajectory.joint_names = names

point = JointTrajectoryPoint()
point.positions = positions
point.time_from_start = Duration(sec=2)

trajectory.points.append(point)

pub.publish(trajectory)
print("Trajectory sent")

deadline = time.monotonic() + 3
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

final = latest[0]
print(
    f"Final:   {final.feedback.positions[index]:.3f} rad"
)

node.destroy_node()
rclpy.shutdown()
