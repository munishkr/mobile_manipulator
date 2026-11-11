
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

NS = "/a200_0000/dex3_joint_trajectory_controller"

rclpy.init()
node = Node("dex3_all_fingers_test")

latest = [None]

def callback(msg):
    latest[0] = msg

node.create_subscription(
    JointTrajectoryControllerState,
    NS + "/controller_state",
    callback,
    10
)

publisher = node.create_publisher(
    JointTrajectory,
    NS + "/joint_trajectory",
    10
)

print("Waiting for controller state...")

# Wait for the controller subscriber
deadline = time.monotonic() + 5

while publisher.get_subscription_count() == 0:
    if time.monotonic() >= deadline:
        raise RuntimeError("Controller subscriber not found")
    rclpy.spin_once(node, timeout_sec=0.1)
    
if latest[0] is None:
    raise RuntimeError("No Dex3 controller state received")

names = list(latest[0].joint_names)
initial = list(latest[0].feedback.positions)

if len(names) != 7 or len(initial) != 7:
    raise RuntimeError("Expected seven Dex3 joints")

root = ET.parse("/tmp/sim_robot.urdf").getroot()

targets = []

for name, current in zip(names, initial):
    joint = root.find(f"./joint[@name='{name}']")
    if joint is None:
        raise RuntimeError(f"Missing URDF joint: {name}")

    limit = joint.find("limit")
    lower = float(limit.get("lower"))
    upper = float(limit.get("upper"))

    midpoint = (lower + upper) / 2

    # Move at most 0.3 rad toward the range midpoint
    delta = max(-0.3, min(0.3, midpoint - current))
    target = max(lower, min(upper, current + delta))

    targets.append(target)
    print(f"{name}: {current:.3f} -> {target:.3f}")

trajectory = JointTrajectory()
trajectory.joint_names = names

def add_point(positions, seconds):
    point = JointTrajectoryPoint()
    point.positions = list(positions)
    point.time_from_start = Duration(sec=seconds)
    trajectory.points.append(point)

# Move in 3 seconds, hold for 8 seconds,
# then return in 3 seconds
add_point(targets, 3)
# add_point(targets, 11)
# add_point(initial, 14)

# Monitor AFTER publishing
start = time.monotonic()
last_print = -1

while time.monotonic() - start < 20:
    rclpy.spin_once(node, timeout_sec=0.1)

    elapsed = int(time.monotonic() - start)

    if elapsed != last_print and elapsed % 2 == 0:
        last_print = elapsed

        state = latest[0]

        print(f"\nElapsed wall time: {elapsed}s")

        for i, name in enumerate(names):
            print(
                f"{name}: "
                f"desired={state.reference.positions[i]:.3f}, "
                f"actual={state.feedback.positions[i]:.3f}"
            )

print("\nFinal joint positions:")
for name, pos in zip(names, latest[0].feedback.positions):
    print(f"{name}: {pos:.3f}")

node.destroy_node()
rclpy.shutdown()
