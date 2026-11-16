
#!/usr/bin/env python3

import rclpy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from place_cup import gazebo_pose, cup_pose_in_base

FRAME = "base_link"

# Gazebo base_link world height
BASE_Z = 0.132279


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def make_box(name, pose, dimensions):
    obj = CollisionObject()
    obj.header.frame_id = FRAME
    obj.id = name

    shape = SolidPrimitive()
    shape.type = SolidPrimitive.BOX
    shape.dimensions = [float(v) for v in dimensions]

    obj.primitives.append(shape)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def make_cylinder(name, pose, height, radius):
    obj = CollisionObject()
    obj.header.frame_id = FRAME
    obj.id = name

    shape = SolidPrimitive()
    shape.type = SolidPrimitive.CYLINDER
    shape.dimensions = [float(height), float(radius)]

    obj.primitives.append(shape)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def main():
    rclpy.init()
    node = rclpy.create_node("add_tabletop_scene")

    client = node.create_client(
        ApplyPlanningScene,
        "/apply_planning_scene"
    )

    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "MoveIt apply_planning_scene service unavailable"
        )
        node.destroy_node()
        rclpy.shutdown()
        return
    base_world = gazebo_pose("a200_0000/robot")
    table_world = gazebo_pose("table")
    cup_world = gazebo_pose("cup")

    # Existing conservative MoveIt table collision box:
    # 0.8 x 0.6 x 0.75 m, centered 0.375 m above the table model origin.
    table_center_world = (
        table_world[0],
        table_world[1],
        table_world[2] + 0.375,
        *table_world[3:]
    )

    # Convert Gazebo world poses into the current robot base frame.
    table_base = cup_pose_in_base(table_center_world, base_world)
    cup_base = cup_pose_in_base(cup_world, base_world)

    table = make_box(
    "work_table", table_base, (0.8, 0.6, 0.75)
    )

    cup = make_cylinder(
    "target_cup", cup_base, 0.12, 0.035
    )

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = [table, cup]

    request = ApplyPlanningScene.Request()
    request.scene = scene

    future = client.call_async(request)
    rclpy.spin_until_future_complete(
        node, future, timeout_sec=10.0
    )

    result = future.result()

    if result is not None and result.success:
        node.get_logger().info(
            "Table and cup added to MoveIt planning scene"
        )
    else:
        node.get_logger().error(
            "Failed to apply planning scene"
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
