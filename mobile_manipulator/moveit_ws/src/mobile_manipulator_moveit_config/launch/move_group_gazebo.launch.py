
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            "a200-0000",
            package_name="mobile_manipulator_moveit_config",
        )
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
           {
                "use_sim_time": True,
                "trajectory_execution.allowed_execution_duration_scaling": 5.0,  # Gazebo RTF dips to ~0.5; MoveIt times out in wall time
                "trajectory_execution.allowed_goal_duration_margin": 2.0,
            },
        ],
        remappings=[
            ("/joint_states", "/a200_0000/platform/joint_states"),
            ("joint_states", "/a200_0000/platform/joint_states"),
        ],
    )

    return LaunchDescription([move_group])
