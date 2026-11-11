from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder(
            "a200-0000",
            package_name="mobile_manipulator_moveit_config",
        )
        .to_moveit_configs()
    )

    # MotionPlanning + wrist camera (RGB image, depth cloud, optical-frame axes).
    rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution([
            FindPackageShare("mobile_manipulator_moveit_config"),
            "config",
            "mm_camera.rviz",
        ]),
    )

    # gdb:=true runs RViz under gdb and prints every thread's backtrace into the
    # log if it segfaults (the MotionPlanning panel has crashed on Execute).
    gdb = DeclareLaunchArgument("gdb", default_value="false")
    gdb_prefix = PythonExpression([
        "'gdb -q -batch -iex \"set debuginfod enabled off\" -ex \"handle SIGPIPE nostop noprint\" -ex run -ex \"thread apply all bt\" --args' if '",
        LaunchConfiguration("gdb"), "'.lower() == 'true' else ''",
    ])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        prefix=gdb_prefix,
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
        remappings=[
            ("/joint_states", "/a200_0000/platform/joint_states"),
            ("joint_states", "/a200_0000/platform/joint_states"),
            # Clearpath's robot_state_publisher publishes TF in the robot
            # namespace; nothing publishes on /tf. Without these, RViz cannot
            # place sensor data (camera frames) relative to base_link.
            ("/tf", "/a200_0000/tf"),
            ("/tf_static", "/a200_0000/tf_static"),
        ],
    )

    return LaunchDescription([rviz_config, gdb, rviz_node])
