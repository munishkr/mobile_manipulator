"""clearpath_nav2_demos/nav2.launch.py with a selectable params file.

The upstream launch hard-codes config/<platform>/nav2.yaml. This wrapper is
identical except that:
  * `params_file` defaults to ../config/nav2/a200_nav2_slow.yaml (reduced
    speed/acceleration so AMCL does not jump while driving);
  * `use_composition` (default false) can load every Nav2 server into ONE
    component_container_isolated process instead of ~14 separate processes.
    Off by default: on this machine the composed MPPI controller segfaults
    while configuring CostCritic (tested 2026-11-17); separate processes work.
"""
import os

from ament_index_python.packages import get_package_share_directory
from clearpath_config.clearpath_config import ClearpathConfig
from clearpath_config.common.utils.yaml import read_yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from nav2_common.launch import RewrittenYaml

DEFAULT_PARAMS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'nav2',
    'a200_nav2_slow.yaml'))

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='false',
                          choices=['true', 'false']),
    DeclareLaunchArgument('setup_path', default_value='/etc/clearpath/'),
    DeclareLaunchArgument('scan_topic', default_value=''),
    DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS,
                          description='Nav2 parameter YAML'),
    DeclareLaunchArgument('use_composition', default_value='false',
                          choices=['true', 'false', 'True', 'False'],
                          description='Run all Nav2 servers in one container process'),
]


def launch_setup(context, *args, **kwargs):
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    use_sim_time = LaunchConfiguration('use_sim_time')
    # navigation_launch.py evaluates this inside PythonExpression: needs True/False.
    use_composition = LaunchConfiguration('use_composition').perform(context).capitalize()
    setup_path = LaunchConfiguration('setup_path').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f'Nav2 params file not found: {params_file}')

    config = ClearpathConfig(read_yaml(os.path.join(setup_path, 'robot.yaml')))
    namespace = config.system.namespace

    scan_topic = LaunchConfiguration('scan_topic').perform(context)
    if not scan_topic:
        scan_topic = f'/{namespace}/sensors/lidar2d_0/scan'

    rewritten = RewrittenYaml(source_file=params_file,
                              param_rewrites={'topic': scan_topic},
                              convert_types=True)

    return [GroupAction([
        PushRosNamespace(namespace),
        SetRemap(f'/{namespace}/odom', f'/{namespace}/platform/odom'),
        SetRemap('/tf', f'/{namespace}/tf'),
        SetRemap('/tf_static', f'/{namespace}/tf_static'),
        # Same container bringup_launch.py creates; navigation_launch.py loads the
        # servers into <namespace>/nav2_container when use_composition is true.
        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[rewritten, {'autostart': True,
                                    'use_sim_time': use_sim_time}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [pkg_nav2_bringup, 'launch', 'navigation_launch.py'])),
            launch_arguments=[('use_sim_time', use_sim_time),
                              ('params_file', rewritten),
                              ('use_composition', use_composition),
                              ('container_name', 'nav2_container'),
                              ('namespace', namespace)]),
    ])]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
