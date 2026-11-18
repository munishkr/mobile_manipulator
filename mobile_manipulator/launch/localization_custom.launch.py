"""clearpath_nav2_demos/localization.launch.py with a selectable params file.

Identical to upstream except `params_file` defaults to
../config/nav2/a200_localization.yaml (AMCL tuned for this simulation).
"""
import os

from ament_index_python.packages import get_package_share_directory
from clearpath_config.clearpath_config import ClearpathConfig
from clearpath_config.common.utils.yaml import read_yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from nav2_common.launch import RewrittenYaml

DEFAULT_PARAMS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'nav2',
    'a200_localization.yaml'))

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='false',
                          choices=['true', 'false']),
    DeclareLaunchArgument('setup_path', default_value='/etc/clearpath/'),
    DeclareLaunchArgument('scan_topic', default_value=''),
    DeclareLaunchArgument('map', description='Occupancy map YAML'),
    DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS),
]


def launch_setup(context, *args, **kwargs):
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    setup_path = LaunchConfiguration('setup_path').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f'Localization params file not found: {params_file}')
    namespace = ClearpathConfig(read_yaml(os.path.join(setup_path, 'robot.yaml'))).system.namespace
    scan_topic = LaunchConfiguration('scan_topic').perform(context) or \
        f'/{namespace}/sensors/lidar2d_0/scan'
    rewritten = RewrittenYaml(source_file=params_file,
                              param_rewrites={'scan_topic': scan_topic},
                              convert_types=True)
    return [GroupAction([
        PushRosNamespace(namespace),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [pkg_nav2_bringup, 'launch', 'localization_launch.py'])),
            launch_arguments=[('namespace', namespace),
                              ('map', LaunchConfiguration('map')),
                              ('use_sim_time', LaunchConfiguration('use_sim_time')),
                              ('params_file', rewritten)]),
    ])]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
