import os

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    gui_lc = LaunchConfiguration("gui")

    return LaunchDescription(
        [
            gui_arg,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("ros_tcp_endpoint"),
                        "launch",
                        "endpoint.py",
                    )
                )
            ),
            SetParameter(name="use_sim_time", value=True),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("diablo_ros_wrapper"),
                        "launch",
                        "diablo_algorithm.launch.py",
                    )
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("diablo_ros_wrapper"),
                        "launch",
                        "diablo_visualize.launch.py",
                    )
                ),
                launch_arguments={"gui": gui_lc}.items(),
            ),
        ]
    )
