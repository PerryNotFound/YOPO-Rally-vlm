import os

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odom",
        description="The topic name of odometry",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/depth/image",
        description="The topic name of depth",
    )
    odom_topic_ld = LaunchConfiguration("odom_topic")
    depth_topic_ld = LaunchConfiguration("depth_topic")
    return LaunchDescription(
        [
            odom_topic_arg,
            depth_topic_arg,
            Node(
                package="diablo_ros_wrapper",
                executable="yopo_node",
                name="yopo_node",
                output="screen",
                remappings=[
                    ("/odom", odom_topic_ld),
                    ("/camera/depth/image", depth_topic_ld),
                ],
                # ros_arguments=[
                #     "--log-level",
                #     "debug",
                # ],
            ),
            Node(
                package="diablo_ros_wrapper",
                executable="mpc_node",
                name="mpc_node",
                output="screen",
                remappings=[
                    ('/cmd_vel', '/cmd_vel_mpc'),
                  ]  
            ),
        ]
    )
