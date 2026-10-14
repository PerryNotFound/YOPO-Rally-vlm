import os

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace, SetParameter


def generate_launch_description():
    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    gui_lc = LaunchConfiguration("gui")

    ctrl_arg = DeclareLaunchArgument("ctrl", default_value="false")
    ctrl_lc = LaunchConfiguration("ctrl")

    vins_arg = DeclareLaunchArgument("vins", default_value="false")
    vins_lc = LaunchConfiguration("vins")

    depthai_arg = DeclareLaunchArgument("depthai", default_value="false")
    depthai_lc = LaunchConfiguration("depthai")

    diablo_ctrl_node = Node(
        package="diablo_ctrl",
        executable="diablo_ctrl_node",
        name="diablo_ctrl_node",
        output="screen",
        condition=IfCondition(ctrl_lc),
    )

    depolyment_msg_convertor_node = Node(
        package="diablo_ros_wrapper",
        executable="depolyment_msg_convertor",
        name="depolyment_msg_convertor",
        output="screen",
    )

    vins_node = Node(
        package="vins",
        executable="vins_node",
        name="vins_node",
        output="screen",
        arguments=[
            "~/Workspaces/vins_ws/src/VINS-Fusion-ROS2-humble-arm/config/oak-d-pro/config.yaml",
        ],
        condition=IfCondition(vins_lc),
    )

    diablo_algorithm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("diablo_ros_wrapper"),
                "launch",
                "diablo_algorithm.launch.py",
            )
        ),
        launch_arguments={
            "depth_topic": "/oak/stereo/image_raw",
        }.items(),
    )
    depthai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("depthai_examples"),
                "launch",
                "stereo_inertial_node.launch.py",
            ),
        ),
        launch_arguments={
            "depth_aligned": "false",
            "enableRviz": "false",
        }.items(),
        condition=IfCondition(depthai_lc),
    )
    visualize_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("diablo_ros_wrapper"),
                "launch",
                "diablo_visualize.launch.py",
            )
        ),
        launch_arguments={"gui": gui_lc}.items(),
    )

    return LaunchDescription(
        [
            gui_arg,
            ctrl_arg,
            vins_arg,
            depthai_arg,
            diablo_ctrl_node,
            depolyment_msg_convertor_node,
            vins_node,
            diablo_algorithm_launch,
            depthai_launch,
            visualize_launch,
        ]
    )
