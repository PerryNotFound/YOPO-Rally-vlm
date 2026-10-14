import os
from glob import glob

from setuptools import find_packages, setup

package_name = "diablo_ros_wrapper"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.py")),
        ),
        (os.path.join("share", package_name, "configs"), glob("configs/*")),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chy",
    maintainer_email="1274653465@qq.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mpc_node = diablo_ros_wrapper.mpc_node:main",
            "yopo_node = diablo_ros_wrapper.yopo_node:main",
            "depolyment_msg_convertor = diablo_ros_wrapper.depolyment_msg_convertor:main",
            "waypoint_tracker_node = diablo_ros_wrapper.waypoint_tracker_node:main",
        ],
    },
)
