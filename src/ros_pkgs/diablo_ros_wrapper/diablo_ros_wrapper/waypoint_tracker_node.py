import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

import rclpy.qos

WAYPOINTS = [
    [10, -5, -np.pi],
    [-10, -5, np.pi / 2],
    [-5, -3, 0],
]

WAYPOINT_TOLERANCE = 1.0
FINE_TOLERANCE = 0.3
YAW_TOLERANCE = np.pi / 2
WAIT_TIME = 2.0


class WaypointTrackerNode(Node):
    def __init__(self):
        super().__init__("waypoint_tracker_node")

        self.waypoints = WAYPOINTS
        self.current_waypoint_index = 0
        self.waypoint_tolerance = WAYPOINT_TOLERANCE
        self.fine_tolerance = FINE_TOLERANCE
        self.yaw_cos_tolerance = np.cos(YAW_TOLERANCE)
        self.wait_time = WAIT_TIME

        self.current_position = np.array([0.0, 0.0])
        self.current_yaw = 0.0

        self.odom_received = False
        self.reached_coarse_tolerance = False
        self.wait_start_time = None

        self.odom_subscriber = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.goal_publisher = self.create_publisher(
            PoseStamped, "/goal_pose", rclpy.qos.qos_profile_parameters
        )

        self.timer = self.create_timer(0.1, self.check_waypoint_reached)
        self.goal_timer = self.create_timer(1.0, self.publish_goal_timer)

        self.get_logger().info(
            f"Waypoint tracker started with {len(self.waypoints)} waypoints"
        )

    def odom_callback(self, msg: Odometry):
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y

        orientation = msg.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rot = R.from_quat(quat)
        self.current_yaw = rot.as_euler("xyz")[2]

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info("Odometry received, starting waypoint tracking")

    def check_waypoint_reached(self):
        if not self.odom_received:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            return

        target_waypoint = self.waypoints[self.current_waypoint_index]
        target_position = np.array([target_waypoint[0], target_waypoint[1]])

        distance = np.linalg.norm(self.current_position - target_position)
        angle_cos = np.cos(self.current_yaw - target_waypoint[2])

        # Check coarse tolerance first
        if distance <= self.waypoint_tolerance and angle_cos >= self.yaw_cos_tolerance:
            if not self.reached_coarse_tolerance:
                self.reached_coarse_tolerance = True
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Reached waypoint {self.current_waypoint_index} coarse tolerance, starting wait"
                )

            # Check if fine tolerance is reached or wait time is exceeded
            fine_reached = (
                distance <= self.fine_tolerance and angle_cos >= self.yaw_cos_tolerance
            )
            wait_time_exceeded = (
                self.get_clock().now() - self.wait_start_time
            ).nanoseconds / 1e9 >= self.wait_time

            if fine_reached or wait_time_exceeded:
                self.get_logger().info(
                    f"Moving to next waypoint. Fine reached: {fine_reached}, Wait exceeded: {wait_time_exceeded}"
                )
                self.get_logger().info(
                    f"Distance error: {distance:.3f}m, Angle cos: {angle_cos:.3f}"
                )
                self.current_waypoint_index += 1
                self.reached_coarse_tolerance = False
                self.wait_start_time = None

                if self.current_waypoint_index < len(self.waypoints):
                    next_waypoint = self.waypoints[self.current_waypoint_index]
                    self.get_logger().info(
                        f"Next waypoint {self.current_waypoint_index}: {next_waypoint}"
                    )
                else:
                    self.get_logger().info("All waypoints completed!")
        else:
            self.reached_coarse_tolerance = False
            self.wait_start_time = None

    def publish_goal_timer(self):
        if not self.odom_received or self.current_waypoint_index >= len(self.waypoints):
            return

        waypoint = self.waypoints[self.current_waypoint_index]
        self.publish_goal_pose(waypoint)

    def publish_goal_pose(self, waypoint):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"

        goal_msg.pose.position.x = float(waypoint[0])
        goal_msg.pose.position.y = float(waypoint[1])
        goal_msg.pose.position.z = 0.0

        yaw = waypoint[2]
        quat = R.from_euler("xyz", [0, 0, yaw]).as_quat()
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]

        self.goal_publisher.publish(goal_msg)

    def get_current_status(self):
        if not self.odom_received:
            return "Waiting for odometry..."

        if self.current_waypoint_index >= len(self.waypoints):
            return "Task completed"

        target = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([target[0], target[1]])
        distance = np.linalg.norm(self.current_position - target_pos)

        return (
            f"Current: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}), "
            f"Target {self.current_waypoint_index}: ({target[0]}, {target[1]}), "
            f"Distance: {distance:.3f}m"
        )


def main():
    rclpy.init()
    waypoint_tracker = WaypointTrackerNode()
    rclpy.spin(waypoint_tracker)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
