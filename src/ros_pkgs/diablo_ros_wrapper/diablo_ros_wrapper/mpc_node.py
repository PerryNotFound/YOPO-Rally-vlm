import time

import numpy as np
import rclpy
import rclpy.qos
from diablo_python.optimizer import MPCLocalPlanner
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

te = 8.0
mpc_period = 0.05
mpc_sampling_time = 0.25

vel_out_multiplier = 0.8
te /= vel_out_multiplier
mpc_sampling_time /= vel_out_multiplier
goal_tolerance = 1.0
goal_change_threshold = 2.0  # Reset MPC when goal changes by this distance


class MPCNode(Node):
    def __init__(self):
        super().__init__("MPC_node")
        self.get_logger().info("MPC node started")

        self._mpc_planner = MPCLocalPlanner()
        self._mpc_planner.setup_opti_params(
            v_max=2 * vel_out_multiplier,
            omega_max=np.pi * vel_out_multiplier,
            acc_max=2,
            alpha_max=np.pi,
            sampling_time=mpc_sampling_time,
            horizon=10,
        )
        self._mpc_planner.bake_optim_func(method="ipopt")

        self._current_state = np.zeros(3)

        self._reference_state = np.zeros((self._mpc_planner.N + 1, 3))
        self._current_z = 0.0

        self._odom_init = False

        self._last_pos_start: np.ndarray = np.zeros(2)
        self._last_vel_start: np.ndarray = np.zeros(2)

        self._sample_time = 0.0

        self._last_ref_valid = np.zeros(3)

        self._end_state_init = False
        self._reset_init_pose_flag = False
        self._mpc_start_flag = False

        # Goal pose tracking variables
        self._goal_pose = None
        self._goal_yaw = 0.0
        self._goal_position = np.zeros(2)
        self._goal_received = False

        # Historical goal pose for change detection
        self._previous_goal_position = np.zeros(2)
        self._goal_change_threshold = goal_change_threshold

        self._handle_topics()

    def _handle_topics(self):
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, rclpy.qos.qos_profile_sensor_data
        )
        self._end_state_sub = self.create_subscription(
            Odometry,
            "/end_state",
            self._end_state_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._init_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self._init_pose_callback,
            rclpy.qos.qos_profile_services_default,
        )
        self._mpc_timer = self.create_timer(mpc_period, self._mpc_callback)
        self._pred_path_pub = self.create_publisher(
            Path, "/visualize/pred_path", rclpy.qos.qos_profile_system_default
        )
        self._ref_path_pub = self.create_publisher(
            Path, "/visualize/ref_path", rclpy.qos.qos_profile_system_default
        )
        self._cmd_vel_pub = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )
        self._goal_pose_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self._goal_pose_callback,
            rclpy.qos.qos_profile_parameters,
        )

    def _goal_pose_callback(self, goal_pose: PoseStamped):
        # Extract new goal position
        new_goal_position = np.array(
            [goal_pose.pose.position.x, goal_pose.pose.position.y]
        )

        # Check if this is a significant goal change
        if self._goal_received:
            distance_change = np.linalg.norm(
                new_goal_position - self._previous_goal_position
            )
            if distance_change > self._goal_change_threshold:
                self.get_logger().info(
                    f"Goal pose changed significantly by {distance_change:.2f}m (threshold: {self._goal_change_threshold:.2f}m), resetting MPC"
                )
                self._reset_init_pose_flag = True

        # Store the previous goal position for next comparison
        self._previous_goal_position = new_goal_position.copy()

        # Store the goal pose information
        self._goal_pose = goal_pose
        self._goal_position[0] = goal_pose.pose.position.x
        self._goal_position[1] = goal_pose.pose.position.y

        # Extract yaw angle from goal pose quaternion
        quat = goal_pose.pose.orientation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        self._goal_yaw = rot.as_euler("xyz", degrees=False)[2]

        self._goal_received = True
        self.get_logger().info(
            f"Goal pose received: position=({self._goal_position[0]:.2f}, {self._goal_position[1]:.2f}), yaw={self._goal_yaw:.2f}"
        )

    def _init_pose_callback(self, _init_pose: PoseWithCovarianceStamped):
        # Clear last_pos_start and last_vel_start
        self._reset_init_pose_flag = True
        self.get_logger().info("Resetting initial pose")
        if not self._mpc_start_flag:
            self.get_logger().info("MPC started")
            self._mpc_start_flag = True

    def _odom_callback(self, odom: Odometry):
        self._odom = odom
        quat = odom.pose.pose.orientation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        yaw = rot.as_euler("xyz", degrees=False)[2]

        current_position = np.asarray(
            [odom.pose.pose.position.x, odom.pose.pose.position.y]
        )

        current_state = np.zeros(3)
        current_state[0] = odom.pose.pose.position.x
        current_state[1] = odom.pose.pose.position.y
        current_state[2] = yaw

        if not self._odom_init:
            self._last_pos_start = np.asarray(
                [odom.pose.pose.position.x, odom.pose.pose.position.y]
            )
            self._last_vel_start = np.asarray(
                [odom.twist.twist.linear.x, odom.twist.twist.linear.y]
            )
            self._odom_init = True
            self.get_logger().info("Resetting initial state")

        self._current_state = current_state
        self._current_z = odom.pose.pose.position.z

        if np.linalg.norm(self._last_pos_start - current_position) > 0.5:
            self._reset_init_pose_flag = True
            self.get_logger().info("Large jump in odometry, resetting initial pose")

    def _end_state_callback(self, end_state: Odometry):
        if not self._reset_init_pose_flag:
            last_pos_start = self._last_pos_start
            last_vel_start = self._last_vel_start
        else:
            last_pos_start = np.asarray(
                [self._odom.pose.pose.position.x, self._odom.pose.pose.position.y]
            )
            last_vel_start = np.asarray(
                [self._odom.twist.twist.linear.x, self._odom.twist.twist.linear.y]
            )
            self._reset_init_pose_flag = False
        self._sample_time = 0.0
        pos_end = np.asarray(
            [end_state.pose.pose.position.x, end_state.pose.pose.position.y]
        )
        vel_end = np.asarray(
            [end_state.twist.twist.linear.x, end_state.twist.twist.linear.y]
        )
        self._pos_spline_x = CubicHermiteSpline(
            [0, te],
            [last_pos_start[0], pos_end[0]],
            [last_vel_start[0], vel_end[0]],
            extrapolate=False,
        )
        self._pos_spline_y = CubicHermiteSpline(
            [0, te],
            [last_pos_start[1], pos_end[1]],
            [last_vel_start[1], vel_end[1]],
            extrapolate=False,
        )
        if not self._end_state_init:
            self._end_state_init = True

    def _mpc_callback(self):
        if not self._mpc_start_flag or not self._end_state_init or not self._odom_init:
            if not self._mpc_start_flag:
                self.get_logger().info(
                    "Waiting for MPC start...", throttle_duration_sec=1
                )
            if not self._end_state_init:
                self.get_logger().info(
                    "Waiting for end_state...", throttle_duration_sec=1
                )
            if not self._odom_init:
                self.get_logger().info("Waiting for odom...", throttle_duration_sec=1)
            return
        self._sample_time += mpc_period
        t_eval = np.linspace(
            self._sample_time,
            self._sample_time + self._mpc_planner.N * mpc_sampling_time,
            self._mpc_planner.N + 1,
        )

        ref_pos_x = self._pos_spline_x(t_eval)
        ref_pos_y = self._pos_spline_y(t_eval)
        ref_yaw = np.arctan2(np.diff(ref_pos_y), np.diff(ref_pos_x))
        ref_yaw = np.append(ref_yaw, ref_yaw[-1])

        # Check if goal is received and if the robot is close to the goal
        if self._goal_received and self._goal_pose is not None:
            current_pos = np.array([self._current_state[0], self._current_state[1]])
            distance_to_goal = np.linalg.norm(current_pos - self._goal_position)

            # If within goal tolerance, track the goal yaw angle
            if distance_to_goal <= goal_tolerance:
                self.get_logger().info(
                    f"Within goal tolerance ({distance_to_goal:.2f} <= {goal_tolerance}), tracking goal yaw: {self._goal_yaw:.2f}",
                    throttle_duration_sec=1,
                )
                # Use goal yaw for all reference points in the horizon
                ref_yaw = np.full_like(ref_yaw, self._goal_yaw)

        # If all values are nan, stop the robot
        if (
            np.all(np.isnan(ref_pos_x))
            or np.all(np.isnan(ref_pos_y))
            or np.all(np.isnan(ref_yaw))
        ):
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self._cmd_vel_pub.publish(cmd_vel)
            self.get_logger().info(
                "Reference trajectory finished", throttle_duration_sec=1
            )
            return
        else:
            # If some values are nan, fill them with the last valid value
            self._fill_nan_with_last_value(ref_pos_x)
            self._fill_nan_with_last_value(ref_pos_y)
            self._fill_nan_with_last_value(ref_yaw)

        reference_state = np.vstack((ref_pos_x, ref_pos_y, ref_yaw)).T

        current_state = self._current_state
        current_z = self._current_z
        start_time = time.time()
        u, pred_state = self._mpc_planner.solve_optim_warm_start(
            x_ref=reference_state, x_cur=current_state
        )
        end_time = time.time()
        time_elapsed = (end_time - start_time) * 1000
        self.get_logger().info(
            f"Time taken for mpc optimization: {time_elapsed:.4f} ms"
        )

        cmd_vel = Twist()
        cmd_vel.linear.x = u[0, 0]
        cmd_vel.angular.z = u[0, 1]
        self._cmd_vel_pub.publish(cmd_vel)

        self._last_pos_start = np.asarray([ref_pos_x[0], ref_pos_y[0]])
        ref_vel_x = self._pos_spline_x.derivative()(self._sample_time)
        ref_vel_y = self._pos_spline_y.derivative()(self._sample_time)
        self._last_vel_start = np.asarray([ref_vel_x, ref_vel_y])

        pred_path = Path()
        pred_path.header = Header()
        pred_path.header.frame_id = "odom"

        ref_path = Path()
        ref_path.header = Header()
        ref_path.header.frame_id = "odom"
        for i in range(self._mpc_planner.N + 1):
            pose = PoseStamped()
            pose.header = pred_path.header
            pose.pose.position.x = pred_state[i, 0]
            pose.pose.position.y = pred_state[i, 1]
            pose.pose.position.z = current_z
            quat = R.from_euler("xyz", [0, 0, pred_state[i, 2]]).as_quat()
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pred_path.poses.append(pose)
            pose = PoseStamped()
            pose.header = ref_path.header
            pose.pose.position.x = reference_state[i, 0]
            pose.pose.position.y = reference_state[i, 1]
            pose.pose.position.z = self._current_z
            quat = R.from_euler("xyz", [0, 0, reference_state[i, 2]]).as_quat()
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            ref_path.poses.append(pose)
        self._ref_path_pub.publish(ref_path)
        self._pred_path_pub.publish(pred_path)

    def _fill_nan_with_last_value(self, data):
        nan_indices = np.isnan(data)
        if np.any(nan_indices) and np.any(~nan_indices):
            last_valid_val = data[~nan_indices][-1]
            data = np.nan_to_num(data, copy=False, nan=last_valid_val)


def main():
    rclpy.init()
    mpc_node = MPCNode()
    rclpy.spin(mpc_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
