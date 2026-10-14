import numpy as np
import rclpy
import rclpy.qos
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Transform, TransformStamped, Twist
from motion_msgs.msg import MotionCtrl
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import (
    StaticTransformBroadcaster,
    TransformBroadcaster,
)

stand_height = 0.0


class DepolymentMsgConvertor(Node):
    def __init__(self):
        super().__init__("depolyment_msg_convertor")
        self.get_logger().info("Depolyment Msg Convertor started")

        self._motion_ctrl = MotionCtrl()
        self._motion_ctrl.value.up = stand_height
        self._map_odom_tf = StaticTransformBroadcaster(self)
        self._odom_base_link_tf = TransformBroadcaster(self)
        self._base_link_t265_tf = StaticTransformBroadcaster(self)
        self._base_link_oak_tf = StaticTransformBroadcaster(self)

        self._odom_base_link = Odometry()
        self._odom_base_link.header.frame_id = "odom"
        self._odom_base_link.child_frame_id = "base_link"

        self._cv_bridge = CvBridge()

        self._send_static_tf()
        self._handle_topics()

    def _handle_topics(self):
        self._image_visualize_camera_info_pub = self.create_publisher(
            CameraInfo,
            "/oak/left/image_rect/camera_info",
            rclpy.qos.qos_profile_system_default,
        )
        self._left_image_info_sub = self.create_subscription(
            CameraInfo,
            "/oak/left/camera_info",
            self._left_camera_info_callback,
            rclpy.qos.qos_profile_system_default,
        )
        self._diablo_msg_pub = self.create_publisher(
            MotionCtrl, "/diablo/MotionCmd", rclpy.qos.qos_profile_system_default
        )
        self._odom_base_link_pub = self.create_publisher(
            Odometry, "/odom", rclpy.qos.qos_profile_system_default
        )
        self._cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_callback, rclpy.qos.qos_profile_sensor_data
        )
        self._odom_t265_sub = self.create_subscription(
            Odometry,
            "/camera/pose/sample",
            self._odom_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._odom_vel_pub = self.create_publisher(
            PointStamped, "/visualize/odom_vel", rclpy.qos.qos_profile_system_default
        )

    def _left_camera_info_callback(self, msg: CameraInfo):
        self._image_visualize_camera_info_pub.publish(msg)

    def _send_static_tf(self):
        self._map_odom_tf.sendTransform(
            TransformStamped(
                header=Header(frame_id="map", stamp=self.get_clock().now().to_msg()),
                child_frame_id="odom",
                transform=Transform(),
            )
        )
        r = R.from_euler("xyz", [0, 0, 0], degrees=True)
        t = np.asarray([0.1, 0, 0.4])

        self._T_base_link_t265 = np.eye(4)
        self._T_base_link_t265[:3, :3] = r.as_matrix()
        self._T_base_link_t265[:3, 3] = t
        self._T_t265_base_link = np.linalg.inv(self._T_base_link_t265)

        tf = Transform()
        tf.translation.x = t[0]
        tf.translation.z = t[2]
        rotation = r.as_quat()
        tf.rotation.x = rotation[0]
        tf.rotation.y = rotation[1]
        tf.rotation.z = rotation[2]
        tf.rotation.w = rotation[3]
        self._base_link_t265_tf.sendTransform(
            TransformStamped(
                header=Header(
                    frame_id="base_link", stamp=self.get_clock().now().to_msg()
                ),
                child_frame_id="t265",
                transform=tf,
            )
        )

        r = R.from_euler("xyz", [0, 0, 0], degrees=True)
        t = np.asarray([0.2, 0, 0.3])

        tf = Transform()
        tf.translation.x = t[0]
        tf.translation.z = t[2]
        rotation = r.as_quat()
        tf.rotation.x = rotation[0]
        tf.rotation.y = rotation[1]
        tf.rotation.z = rotation[2]
        tf.rotation.w = rotation[3]
        self._base_link_oak_tf.sendTransform(
            TransformStamped(
                header=Header(
                    frame_id="base_link", stamp=self.get_clock().now().to_msg()
                ),
                child_frame_id="oak-d-base-frame",
                transform=tf,
            )
        )

    def _cmd_callback(self, cmd_msg: Twist):
        self._motion_ctrl.value.forward = cmd_msg.linear.x
        self._motion_ctrl.value.left = cmd_msg.angular.z

        self._diablo_msg_pub.publish(self._motion_ctrl)

    def _odom_callback(self, odom_t265_msg: Odometry):
        pos_t265 = odom_t265_msg.pose.pose.position
        rot_t265 = odom_t265_msg.pose.pose.orientation
        vel_t265 = odom_t265_msg.twist.twist.linear

        pos_t265 = np.asarray([pos_t265.x, pos_t265.y, pos_t265.z])
        rot_t265 = R.from_quat(
            [rot_t265.x, rot_t265.y, rot_t265.z, rot_t265.w]
        ).as_matrix()
        T_map_t265 = np.eye(4)
        T_map_t265[:3, :3] = rot_t265
        T_map_t265[:3, 3] = pos_t265

        vel_t265 = np.asarray([vel_t265.x, vel_t265.y, vel_t265.z])
        # NOTE: Transform velocity from t265 to map
        vel_t265 = rot_t265 @ vel_t265

        T_map_base_link = T_map_t265 @ self._T_t265_base_link
        pos_base_link = T_map_base_link[:3, 3]
        rot_base_link = R.from_matrix(T_map_base_link[:3, :3]).as_quat()
        vel_base_link = vel_t265

        self._odom_base_link.header.stamp = odom_t265_msg.header.stamp
        self._odom_base_link.pose.pose.position.x = pos_base_link[0]
        self._odom_base_link.pose.pose.position.y = pos_base_link[1]
        self._odom_base_link.pose.pose.position.z = pos_base_link[2]
        self._odom_base_link.pose.pose.orientation.x = rot_base_link[0]
        self._odom_base_link.pose.pose.orientation.y = rot_base_link[1]
        self._odom_base_link.pose.pose.orientation.z = rot_base_link[2]
        self._odom_base_link.pose.pose.orientation.w = rot_base_link[3]
        self._odom_base_link.twist.twist.linear.x = vel_base_link[0]
        self._odom_base_link.twist.twist.linear.y = vel_base_link[1]
        self._odom_base_link.twist.twist.linear.z = vel_base_link[2]

        self._odom_base_link_pub.publish(self._odom_base_link)
        tf = Transform()
        tf.translation.x = pos_base_link[0]
        tf.translation.y = pos_base_link[1]
        tf.translation.z = pos_base_link[2]
        tf.rotation.x = rot_base_link[0]
        tf.rotation.y = rot_base_link[1]
        tf.rotation.z = rot_base_link[2]
        tf.rotation.w = rot_base_link[3]

        self._odom_base_link_tf.sendTransform(
            TransformStamped(
                header=Header(frame_id="odom", stamp=odom_t265_msg.header.stamp),
                child_frame_id="base_link",
                transform=tf,
            )
        )

        point = PointStamped()
        point.header.stamp = odom_t265_msg.header.stamp
        point.header.frame_id = "odom"
        vel_dir = pos_base_link + vel_base_link
        point.point.x = vel_dir[0]
        point.point.y = vel_dir[1]
        point.point.z = vel_dir[2]
        self._odom_vel_pub.publish(point)


def main():
    rclpy.init()
    rclpy.spin(DepolymentMsgConvertor())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
