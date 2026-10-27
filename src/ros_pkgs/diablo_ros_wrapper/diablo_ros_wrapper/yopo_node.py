import time

import cv2
import cv_bridge
import numpy as np
import rclpy
import rclpy.clock
import rclpy.logging
import rclpy.qos
import rclpy.time
import tf2_geometry_msgs
import torch
from diablo_python.network import (
    CoordTransform,
    OutputTransform,
    StateTransform,
)
from diablo_python.utils import generate_splines
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Float32MultiArray, Header, Int32MultiArray
from tf2_ros import Buffer, TransformListener

max_depth = 20.0
image_size = (160, 32)

model_file = "models/yopo_net.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
te = 8.0
candidate_num = 5

vel_out_multiplier = 1.0
te /= vel_out_multiplier


nn_frequncy = 10

min_cost_jump_threshold = 0.2

# VLM 语义否决：/vlm_veto 超时时间（秒），超时未收到新掩码则回退纯 YOPO
vlm_veto_timeout = 2.0

np.set_printoptions(precision=3)


class YOPONode(Node):
    def __init__(self) -> None:
        super().__init__("YOPO_node")
        self.get_logger().info("YOPO node started")
        self._cv_bridge = cv_bridge.CvBridge()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._state_transform = StateTransform()
        self._coord_transform = CoordTransform()
        self._output_transform = OutputTransform()

        self._state: torch.Tensor = torch.zeros((4, 1, 5))
        self._state_valid = False
        self._image: torch.Tensor = torch.zeros((1, image_size[1], image_size[0]))
        self._image_valid = False
        self._image_stamp = 0.0
        self._last_image_stamp = 0.0

        net = torch.jit.load(model_file).eval().to(device)
        self._net = torch.jit.optimize_for_inference(net)

        self._T_pw: torch.Tensor = torch.eye(3).repeat(candidate_num, 1, 1)
        self._T_wp: torch.Tensor = torch.eye(3).repeat(candidate_num, 1, 1)
        self._pos_start_w: torch.Tensor = torch.zeros(2)
        self._vel_start_w: torch.Tensor = torch.zeros(2)

        self._net_init = False
        self._last_min_cost_idx = -1
        self._reach_final_goal = False

        self._goal_pose = torch.tensor([0.0, 0.0])
        self._goal_pose_valid = False
        self._vlm_goal = None
        self._vlm_goal_stamp = 0.0
        self._vlm_veto_indices: set[int] = set()
        self._vlm_veto_stamp = 0.0
        self._vlm_penalty: np.ndarray | None = None
        self._vlm_penalty_stamp = 0.0

        self._handle_topics()
        self._handle_point_cloud()

    def _handle_topics(self):
        # TODO: Check different QoS profiles
        self._depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image",
            self._depth_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._depth_nn_pub = self.create_publisher(
            Image, "/visualize/depth_nn", rclpy.qos.qos_profile_system_default
        )
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, rclpy.qos.qos_profile_sensor_data
        )
        self._goal_pose_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self._goal_pose_callback,
            rclpy.qos.qos_profile_parameters,
        )
        self._vlm_goal_sub = self.create_subscription(
            PoseStamped,
            "/vlm_goal",
            self._vlm_goal_callback,
            rclpy.qos.qos_profile_parameters,
        )
        self._vlm_veto_sub = self.create_subscription(
            Int32MultiArray,
            "/vlm_veto",
            self._vlm_veto_callback,
            rclpy.qos.qos_profile_parameters,
        )
        self._vlm_penalty_sub = self.create_subscription(
            Float32MultiArray,
            "/vlm_penalty",
            self._vlm_penalty_callback,
            rclpy.qos.qos_profile_parameters,
        )
        self._goal_local_pub = self.create_publisher(
            PointStamped, "/visualize/goal_local", rclpy.qos.qos_profile_system_default
        )
        self._net_timer = self.create_timer(1 / nn_frequncy, self._net_callback)
        self._goal_local_point = PointStamped()
        self._pos_end_pub_list = [
            self.create_publisher(
                PointStamped,
                f"/visualize/pos_end_{i}",
                rclpy.qos.qos_profile_system_default,
            )
            for i in range(candidate_num)
        ]
        self._path_min_cost_pub = self.create_publisher(
            Path, "/visualize/path_min_cost", rclpy.qos.qos_profile_system_default
        )
        self._end_state_pub = self.create_publisher(
            Odometry, "/end_state", rclpy.qos.qos_profile_system_default
        )
        self._path_pub_list = [
            self.create_publisher(
                Path, f"/visualize/path_{i}", rclpy.qos.qos_profile_system_default
            )
            for i in range(candidate_num)
        ]

    def _handle_point_cloud(self):
        try:
            import open3d as o3d
        except ImportError:
            self.get_logger().error(
                "Open3D is not installed, point cloud will not be published"
            )
            return
        self._terrain_pub = self.create_publisher(
            PointCloud2, "/visualize/terrain", rclpy.qos.qos_profile_system_default
        )
        self._tree_pub = self.create_publisher(
            PointCloud2, "/visualize/tree", rclpy.qos.qos_profile_system_default
        )
        terrain_pt = o3d.io.read_point_cloud("terrain.ply")
        tree_pt = o3d.io.read_point_cloud("tree.ply")
        terrain_pt_subsample = terrain_pt.voxel_down_sample(voxel_size=0.2)
        tree_pt_subsample = tree_pt.voxel_down_sample(voxel_size=0.2)

        header = Header()
        header.frame_id = "map"

        def point_cloud_callback():
            if self._state_valid:
                current_pos = self._pos_start_w.cpu().numpy()
                half_size = 20
                aabb = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=[
                        current_pos[0] - half_size,
                        current_pos[1] - half_size,
                        -half_size,
                    ],
                    max_bound=[
                        current_pos[0] + half_size,
                        current_pos[1] + half_size,
                        half_size,
                    ],
                )
                terrain_pt_subsample_crop = terrain_pt_subsample.crop(aabb)
                tree_pt_subsample_crop = tree_pt_subsample.crop(aabb)
            else:
                terrain_pt_subsample_crop = terrain_pt_subsample
                tree_pt_subsample_crop = tree_pt_subsample

            _terrain_pt_msg = create_cloud_xyz32(
                header, np.asarray(terrain_pt_subsample_crop.points)
            )
            _tree_pt_msg = create_cloud_xyz32(
                header, np.asarray(tree_pt_subsample_crop.points)
            )
            self._terrain_pub.publish(_terrain_pt_msg)
            self._tree_pub.publish(_tree_pt_msg)

        self._point_cloud_timer = self.create_timer(1.0, point_cloud_callback)

    def _vlm_goal_callback(self, msg: PoseStamped):
        """方案 A：VLM 语义安全目标点（map 系），2s 超时回退由 _odom_callback 兜底"""
        try:
            tf = self._tf_buffer.lookup_transform(
                "map", msg.header.frame_id, rclpy.time.Time()
            )
            msg = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)
        except Exception:
            pass  # 已是 map 系则直接用
        self._vlm_goal = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y], dtype=torch.float32
        )
        self._vlm_goal_stamp = time.time()

    def _vlm_veto_callback(self, msg: Int32MultiArray):
        """VLM 语义否决掩码：被否决的候选轨迹序号，超时由 _net_callback 兜底"""
        self._vlm_veto_indices = {int(i) for i in msg.data}
        self._vlm_veto_stamp = time.time()

    def _vlm_penalty_callback(self, msg: Float32MultiArray):
        """M3 正向融合：每条候选的语义代价加成（与候选序号对齐），超时由 _net_callback 兜底"""
        arr = np.asarray(msg.data, dtype=np.float64)
        if arr.shape[0] == candidate_num:
            self._vlm_penalty = arr
            self._vlm_penalty_stamp = time.time()

    def _goal_pose_callback(self, goal_pose_msg: PoseStamped):
        try:
            tf = self._tf_buffer.lookup_transform(
                "map", goal_pose_msg.header.frame_id, rclpy.time.Time()
            )
            goal_pose_msg = tf2_geometry_msgs.do_transform_pose_stamped(
                goal_pose_msg, tf
            )
            goal_pose = torch.tensor(
                [goal_pose_msg.pose.position.x, goal_pose_msg.pose.position.y],
                dtype=torch.float32,
            )
        except Exception as e:
            self.get_logger().error(f"Error transforming goal pose: {e}")
            goal_pose = torch.tensor([0.0, 0.0], dtype=torch.float32)

        self._goal_pose = goal_pose
        self._goal_pose_valid = True
        self.get_logger().info(
            self.pretty_print(goal_pose_msg.pose.position, "Goal pose")
        )

    @torch.inference_mode()
    def _depth_callback(self, image: Image):
        start_time = time.time()
        self._image_stamp = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
        depth_map = self._cv_bridge.imgmsg_to_cv2(image).astype(np.float32)
        depth_map = cv2.resize(
            depth_map, image_size, interpolation=cv2.INTER_NEAREST_EXACT
        )
        match image.encoding.upper():
            case "16UC1":
                invalid_mask = depth_map == 0
                depth_map[depth_map == 65535] = max_depth * 1e3
                depth_map /= max_depth * 1e3
            case "32FC1":
                invalid_mask = np.isnan(depth_map) | np.isneginf(depth_map)
                depth_map[np.isposinf(depth_map)] = 20.0
                depth_map /= max_depth
            case _:
                self.get_logger().error(f"Unsupported encoding:{image.encoding}")
                return
        depth_map = cv2.inpaint(
            depth_map, invalid_mask.astype(np.uint8), 3, cv2.INPAINT_NS
        )

        distance_map = self._transform_depth_to_distance(depth_map)

        distance_map_msg = self._cv_bridge.cv2_to_imgmsg(distance_map)
        distance_map_msg.header = image.header
        self._depth_nn_pub.publish(distance_map_msg)

        distance_map = np.expand_dims(distance_map, axis=-1)

        image_tensor = torch.from_numpy(distance_map)

        # (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        self._image = image_tensor
        if not self._image_valid:
            self._image_valid = True
        end_time = time.time()
        self.get_logger().debug(
            f"Depth callback time: {(end_time - start_time) * 1000:.2f} ms"
        )
        if end_time - start_time > 1 / 30:
            self.get_logger().error(
                f"Depth callback time: {(end_time - start_time) * 1000:.2f} ms"
            )

    @staticmethod
    def _transform_depth_to_distance(depth_map: np.ndarray):
        _height, width = depth_map.shape
        fov_rad = np.pi / 2
        half_width = width / 2
        x_indices = np.arange(width) - half_width
        angles = np.arctan(x_indices / half_width * np.tan(fov_rad / 2))
        distance_map: np.ndarray = depth_map / np.cos(angles).astype(np.float32)
        return distance_map

    @torch.inference_mode()
    def _odom_callback(self, odom: Odometry):
        start_time = time.time()
        quat = odom.pose.pose.orientation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        yaw_w = rot.as_euler("xyz", degrees=False)[2]

        pos_start_w = odom.pose.pose.position
        vel_start_w = odom.twist.twist.linear
        vel_start_w.x /= vel_out_multiplier
        vel_start_w.y /= vel_out_multiplier

        yaw_w: torch.Tensor = torch.tensor(yaw_w)
        pos_start_w = torch.tensor([pos_start_w.x, pos_start_w.y])
        vel_start_w = torch.tensor([vel_start_w.x, vel_start_w.y])

        goal_tolerance = 4.0
        diff_dir = self._goal_pose - pos_start_w
        self._reach_final_goal = torch.norm(diff_dir) <= goal_tolerance

        goal_local_distance_norm = min(7, torch.norm(diff_dir).cpu().numpy())
        diff_dir = diff_dir / torch.norm(diff_dir)

        # 方案 A：VLM 语义目标优先（软约束，2s 超时回退纯 YOPO）
        vlm_active = self._vlm_goal is not None and time.time() - self._vlm_goal_stamp < 2.0
        if vlm_active:
            diff_vlm = self._vlm_goal - pos_start_w
            if torch.norm(diff_vlm) > 1e-6:
                vlm_dist = min(7, torch.norm(diff_vlm).cpu().numpy())
                goal_local_w = pos_start_w + diff_vlm / torch.norm(diff_vlm) * vlm_dist
            else:
                goal_local_w = pos_start_w + diff_dir * goal_local_distance_norm
        else:
            goal_local_w = pos_start_w + diff_dir * goal_local_distance_norm

        goal_local_w_tmp = goal_local_w.cpu().numpy()
        self._goal_local_point.header = odom.header
        self._goal_local_point.point.x = float(goal_local_w_tmp[0])
        self._goal_local_point.point.y = float(goal_local_w_tmp[1])
        self._goal_local_point.point.z = odom.pose.pose.position.z + 1.0
        self._goal_local_point.point.z = odom.pose.pose.position.z

        self._goal_local_pub.publish(self._goal_local_point)

        self._coord_transform.construct_primitive_frame(yaw_w, pos_start_w)
        T_wp = self._coord_transform.T_wp
        T_pw = self._coord_transform.T_pw
        state: torch.Tensor = self._state_transform(T_wp, vel_start_w, goal_local_w)
        self._T_pw = T_pw
        self._state = state
        if not self._state_valid:
            self._state_valid = True
        self._pos_start_w = pos_start_w
        self._vel_start_w = vel_start_w
        end_time = time.time()
        self.get_logger().debug(
            f"Odom callback time: {(end_time - start_time) * 1000:.2f} ms"
        )

    @torch.inference_mode()
    def _net_callback(self):
        start_time = time.time()
        if not (self._state_valid and self._image_valid and self._goal_pose_valid):
            self.get_logger().info(
                f"Waiting for state, image and goal..., state: {self._state_valid}, image: {self._image_valid}, goal: {self._goal_pose_valid}",
                throttle_duration_sec=1,
            )
            return
        if self._reach_final_goal:
            self.get_logger().info("Reached final goal", throttle_duration_sec=1)
            return
        if self._image_stamp == self._last_image_stamp:
            self.get_logger().warn(
                f"No image recieved, last image stamp: {self._image_stamp}"
            )
        elif self._image_stamp - self._last_image_stamp > 2 / nn_frequncy:
            self.get_logger().warn(
                f"Image delay: {self._image_stamp - self._last_image_stamp:.2f} s"
            )
        self._last_image_stamp = self._image_stamp
        state = self._state.to(device).unsqueeze(0)
        T_pw = self._T_pw
        pos_start_w = self._pos_start_w.cpu().numpy()
        vel_start_w = self._vel_start_w.cpu().numpy()
        image = self._image.to(device).unsqueeze(0)

        start_time_cuda = torch.cuda.Event(enable_timing=True)
        end_time_cuda = torch.cuda.Event(enable_timing=True)
        start_time_cuda.record()
        output: torch.Tensor = self._net(image, state)
        end_time_cuda.record()
        torch.cuda.synchronize()
        time_elapsed = start_time_cuda.elapsed_time(end_time_cuda)
        self.get_logger().info(f"NN inference time: {time_elapsed:.2f} ms")
        if time_elapsed > 1000 / nn_frequncy:
            self.get_logger().error(f"NN inference time: {time_elapsed:.2f} ms")
        output = output.squeeze(0).cpu()

        pos_end_w, vel_end_w, cost = self._output_transform.inverse_transform(
            output, T_pw
        )
        pos_end_w = pos_end_w.squeeze().numpy().T
        vel_end_w = vel_end_w.squeeze().numpy().T * vel_out_multiplier
        cost = cost.squeeze().numpy()

        vel_start_w *= vel_out_multiplier

        self.get_logger().info(f"Path costs: {cost}")

        # VLM 语义否决：被否决候选的代价置 +inf 后再 argmin；超时未更新回退纯 YOPO
        cost_sel = cost.copy()
        if time.time() - self._vlm_veto_stamp < vlm_veto_timeout:
            for idx in self._vlm_veto_indices:
                if 0 <= idx < len(cost_sel):
                    cost_sel[idx] = np.inf

        # M3 正向融合：语义代价加成（软约束，不否决，与否决可叠加）
        if (
            self._vlm_penalty is not None
            and time.time() - self._vlm_penalty_stamp < vlm_veto_timeout
        ):
            cost_sel = cost_sel + self._vlm_penalty

        if np.all(np.isinf(cost_sel)):
            # 全部候选被否决：发布零速末端状态让 MPC 停车，等待 VLM 更新
            self.get_logger().warn(
                f"VLM vetoed all {len(cost_sel)} candidates, stopping"
            )
            stop_msg = Odometry()
            stop_msg.header = self._goal_local_point.header
            stop_msg.pose.pose.position.x = float(pos_start_w[0])
            stop_msg.pose.pose.position.y = float(pos_start_w[1])
            stop_msg.pose.pose.position.z = self._goal_local_point.point.z
            self._end_state_pub.publish(stop_msg)
            # 候选轨迹照常发布（/visualize/path_i 是否决节点的输入，不能断）
            self._path_visualize(
                pos_start_w, pos_end_w, vel_start_w, vel_end_w,
                self._last_min_cost_idx if self._net_init else int(np.argmin(cost)),
            )
        else:
            min_cost_idx: int = int(np.argmin(cost_sel))

            if not self._net_init:
                self._last_min_cost_idx = min_cost_idx

            if min_cost_idx != self._last_min_cost_idx:
                last_sel_cost = cost_sel[self._last_min_cost_idx]
                if np.isinf(last_sel_cost):
                    # 上次选择的轨迹已被 VLM 否决，强制切换
                    self._last_min_cost_idx = min_cost_idx
                elif (
                    np.abs(cost_sel[min_cost_idx] - last_sel_cost)
                    / max(last_sel_cost, 1e-6)
                    < min_cost_jump_threshold
                ):
                    min_cost_idx = self._last_min_cost_idx
                else:
                    self._last_min_cost_idx = min_cost_idx

            pos_end_point = Point()
            pos_end_point.x = float(pos_end_w[min_cost_idx, 0])
            pos_end_point.y = float(pos_end_w[min_cost_idx, 1])
            pos_end_point.z = self._goal_local_point.point.z
            end_state_msg = Odometry()
            end_state_msg.header = self._goal_local_point.header
            end_state_msg.pose.pose.position = pos_end_point
            end_state_msg.twist.twist.linear.x = float(vel_end_w[min_cost_idx, 0])
            end_state_msg.twist.twist.linear.y = float(vel_end_w[min_cost_idx, 1])
            self._end_state_pub.publish(end_state_msg)

            self._path_visualize(
                pos_start_w, pos_end_w, vel_start_w, vel_end_w, min_cost_idx
            )

        self._net_init = True
        end_time = time.time()
        self.get_logger().debug(
            f"Net callback time: {(end_time - start_time) * 1000:.2f} ms"
        )
        if end_time - start_time > 1 / nn_frequncy:
            self.get_logger().error(
                f"Net callback time: {(end_time - start_time) * 1000:.2f} ms"
            )

    def _path_visualize(
        self,
        pos_start_w: np.ndarray,
        pos_end_w: np.ndarray,
        vel_start_w: np.ndarray,
        vel_end_w: np.ndarray,
        min_cost_idx: int,
    ):
        Ts = 0.4 / vel_out_multiplier
        N = int(te / Ts) + 1
        t_eval = np.linspace(0, te, N)
        pos_spline_x_list, _, pos_spline_y_list, _ = generate_splines(
            candidate_num, pos_start_w, vel_start_w, pos_end_w, vel_end_w, t_eval
        )
        for i in range(candidate_num):
            pos_end = PointStamped()
            pos_end.header = self._goal_local_point.header
            pos_end.point.x = float(pos_end_w[i, 0])
            pos_end.point.y = float(pos_end_w[i, 1])
            pos_end.point.z = self._goal_local_point.point.z
            self._pos_end_pub_list[i].publish(pos_end)

            p_eval_x = pos_spline_x_list[i](t_eval)
            p_eval_y = pos_spline_y_list[i](t_eval)
            path = Path()
            path.header = self._goal_local_point.header
            poses = []
            for t, x, y in zip(t_eval, p_eval_x, p_eval_y, strict=True):
                pose = PoseStamped()
                pose.header = Header()
                pose.header.frame_id = self._goal_local_point.header.frame_id
                pose.header.stamp.sec = int(t)
                pose.header.stamp.nanosec = int((t - int(t)) * 1e9)
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = self._goal_local_point.point.z
                poses.append(pose)
            path.poses = poses
            self._path_pub_list[i].publish(path)
            if i == min_cost_idx:
                self._path_min_cost_pub.publish(path)

    def pretty_print(self, point, name):
        return f"{name}: {point.x:.2f} {point.y:.2f} {point.z:.2f}"


def main():
    rclpy.init()
    topic_handler = YOPONode()
    rclpy.spin(topic_handler)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
