# Real-World Deployment

## Hardware Configuration

| Device | Model | Description |
|------|------|------|
| Robot Chassis | Diablo | Two-wheeled self-balancing robot, controlled via the `diablo_ctrl` ROS2 package. |
| Depth Camera | OAK-D Pro | Provides stereo depth maps, topic: `/oak/stereo/image_raw`. |
| Visual Odometry | Intel T265 | Provides 6-DOF pose and velocity, topic: `/camera/pose/sample`. |
| Computing Platform | Jetson Xavier NX (Recommended) or a laptop with CUDA | Runs the inference node. |

## Launch Procedure

### Full Launch Command

```bash
ros2 launch diablo_ros_wrapper diablo_depolyment.launch.py
```

**Optional Parameters:**

| Parameter | Default Value | Description |
|------|--------|------|
| `gui` | `true` | Whether to enable RViz2 visualization |
| `ctrl` | `false` | Whether to launch the chassis driver node (`diablo_ctrl`) |
| `depthai` | `false` | Whether to launch the OAK-D camera driver |
| `vins` | `false` | Whether to launch VINS-Fusion odometry (alternative to T265) |

Typical launch command (enabling both chassis driver and camera):

```bash
ros2 launch diablo_ros_wrapper diablo_depolyment.launch.py ctrl:=true depthai:=true
```

## Node Architecture

```
T265 → /camera/pose/sample ──┐
                              ├─► DepolymentMsgConvertor ──► /odom
OAK-D → /oak/stereo/image_raw ┘
                │
                ▼
          yopo_node (YOPO Inference)
                │
                ▼
          mpc_node (MPC Tracking)
                │
          /cmd_vel ──► DepolymentMsgConvertor ──► /diablo/MotionCmd ──► Chassis
```

## Message Converter Node

Node file: `src/ros_pkgs/diablo_ros_wrapper/diablo_ros_wrapper/depolyment_msg_convertor.py`

This node handles two tasks:

**Odometry Conversion**: Converts the pose published by T265 in its own coordinate frame into the standard `base_link` frame on the `/odom` topic, for use by the YOPO and MPC nodes.

**Control Command Conversion**: Converts `geometry_msgs/Twist` (`/cmd_vel`) issued by the planner into the `motion_msgs/MotionCtrl` format required by the chassis driver, publishing it to `/diablo/MotionCmd`.

Simultaneously, this node broadcasts the static TF of the sensors (`map → odom → base_link → {t265, oak-d-base-frame}`) through hardcoding. If the sensor installation position changes, the corresponding translation vectors must be modified.

## Waypoint Tracking Mode

Node file: `src/ros_pkgs/diablo_ros_wrapper/diablo_ros_wrapper/waypoint_tracker_node.py`

This node can sequentially publish goal points from a preset list, eliminating the need to manually click points in RViz2. Waypoints are configured in the `WAYPOINTS` list at the top of the node file in the `[x, y, yaw]` format.

## MPC Node Description

The YOPO node outputs the endpoint state (position + velocity) of the optimal trajectory. Bound by this constraint, the MPC node solves a sequence of low-level velocity commands, publishing `/cmd_vel` at 20 Hz, with a planning horizon of 10 steps (step size 0.25 s).

Velocity range constraints: linear velocity ±2.0 m/s, angular velocity ±π rad/s. Output velocities can be scaled via the `vel_out_multiplier` parameter (default 1.0) at the top of `mpc_node.py`.