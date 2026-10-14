# Simulation and Inference

## Overview

This project uses the [ROS-TCP-Endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint) to connect ROS2 with the Unity simulator, testing terrain planning within the simulation.

## Launch Procedure

### 1. Launch YOPO-Sim

Start YOPO-Sim and switch to the corresponding testing scene (`Assets\Scenes\EvaluationScene`).

### 2. Launch ROS2 Nodes

```bash
ros2 launch diablo_ros_wrapper diablo_sim.launch.py
```

`diablo_sim.launch.py` will sequentially launch three submodules:

| Submodule | Function |
|--------|------|
| `ros_tcp_endpoint` | Listens for Unity messages (default port 10000) and bridges them to ROS2 topics. |
| `diablo_algorithm` | Launches the YOPO inference node and the MPC node. |
| `diablo_visualize` | Launches RViz2 visualization (can be disabled via `gui:=false`). |

### 3. Connect in Unity

Click the **Connect** button in the simulator UI, entering the IP address and port (default 10000) of your ROS2 machine.

### 4. Start Planning

Use the **2D Pose Estimate** tool in RViz2 to click on a goal position. The MPC node will then start tracking the optimal trajectory output by YOPO.

## YOPO Inference Node

Node file: `src/ros_pkgs/diablo_ros_wrapper/diablo_ros_wrapper/yopo_node.py`

**Subscribed Topics:**

| Topic | Type | Description |
|------|------|------|
| `/camera/depth/image` | `sensor_msgs/Image` | Depth map, automatically resized to 160×32 |
| `/odom` | `nav_msgs/Odometry` | Position and velocity |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal position |

**Published Topics:**

| Topic | Type | Description |
|------|------|------|
| `/end_state` | `nav_msgs/Odometry` | Optimal trajectory endpoint (for MPC tracking) |
| `/visualize/depth_nn` | `sensor_msgs/Image` | Preprocessed depth map |
| `/visualize/path_0` to `/visualize/path_4` | `nav_msgs/Path` | 5 candidate trajectories (for visualization) |
| `/visualize/path_min_cost` | `nav_msgs/Path` | Lowest-cost trajectory |

## MPC Node

Node file: `src/ros_pkgs/diablo_ros_wrapper/diablo_ros_wrapper/mpc_node.py`

The MPC node runs at 20 Hz with a planning horizon of 10 steps (step size 0.25 s). It subscribes to `/end_state` and `/odom`, and publishes `/cmd_vel` velocity commands.

## Key Configuration Parameters

Training and model paths are centrally configured in `src/diablo_project/configs.py`:

| Parameter | Default Value | Description |
|------|--------|------|
| `YOPO_SIM_Path` | `/home/chy/Downloads/YOPO-Sim_Linux` | YOPO-Sim root directory |
| `load_model_path` | `models/yopo_net.pth` | Inference model path |
| `trained_model_folder` | `models/temp` | Training output directory |

YOPO node inference parameters (inference frequency, planning duration, number of candidates, etc.) are configured at the top of the node file.