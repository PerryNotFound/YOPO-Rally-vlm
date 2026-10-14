# YOPO-Rally

A single-stage off-road terrain planner based on imitation learning, supporting Zero-Shot Sim-to-Real transfer.

Video Demos: [YouTube](https://youtu.be/dyoufaKgVa0) · [Bilibili](https://www.bilibili.com/video/BV1D1V7zQEWr)

[English](README.md) | [中文](README_zh.md)

## System Overview

![System Overview](.media/system-overview.svg)

The planner takes the depth map and vehicle state as inputs, and directly outputs multiple candidate trajectories along with their cost scores. The MPC local planner then tracks the optimal trajectory. Training is conducted entirely in the simulation environment (YOPO-Sim), where expert trajectory labels are generated through terrain cost maps, achieving zero-shot transfer from simulation to real world.

For detailed architecture, see [System Overview](docs/01-system-overview.md).

## Simulator

We use [YOPO-Sim](https://github.com/TJU-Aerial-Robotics/YOPO-Sim.git) as the simulation and data collection environment.

![YOPO-Sim](.media/yopo-sim.jpg)

## Imitation Learning

### Cost Map Generation

The terrain point cloud is analyzed for Terrain Type Analysis (TTA) using CloudCompare, extracting slope and roughness features to generate a terrain traversability cost map in GeoTIFF format.

https://github.com/user-attachments/assets/9114d64a-cec4-42c8-93c3-7d9c15ee0e10

### Data Acquisition

Record depth maps, vehicle position, and velocity data in the simulator.

![Data Acquisition](.media/data-acquisition.jpg)

### Trajectory Optimization

For each reference starting point, CasADi is used to solve multiple candidate Hermite curve trajectories under sector constraints. Terrain costs are queried from the cost map to generate expert trajectory labels.

https://github.com/user-attachments/assets/d9902d38-5c5b-4b8a-aec8-42a6decb11fb

## Inference

The planner takes the depth map, velocity, and goal vector as inputs, outputting a set of candidate trajectories with cost scores. The MPC node tracks the optimal trajectory.

https://github.com/user-attachments/assets/a4583249-a9d8-4ef8-8d7d-87dd9a78e298

## Quick Start

**Requirements:** ROS2 Humble · Python 3.10 · `uv` · CloudCompare 2.13+

```bash
# 1. Clone the repository
git clone https://github.com/TJU-Aerial-Robotics/YOPO-Rally.git --recursive

# 2. Create virtual environment and install dependencies
uv venv --system-site-packages
uv sync

# 3. Build ROS2 packages (DO NOT use --symlink-install)
source .venv/bin/activate
colcon build && source install/setup.bash

# 4. Launch inference node (requires YOPO-Sim running)
ros2 launch diablo_ros_wrapper diablo_sim.launch.py
```

After launching, load the `Assets\Scenes\EvaluationScene.unity` scene in the simulator, click **Connect** in the simulator, and then trigger planning by publishing `/initialpose (geometry_msgs/msg/PoseWithCovarianceStamped)` topic.

*Note: Please download the pre-trained weights from the **Release** page and place them in the `models/` directory before running.*

For full installation steps, see [Installation and Environment](docs/02-installation-and-environment.md).

## Training Pipeline

1. Collect depth maps and pose data in YOPO-Sim
2. Process the point cloud into a terrain cost map (GeoTIFF) using CloudCompare
3. Run `generate_data.py` to generate expert trajectory labels
4. Run `train_network.py` to train the network

For detailed steps, see [Data Training Pipeline](docs/04-data-training-pipeline.md).

## Real-World Deployment

For detailed instructions, see [Real-World Deployment](docs/05-real-world-deployment.md).

## Documentation

| Document | Content |
|------|------|
| [System Overview](docs/01-system-overview.md) | Architecture, directory structure, network I/O formats |
| [Installation & Environment](docs/02-installation-and-environment.md) | Complete installation steps |
| [Simulation & Inference](docs/03-simulation-and-inference.md) | Running jointly with YOPO-Sim |
| [Data Training Pipeline](docs/04-data-training-pipeline.md) | Data acquisition → Cost map → Trajectory optimization → Training |
| [Real-World Deployment](docs/05-real-world-deployment.md) | ROS nodes, sensor configuration |
| [FAQ](docs/06-faq.md) | Frequently Asked Questions |
