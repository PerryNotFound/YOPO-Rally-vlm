# YOPO-Rally

基于模仿学习的单阶段越野地形规划器，支持从仿真到实车的零样本迁移（Sim-to-Real）。

视频演示：[YouTube](https://youtu.be/dyoufaKgVa0) · [Bilibili](https://www.bilibili.com/video/BV1D1V7zQEWr)

[English](README.md) | [中文](README_zh.md)

## 系统概述

![系统概述](.media/system-overview.svg)

规划器以深度图和车辆状态为输入，直接输出多条候选轨迹及代价评分，由 MPC 局部规划器跟踪最优轨迹。训练完全在仿真环境（YOPO-Sim）中进行，通过地形代价图生成专家轨迹标签，实现从仿真到实车的零样本迁移。

详细架构说明见 [系统概述](docs/01-系统概述.md)。

## 仿真器

使用 [YOPO-Sim](https://github.com/TJU-Aerial-Robotics/YOPO-Sim.git) 作为仿真与数据采集环境。

![YOPO-Sim](.media/yopo-sim.jpg)

## 模仿学习

### 代价图生成

地形点云经 CloudCompare 进行地形类型分析（TTA），提取倾斜度与粗糙度特征，生成 GeoTIFF 格式的地形通行代价图。

https://github.com/user-attachments/assets/9114d64a-cec4-42c8-93c3-7d9c15ee0e10

### 数据采集

在仿真器中录制深度图、车辆位置与速度数据。

![数据采集](.media/data-acquisition.jpg)

### 轨迹优化

对每个参考起点，使用 CasADi 在扇形约束下求解多条候选 Hermite 曲线轨迹，并从代价图中查询地形代价，生成专家轨迹标签。

https://github.com/user-attachments/assets/d9902d38-5c5b-4b8a-aec8-42a6decb11fb

## 推理

规划器以深度图、速度和目标向量为输入，输出带代价评分的候选轨迹集合，MPC 节点跟踪最优轨迹。

https://github.com/user-attachments/assets/a4583249-a9d8-4ef8-8d7d-87dd9a78e298

## 快速开始

**环境要求：** ROS2 Humble · Python 3.10 · `uv` · CloudCompare 2.13+

```bash
# 1. 克隆仓库
git clone https://github.com/TJU-Aerial-Robotics/YOPO-Rally.git --recursive

# 2. 创建虚拟环境并安装依赖
uv venv --system-site-packages
uv sync

# 3. 编译 ROS2 包（禁止使用 --symlink-install）
source .venv/bin/activate
colcon build && source install/setup.bash

# 4. 启动推理节点（需先运行 YOPO-Sim）
ros2 launch diablo_ros_wrapper diablo_sim.launch.py
```

启动后，仿真器加载`Assets\Scenes\EvaluationScene.unity`场景，在仿真器中点击 **Connect**，再在发布 `/initialpose (geometry_msgs/msg/PoseWithCovarianceStamped)` 话题即可开始规划。

*注：运行前请从 **Release** 页面下载预训练权重文件，并将其放置在 `models/` 目录下。*

完整安装步骤见 [安装与环境](docs/02-安装与环境.md)。

## 训练流水线

1. 在 YOPO-Sim 中采集深度图与位姿数据
2. 使用 CloudCompare 将点云处理为地形代价图（GeoTIFF）
3. 运行 `generate_data.py` 生成专家轨迹标签
4. 运行 `train_network.py` 训练网络

详细步骤见 [数据训练流水线](docs/04-数据训练流水线.md)。

## 实车部署

详细说明见 [实车部署](docs/05-实车部署.md)。

## 文档

| 文档 | 内容 |
|------|------|
| [系统概述](docs/01-系统概述.md) | 架构、目录结构、网络 I/O 格式 |
| [安装与环境](docs/02-安装与环境.md) | 完整安装步骤 |
| [仿真推理](docs/03-仿真推理.md) | 与 YOPO-Sim 联调运行 |
| [数据训练流水线](docs/04-数据训练流水线.md) | 数据采集 → 代价图 → 轨迹优化 → 训练 |
| [实车部署](docs/05-实车部署.md) | ROS 节点、传感器配置 |
| [常见问题](docs/06-常见问题.md) | FAQ |

---

## 本 fork 的改动（VLM 语义扩展）

本 fork 为配套的 System-2 仓库 [yopo-vlm-sys2](https://github.com/PerryNotFound/yopo-vlm-sys2)（VLM 水面/泥地/沙地避障）增加了语义危险接口：

- `yopo_node.py`：订阅 `/vlm_veto`（硬否决：候选代价置 +inf）、`/vlm_penalty`（软融合进代价函数）、`/vlm_goal`（恢复重定向）；2 秒软超时自动回退为纯 YOPO。
- `mpc_node.py` / `yopo_node.py`：输出速度倍率 `vel_out_multiplier = 0.8`。
- 预训练权重 `models/yopo_net.pth`（约 53 MB）**未纳入版本管理**——请从上游 release 获取后放入 `models/`。

上游仓库：https://github.com/TJU-Aerial-Robotics/YOPO-Rally
