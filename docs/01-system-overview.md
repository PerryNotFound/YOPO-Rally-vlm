# System Overview

## Functional Positioning

YOPO-Rally is a single-stage, off-road terrain planner based on imitation learning, supporting Zero-Shot Sim-to-Real transfer from a simulation environment (YOPO-Sim) to an actual robot. The planner takes depth maps and vehicle states as input, directly outputting multiple candidate trajectories and their corresponding cost scores. An MPC local planner then tracks the trajectory with the minimum cost.

## Overall Architecture

```
Unity Simulation / Real Robot
    │
    ▼
Depth Map (OAK-D / Unity) + Odometry (T265 / Unity) + Goal Point
    │
    ▼
yopo_node (YOPO Inference, 10 Hz)
  ├─ Input: Single-frame Depth Map + Current Velocity + Goal Vector
  └─ Output: 5 Candidate Trajectories + Costs
    │
    ▼
mpc_node (MPC Local Tracking, 20 Hz)
  └─ Output: /cmd_vel → Robot Chassis
```

## Directory Structure

```
.
├── docs/                               This documentation
├── models/                             Model weights (manually place here)
├── package/diablo_python/              Core Python package
│   └── src/diablo_python/
│       ├── network/                    Network definitions, datasets, trainers
│       ├── optimizer/                  MPC planner
│       ├── data_process/               Data processing (point cloud → cost map, trajectory optimization)
│       └── geotiff/                    GeoTIFF cost map tools
├── src/
│   ├── diablo_project/                 Training script entrypoints (generate_data.py, etc.)
│   └── ros_pkgs/
│       ├── diablo_ros_wrapper/         Inference node, MPC node, launch files
│       └── ROS-TCP-Endpoint/           Unity communication bridge
└── pyproject.toml
```

## Network Input/Output

| Item | Description |
|------|------|
| Depth Map Input | Single-frame depth map, shape 1×32×160, unit: meters |
| State Input | 4 Channels: `[vx, vy, goal_x, goal_y]`, all normalized to [-1, 1] |
| Trajectory Output | 5 Candidates, each containing position polar coordinates `[pe_n, pe_θ]` and velocity `[ve_x, ve_y]` (normalized) |
| Cost Output | Cost prediction for each of the 5 candidates, normalized to [0, 1] |

## Training Data Format

Training data is organized by scene, stored in the `TrainingData/` directory of YOPO-Sim:

```
TrainingData/
  Scene_0/
    terrain.ply, tree.ply   # Terrain and vegetation point clouds
    Textures/map*.tif       # GeoTIFF cost map output by CloudCompare
    data.toml               # Raw frame data (image paths, positions, velocities)
    data_opt.toml           # Optimized expert trajectory labels (5 candidates and their costs)
  Scene_1/
    ...
```

The cost map is stored in GeoTIFF format with a grid resolution of 0.3m, containing two bands: Dip and Roughness, which are fused to generate a terrain traversability cost in the range [0, 1].