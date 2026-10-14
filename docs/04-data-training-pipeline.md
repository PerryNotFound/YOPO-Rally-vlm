# Data Training Pipeline

The process from recording raw data in the simulator to training a model consists of the following steps:

```
Step 1  Record data in YOPO-Sim
  └─► Scene_i/data.toml + Depth maps + terrain.ply + tree.ply

Step 2  Point cloud processing via CloudCompare
  └─► Textures/map*.tif (GeoTIFF cost map)

Step 3  Generate expert trajectory labels
  └─► Scene_i/data_opt.toml

Step 4  Train the network
  └─► models/<name>.pth

Step 5  Evaluate
  └─► Loss component statistics
```

---

## Quick Start

The following is the minimal workflow from zero to completing the training, suitable for first-time setup or quick reproduction.

### Prerequisites

| Dependency | Description |
|--------|------|
| YOPO-Sim | Simulator installed, see [02-installation-and-environment.md](02-installation-and-environment.md) |
| CloudCompare 2.13+ | For point cloud processing, must be added to system PATH |
| Python Dependencies | Installed via `uv sync` |

### I. Configure Paths

Edit `src/diablo_project/configs.py` and modify the following three variables according to your actual paths:

```python
# YOPO-Sim installation directory, containing TrainingData/ and TestingData/
YOPO_SIM_Path = "/path/to/YOPO-Sim"

# Directory for outputting training weights
trained_model_folder = "models/temp"

# Model path to load for inference (used for evaluation and deployment)
load_model_path = "models/yopo_net.pth"
```

### II. Record Data in the Simulator

1. Open YOPO-Sim and switch to the scene: `Assets/Scenes/DataGenerationScene`
2. Adjust scene parameters (terrain type, obstacle density, etc.)
3. Click **Generate Data** to start recording
4. After recording is finished, several `Scene_i/` folders are generated under `$YOPO_SIM_Path/TrainingData/`.

### III. Generate Expert Trajectory Labels

```bash
# Execute in the project root directory
python src/diablo_project/generate_data.py
```

This script automatically handles the CloudCompare point cloud processing (Step 2) and trajectory optimization label generation (Step 3). The time required depends on the number of scenes. Multi-processing is recommended and enabled by default for multiple scenes.

### IV. Train the Model

```bash
python src/diablo_project/train_network.py
```

The training process is recorded via TensorBoard to view the loss curve in real-time:

```bash
tensorboard --logdir models/temp
```

Once training is complete, weight files are saved into the `trained_model_folder` directory (default `models/temp/`).

### V. Evaluate the Model

```bash
python src/diablo_project/eval_network.py
```

Outputs loss statistics for position, velocity, and cost components, used to compare the performance of different training configurations.

> **Tip**: After training is complete, update the generated `.pth` file path into `load_model_path` in `configs.py` to use the new model for simulation inference ([03-simulation-and-inference.md](03-simulation-and-inference.md)) or real-world deployment ([05-real-world-deployment.md](05-real-world-deployment.md)).

---

## Step 1: Record Data

In YOPO-Sim, switch to the data collection scene (`Assets\Scenes\DataGenerationScene`), adjust the parameters, and click **Generate Data** to start recording. After completion, several `Scene_i/` folders are generated under `TrainingData/`, each containing:

- `data.toml`: Frame sequence recording timestamps, robot position/velocity, goal position, and corresponding depth map filenames.
- Depth map files: Single-channel floating-point format (16UC1 or 32FC1), size 160×32.
- `terrain.ply`, `tree.ply`: Terrain and vegetation point clouds used for next step processing.

---

## Step 2: CloudCompare Point Cloud Processing

The script `src/diablo_project/generate_data.py` internally invokes the CloudCompare CLI to merge `terrain.ply` and `tree.ply` and perform Terrain Type Analysis (TTA):

1. Calculates normals, extracting inclination (DIP) and roughness features.
2. Rasterizes the results into GeoTIFF files (`Textures/map*.tif`) with a resolution of 0.3 m.

The cost map fuses both bands: regions exceeding 30° dip or 0.1 roughness are assigned high costs, while a Signed Distance Field (SDF) is overlaid for obstacle inflation, bringing the final cost range to [0, 1].

> If you need to run this step manually, you can process point clouds through the CloudCompare 2.13+ GUI or CLI.

---

## Step 3: Generate Expert Trajectory Labels

Run the following command to generate optimization labels:

```bash
python src/diablo_project/generate_data.py
```

The script reads starting and goal points from `data.toml` and relies on CasADi (IPOPT solver) to generate multiple candidate Hermite curve trajectories for each starting point. It then queries the GeoTIFF cost map for the terrain cost of each trajectory, writing the results into `data_opt.toml`.

Each data point contains 5 candidate trajectories and their respective costs, where the 5 candidate exits are evenly distributed across a sectoral angle range. After generation, intermediate cache files can be found in the directory set by `trained_model_folder` in `configs.py`.

---

## Step 4: Training

After modifying the path variables in `src/diablo_project/configs.py`, run:

```bash
python src/diablo_project/train_network.py
```

Main hyper-parameters:

| Parameter | Default | Description |
|------|--------|------|
| `batch_size` | 64 | Batch size |
| `epochs` | 20 | Number of training epochs |
| Learning Rate | 1e-3 | AdamW optimizer |
| Data Split | 90% / 10% | Training set / Validation set |

The loss function is MSE, calculated across 5 channels: position polar coordinates, velocity, and costs. The training process is recorded via TensorBoard, and weights are saved in the `trained_model_folder`.

---