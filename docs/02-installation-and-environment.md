# Installation and Environment Configuration

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- CUDA 12.4 (Optional, if using GPU)
- CloudCompare 2.13+ (For point cloud processing during training)

It is recommended to install CloudCompare via Flathub, see [Flathub Page](https://flathub.org/apps/org.cloudcompare.CloudCompare).

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/TJU-Aerial-Robotics/YOPO-Rally.git --recursive
```

### 2. Install uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment

The virtual environment must include the system `site-packages` to access ROS2 Python packages:

```bash
SYSTEM_PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1-2)
uv python pin $SYSTEM_PYTHON_VERSION
uv venv --system-site-packages
```

Alternatively, you can use `python -m venv .venv --system-site-packages`.

### 4. Install Python Dependencies

```bash
uv sync
```

This command will automatically install PyTorch (cu124), open3d, casadi, and all required dependencies of the `diablo-python` package itself.

### 5. Build ROS2 Packages

```bash
source .venv/bin/activate
colcon build
source install/setup.bash
```

> Note: Do NOT use `--symlink-install` here, otherwise ROS2 nodes will not be able to invoke the Python interpreter from the virtual environment.

### 6. Place Model Files

Download pre-trained `.pth` weight files from the Release section, place them under the `models/` directory at the project root, and then set `load_model_path` pointing to the respective file in `src/diablo_project/configs.py`.

## Running Before Every Session

Always run the following two lines upon opening a new terminal session:

```bash
source .venv/bin/activate
source install/setup.bash
```