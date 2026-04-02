# Gaussian Splatting Setup Guide

Step-by-step instructions for setting up 3 Gaussian Splatting repositories, each running in its own Docker container with GPU access.

---

## Prerequisites

Before starting, ensure the following host-level software is installed. If you are starting from a fresh Ubuntu 24.04 GCP instance, run every command below in order.

### Step 1: NVIDIA Driver

The standard `nvidia-driver-565` package fails DKMS on kernel 6.17. Use the open kernel modules from driver branch 590 instead.

```bash
# 1a. Add the NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 1b. Install the driver
sudo apt-get install -y nvidia-driver-590-open

# 1c. Load the kernel module
sudo modprobe nvidia

# 1d. Verify — you should see your GPU listed
nvidia-smi
```

### Step 2: CUDA Toolkit

```bash
# 2a. Install CUDA 12.6
sudo apt-get install -y cuda-toolkit-12-6

# 2b. Add to PATH (persists across logins)
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh

# 2c. Source it for the current session
source /etc/profile.d/cuda.sh

# 2d. Verify
nvcc --version
# Expected: Cuda compilation tools, release 12.6, V12.6.85
```

### Step 3: cuDNN

```bash
sudo apt-get install -y cudnn-cuda-12
```

### Step 4: Docker + NVIDIA Container Toolkit

```bash
# 4a. Install Docker Engine
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 4b. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4c. Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4d. Add your user to the docker group (log out and back in afterward)
sudo usermod -aG docker $USER

# 4e. Verify GPU access from inside a container
sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Step 5: Common Build Dependencies

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev \
    colmap imagemagick python3-pip
```

### Step 6: Miniconda (optional)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## Create the Project Directory

All three repositories and their Docker infrastructure live under a single project directory.

```bash
mkdir -p ~/Documents/gaussian-splatting-projects
cd ~/Documents/gaussian-splatting-projects
mkdir -p data output
```

The final structure will look like this:

```
gaussian-splatting-projects/
├── gaussian-splatting/       # Repo 1: Original 3DGS (cloned on host)
│   └── Dockerfile
├── gsplat.Dockerfile         # Repo 2: gsplat (cloned inside image)
├── 2dgs.Dockerfile           # Repo 3: 2DGS (cloned inside image)
├── docker-compose.yml        # Orchestrates all 3 services
├── create_subset.py          # Create evenly-spaced subsets or chunks of a COLMAP scene
├── flatten_scene.py          # Flatten deep image paths for 3DGS/2DGS compatibility
├── data/                     # Prepared datasets (subsets, chunks)
└── output/                   # Shared training output (mounted into all containers)
```

---

## Repository 1: Original 3D Gaussian Splatting (INRIA)

**Repo:** https://github.com/graphdeco-inria/gaussian-splatting
**Paper:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., SIGGRAPH 2023)

The original method. Represents scenes as millions of 3D Gaussians optimized from multi-view images. Supports real-time rendering at high quality.

### Step 1: Clone the repository

```bash
cd ~/Documents/gaussian-splatting-projects
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
```

### Step 2: Create the Dockerfile

Create the file `gaussian-splatting/Dockerfile` with the following contents:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev \
    colmap imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy the cloned repo into the image
COPY . /workspace/gaussian-splatting/

# Pin numpy <2.0 (PyTorch 2.0.1 is incompatible with numpy 2.x)
RUN pip install "numpy<2.0"

# Build custom CUDA modules
RUN pip install /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN pip install /workspace/gaussian-splatting/submodules/simple-knn
RUN pip install /workspace/gaussian-splatting/submodules/fused-ssim

# Install remaining Python dependencies
RUN pip install plyfile tqdm

CMD ["bash"]
```

### Step 3: Add the service to docker-compose.yml

See the [docker-compose.yml section](#docker-composeyml) below for the full file. The relevant service block is:

```yaml
  3dgs:
    build:
      context: ./gaussian-splatting
      dockerfile: Dockerfile
    image: 3dgs:latest
```

### Step 4: Build the Docker image

```bash
cd ~/Documents/gaussian-splatting-projects
sudo docker compose build 3dgs
```

This compiles three custom CUDA modules inside the image:
- `diff-gaussian-rasterization` — differentiable Gaussian rasterizer
- `simple-knn` — CUDA k-nearest-neighbors
- `fused-ssim` — fused SSIM loss computation

The resulting image is ~25 GB.

### Step 5: Verify the build

```bash
sudo docker compose run --rm 3dgs python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
from diff_gaussian_rasterization import GaussianRasterizer
print('diff-gaussian-rasterization OK')
import fused_ssim
print('fused-ssim OK')
"
```

### Usage

```bash
# Interactive shell
sudo docker compose run --rm 3dgs bash

# Train (inside container)
python gaussian-splatting/train.py -s /workspace/data/your_scene

# Render
python gaussian-splatting/render.py -m /workspace/output/your_model

# Evaluate
python gaussian-splatting/metrics.py -m /workspace/output/your_model
```

### Docker image details

| Component | Version |
|-----------|---------|
| Base Image | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` |
| Python | 3.10 |
| PyTorch | 2.0.1 |
| CUDA | 11.7 |
| numpy | <2.0 (pinned) |

---

## Repository 2: gsplat (Nerfstudio)

**Repo:** https://github.com/nerfstudio-project/gsplat
**Docs:** https://docs.gsplat.studio/

A modular, research-friendly Gaussian Splatting library from the Nerfstudio team. Cleaner API, pip-installable, and easier to extend. Includes anti-aliased splatting, densification strategies, and full nerfstudio integration.

### Step 1: Create the Dockerfile

The gsplat repo is cloned inside the Docker image (not on the host). Create the file `gsplat.Dockerfile` in the project root:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch
RUN pip3 install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install gsplat
RUN pip3 install gsplat==1.4.0

# Install nerfstudio
RUN pip3 install nerfstudio

CMD ["bash"]
```

### Step 2: Add the service to docker-compose.yml

See the [docker-compose.yml section](#docker-composeyml) below. The relevant service block is:

```yaml
  gsplat:
    build:
      context: .
      dockerfile: gsplat.Dockerfile
    image: gsplat:latest
```

### Step 3: Build the Docker image

```bash
cd ~/Documents/gaussian-splatting-projects
sudo docker compose build gsplat
```

The resulting image is ~24.1 GB.

### Step 4: Verify the build

```bash
sudo docker compose run --rm gsplat python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import gsplat
print(f'gsplat {gsplat.__version__} OK')
"
```

### Usage

**Important:** Nerfstudio defaults to its own `transforms.json` data format. For COLMAP data, you must specify the `colmap` dataparser and set the `--colmap-path`. Note that `--output-dir` goes BEFORE the dataparser name, and `colmap` goes AFTER the method name.

```bash
# Interactive shell
sudo docker compose run --rm gsplat bash

# Train with COLMAP data (inside container)
ns-train splatfacto --output-dir /workspace/output/gsplat-29palms \
    colmap --data /workspace/data/29-palms-subset --colmap-path sparse/0

# Using the gsplat Python API directly
python3 -c "
import torch
from gsplat import rasterization
# ... your code here
"
```

### Docker image details

| Component | Version |
|-----------|---------|
| Base Image | `nvidia/cuda:12.1.0-devel-ubuntu22.04` |
| Python | 3.10 |
| PyTorch | 2.4.0+cu121 |
| CUDA | 12.1 |
| gsplat | 1.4.0 |
| nerfstudio | included |

---

## Repository 3: 2D Gaussian Splatting

**Repo:** https://github.com/hbb1/2d-gaussian-splatting
**Paper:** "2D Gaussian Splatting for Geometrically Accurate Radiance Fields" (Huang et al., SIGGRAPH 2024)

Uses flat 2D Gaussian discs (surfels) instead of 3D ellipsoids. Produces significantly better surface geometry and mesh extraction. Best choice when you need actual 3D meshes from your data.

### Step 1: Create the Dockerfile

Like gsplat, the 2DGS repo is cloned inside the Docker image. Create the file `2dgs.Dockerfile` in the project root:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev \
    colmap imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone 2DGS
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

# Pin numpy <2.0 (PyTorch 2.0.1 is incompatible with numpy 2.x)
RUN pip install "numpy<2.0"

# Build custom CUDA modules
RUN pip install /workspace/2d-gaussian-splatting/submodules/diff-surfel-rasterization
RUN pip install /workspace/2d-gaussian-splatting/submodules/simple-knn

# Install remaining Python dependencies
RUN pip install plyfile tqdm open3d trimesh lpips scikit-image

CMD ["bash"]
```

### Step 2: Add the service to docker-compose.yml

See the [docker-compose.yml section](#docker-composeyml) below. The relevant service block is:

```yaml
  2dgs:
    build:
      context: .
      dockerfile: 2dgs.Dockerfile
    image: 2dgs:latest
```

### Step 3: Build the Docker image

```bash
cd ~/Documents/gaussian-splatting-projects
sudo docker compose build 2dgs
```

This compiles two custom CUDA modules inside the image:
- `diff-surfel-rasterization` — differentiable 2D surfel rasterizer
- `simple-knn` — CUDA k-nearest-neighbors

The resulting image is ~23.7 GB.

### Step 4: Verify the build

```bash
sudo docker compose run --rm 2dgs python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
from diff_surfel_rasterization import GaussianRasterizer
print('diff-surfel-rasterization OK')
"
```

### Usage

```bash
# Interactive shell
sudo docker compose run --rm 2dgs bash

# Train (inside container)
python 2d-gaussian-splatting/train.py -s /workspace/data/your_scene

# Render
python 2d-gaussian-splatting/render.py -m /workspace/output/your_model -s /workspace/data/your_scene

# Extract mesh
python 2d-gaussian-splatting/render.py -m /workspace/output/your_model -s /workspace/data/your_scene --mesh
```

### Docker image details

| Component | Version |
|-----------|---------|
| Base Image | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` |
| Python | 3.10 |
| PyTorch | 2.0.1 |
| CUDA | 11.7 |
| numpy | <2.0 (pinned) |
| Additional | open3d, trimesh, lpips, scikit-image |

---

## docker-compose.yml

Create this file at `~/Documents/gaussian-splatting-projects/docker-compose.yml`. It orchestrates all three services and defines the volume mounts that make data available inside each container.

```yaml
services:
  3dgs:
    build:
      context: ./gaussian-splatting
      dockerfile: Dockerfile
    image: 3dgs:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
    volumes:
      - ./output:/workspace/output
      # Add your prepared data directories here:
      # - ./data/YOUR-DATASET-subset:/workspace/data/YOUR-DATASET-subset
      # Add source image mount for symlink resolution (absolute path, read-only):
      # - /path/to/original/images:/path/to/original/images:ro

  gsplat:
    build:
      context: .
      dockerfile: gsplat.Dockerfile
    image: gsplat:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
    volumes:
      - ./output:/workspace/output
      # Add your prepared data directories here:
      # - ./data/YOUR-DATASET-subset:/workspace/data/YOUR-DATASET-subset
      # Add source image mount for symlink resolution (absolute path, read-only):
      # - /path/to/original/images:/path/to/original/images:ro

  2dgs:
    build:
      context: .
      dockerfile: 2dgs.Dockerfile
    image: 2dgs:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
    volumes:
      - ./output:/workspace/output
      # Add your prepared data directories here:
      # - ./data/YOUR-DATASET-subset:/workspace/data/YOUR-DATASET-subset
      # Add source image mount for symlink resolution (absolute path, read-only):
      # - /path/to/original/images:/path/to/original/images:ro
```

### Build all images at once

```bash
cd ~/Documents/gaussian-splatting-projects
sudo docker compose build
```

---

## Adding Your Data

### Volume mounts explained

Each service in `docker-compose.yml` needs up to 3 categories of volume mounts:

**1. Output directory (always present)**

```yaml
- ./output:/workspace/output
```

All training results go here. Shared across all services.

**2. Prepared data directories (one per dataset you create)**

```yaml
- ./data/29-palms-subset:/workspace/data/29-palms-subset
- ./data/cal-poly-chunks:/workspace/data/cal-poly-chunks
```

These are subsetted/chunked/flattened scenes created by `create_subset.py` or `flatten_scene.py`. They contain `images/` (symlinks to the originals) and `sparse/0/` (COLMAP text files). Must be read-write because training converts `points3D.txt` to `.ply` in-place on first run.

**3. Source image directory for symlink resolution (absolute path, read-only)**

```yaml
- /home/edgar/Desktop/Cal-Poly:/home/edgar/Desktop/Cal-Poly:ro
```

The prepared data directories contain symlinks pointing to the original full-resolution images on the host. Docker must mount the original image location at the **exact same absolute path** inside the container so the symlinks resolve.

**CRITICAL:** This must be an absolute path (starts with `/`). A common mistake is writing `./home/edgar/...` which Docker interprets as a relative subdirectory.

To find what path your symlinks point to:

```bash
ls -la data/your-dataset/images/ | head -3
```

### Known issue: image subdirectories

Both 3DGS and 2DGS use `os.path.basename()` when reading image paths from COLMAP's `images.txt`. If your images are in subdirectories (e.g. `000000/000000_0.png`), the code strips the directory prefix and cannot find the file.

- **Simple case** (unique filenames across subdirs): Use `create_subset.py` which replaces `/` with `_` in image names.
- **Multi-lens case** (duplicate filenames across subdirs): Use `flatten_scene.py` which converts the full relative path to a flat name.
- **gsplat/nerfstudio**: Handles subdirectories natively. No flattening needed.

### Preparing a subset

```bash
python3 create_subset.py \
    --source data/your-scene \
    --images-dir /path/to/original/images \
    --output data/your-scene-subset \
    --every 10    # every 10th image = 10% subset
```

### Chunking large datasets

When a dataset is too large to train at once, split it into consecutive chunks:

```bash
# Using create_subset.py (simple subdirectory structure)
python3 create_subset.py \
    --source data/your-scene \
    --images-dir /path/to/original/images \
    --output data/your-scene-chunks \
    --num-chunks 7

# Using flatten_scene.py (multi-lens / deep subdirectory data)
python3 flatten_scene.py \
    --source /path/to/source \
    --images-dir /path/to/source \
    --output data/your-scene-chunks \
    --num-chunks 7
```

Both scripts produce `chunk-0/` through `chunk-N/`, each with its own `images/`, `sparse/0/`, and filtered `points3D.txt`.

---

## Quick Reference

### Method comparison

| Feature | Original 3DGS | gsplat | 2DGS |
|---------|---------------|--------|------|
| Rendering quality | Excellent | Excellent | Excellent |
| Surface reconstruction | Poor | Moderate | Best |
| Training speed | Fast (~15-30 min) | Fast (~15-30 min) | Moderate (~20-40 min) |
| Mesh extraction | No | Limited | Yes (built-in) |
| API / extensibility | Basic scripts | Modular Python API | Basic scripts |
| Best for | Baseline comparison | Research / integration | Geometry / meshes |

### Typical VRAM usage (training)

| Method | VRAM |
|--------|------|
| Original 3DGS | ~8-12 GB |
| gsplat | ~6-10 GB |
| 2DGS | ~8-12 GB |

### Preparing your own data with COLMAP

If you have raw images and no COLMAP reconstruction, you can run COLMAP inside the 3DGS container:

```bash
# Place images in data/your_scene/input/
sudo docker compose run --rm 3dgs bash -c "
  cd /workspace/data/your_scene && \
  mkdir -p distorted/sparse && \
  colmap feature_extractor --database_path distorted/database.db --image_path input && \
  colmap exhaustive_matcher --database_path distorted/database.db && \
  colmap mapper --database_path distorted/database.db --image_path input --output_path distorted/sparse
"
```
