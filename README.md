# Gaussian Splatting Pipeline

End-to-end pipeline for training georeferenced Gaussian Splats from multi-lens aerial capture data. Supports three splatting methods running in Docker containers with GPU acceleration.

## Methods

| Method | Best for | Container |
|--------|----------|-----------|
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (INRIA) | Baseline, fast training | `3dgs` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) (Nerfstudio) | Research, modular API | `gsplat` |
| [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) | Mesh extraction, surface geometry | `2dgs` |

## Pipeline overview

```
1. Install prerequisites (GPU driver, CUDA, Docker)
2. Build Docker images for each method
3. Prepare dataset (flatten multi-lens images, create subsets)
4. Train Gaussian Splat
5. Georeference the output PLY
6. Convert to 3D Tiles (coming soon)
```

## Quick start

```bash
# Clone this repo
git clone https://github.com/YOUR_ORG/gaussian-splatting-pipeline.git
cd gaussian-splatting-pipeline

# 1. Set up the environment (see docs/SETUP_GUIDE.md)
# 2. Clone the 3DGS repo and build Docker images
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
docker compose build

# 3. Prepare your dataset
python3 flatten_scene.py \
    --source /path/to/your-dataset \
    --images-dir /path/to/your-dataset/images \
    --output data/your-dataset

# 4. Train (update volume mounts in docker-compose.yml first)
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset \
    -m /workspace/output/my-splat \
    -r 2 --iterations 50000 --save_iterations 7000 30000 50000

# 5. Georeference
python3 georeference_splat.py \
    --data data/your-dataset \
    --ply output/my-splat/point_cloud/iteration_50000/point_cloud.ply \
    --output output/my-splat/point_cloud_georeferenced.ply
```

## Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** — Install NVIDIA drivers, CUDA, Docker, build dependencies, and Docker images
- **[Data Preparation Tutorial](docs/DATA_PREPARATION_TUTORIAL.md)** — Convert production datasets to splat-ready format
- **[Running Guide](docs/RUNNING_GUIDE.md)** — Train, render, and evaluate with each method
- **[Georeferencing](docs/GEOREFERENCING.md)** — Apply real-world coordinates to trained splats

## Repository structure

```
.
├── README.md
├── flatten_scene.py              # Prepare datasets (flatten, subset, chunk)
├── georeference_splat.py         # Georeference trained splat PLYs
├── docker-compose.example.yml    # Docker Compose template
├── dockerfiles/
│   ├── 3dgs.Dockerfile           # Original 3D Gaussian Splatting
│   ├── gsplat.Dockerfile         # gsplat / Nerfstudio
│   └── 2dgs.Dockerfile           # 2D Gaussian Splatting
└── docs/
    ├── SETUP_GUIDE.md
    ├── DATA_PREPARATION_TUTORIAL.md
    ├── RUNNING_GUIDE.md
    └── GEOREFERENCING.md
```

## Input data format

This pipeline expects datasets exported from the Geocam production pipeline:

```
your-dataset/
├── images/                  # Multi-lens images in subdirectories
│   └── flight_folder/
│       ├── 0/               # Lens 0
│       ├── 1/               # Lens 1
│       └── ...
├── sparse/
│   ├── cameras.txt          # COLMAP camera intrinsics
│   ├── images.txt           # COLMAP image poses
│   └── points3D.txt         # COLMAP 3D point cloud
├── flightlog.txt            # GPS coordinates per image
├── crs.json                 # Coordinate reference system
└── info.json                # Processing metadata
```

## VRAM requirements

All images are loaded into GPU memory by default. Approximate limits for a 22 GB GPU (NVIDIA L4):

| Source resolution | `-r 4` | `-r 2` | Full res |
|------------------|--------|--------|----------|
| 504x504 | 50,000+ images | ~12,000 images | ~1,000 images |
| 504x252 | 15,000+ images | 15,000+ images | ~3,700 images |

For larger datasets, split into consecutive chunks and train each independently.

## License

MIT
