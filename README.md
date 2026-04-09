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
cp docker-compose.example.yml docker-compose.yml
docker compose build

# 3. Prepare your dataset
python3 flatten_scene.py \
    --source /path/to/your-dataset \
    --images-dir /path/to/your-dataset/images \
    --output data/your-dataset

# 4. Check VRAM requirements
python3 vram_calculator.py --images 15060 --width 504 --height 252 --points 588000

# 5. Update volume mounts in docker-compose.yml, then train
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset \
    -m /workspace/output/my-splat \
    -r 2 --iterations 50000 --save_iterations 7000 30000 50000

# 6. Georeference
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

## Tools

### flatten_scene.py

Prepare datasets for training: flatten multi-lens subdirectory layouts, create consecutive subsets, split into chunks.

```bash
# Flatten full dataset
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/my-dataset

# First quarter only
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/my-dataset-q1 --num-chunks 4 --chunk 0

# Split into 5 chunks for full coverage at high res
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/my-dataset-chunks --num-chunks 5

# Filter to specific images using a CSV file (must have a "Name" column with image paths)
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/my-dataset-filtered --filter /path/to/filtered_flightlog.csv

# Filter and chunk (useful when filtered images still exceed VRAM at full res)
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/my-dataset-filtered-chunks --filter /path/to/filtered_flightlog.csv --num-chunks 4
```

The `--filter` flag accepts a CSV file with a `Name` column containing image paths (e.g. `flight_folder/0/00006/00006429.jpg`). Only images present in the CSV are included in the output. This can be combined with `--num-chunks` to further split filtered results.

Automatically copies `flightlog.txt`, `info.json`, and `crs.json` for georeferencing.

### georeference_splat.py

Apply real-world coordinates to trained splat PLY files using GPS data from the flightlog.

```bash
python3 georeference_splat.py --data data/my-dataset --ply output/my-splat/point_cloud/iteration_50000/point_cloud.ply --output output/my-splat/point_cloud_georeferenced.ply
```

Supports lat/lon flightlogs (auto-converts to UTM) and local coordinate flightlogs. Handles flattened image names automatically.

### vram_calculator.py

Estimate GPU memory requirements before training.

```bash
# Check a specific dataset
python3 vram_calculator.py --images 15060 --width 504 --height 252 --points 588000

# Check with a different GPU
python3 vram_calculator.py --images 15060 --width 504 --height 252 --points 588000 --vram 48

# Show estimates for example datasets
python3 vram_calculator.py
```

## Repository structure

```
.
├── README.md
├── flatten_scene.py              # Prepare datasets (flatten, subset, chunk)
├── georeference_splat.py         # Georeference trained splat PLYs
├── vram_calculator.py            # Estimate GPU memory requirements
├── docker-compose.example.yml    # Docker Compose template (with memory limits)
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
├── flightlog.txt            # GPS coordinates per image (for georeferencing)
├── crs.json                 # Coordinate reference system (for georeferencing)
└── info.json                # Processing metadata
```

## VRAM requirements

All images are loaded into GPU memory by default. Use `vram_calculator.py` to check your specific dataset, or refer to these empirical limits for a 22 GB GPU (NVIDIA L4):

| Source resolution | `-r 8` | `-r 4` | `-r 2` | Full res |
|------------------|--------|--------|--------|----------|
| 504x504 | 275K+ | 50K+ | ~12K | ~1K |
| 504x252 | 670K+ | 165K+ | 15K+ | ~3.7K |
| 3000x4096 | ~6.8K | ~1.7K | ~430 | ~106 |

For datasets that exceed your VRAM at the desired resolution, split into consecutive chunks:

```bash
# Split into 5 chunks, train each independently
python3 flatten_scene.py --source /path/to/dataset --images-dir /path/to/dataset/images --output data/chunks --num-chunks 5

for i in 0 1 2 3 4; do
    sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
        -s /workspace/data/chunks/chunk-$i \
        -m /workspace/output/chunk-$i \
        --iterations 50000 --save_iterations 7000 30000 50000
done
```

All chunks georeference to the same coordinate system, so their PLYs align when combined.

## Memory safety

Docker containers have `mem_limit` set in `docker-compose.example.yml` to prevent the Linux OOM killer from taking down system services (sshd, etc.) during large training runs. Adjust based on your machine's total RAM — leave ~5 GB for the OS. Adding swap space is also recommended:

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## License

MIT
