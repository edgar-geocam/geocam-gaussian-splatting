# Running Guide — Gaussian Splatting Methods

How to prepare datasets and train, render, and evaluate with each of the 3 Gaussian Splatting methods.

All commands run from `~/Documents/gaussian-splatting-projects/`.

---

## Preparing a Dataset

### Source data layout

Datasets exported from RealityCapture typically look like this:

```
your-dataset/
├── images/
│   └── flight_folder/
│       ├── 0/          (lens 0)
│       │   └── 00000/  (batch)
│       │       ├── 00000006.png (or .jpg)
│       │       └── ...
│       ├── 1/          (lens 1)
│       └── ...
├── sparse/
│   ├── cameras.txt     (COLMAP camera intrinsics)
│   ├── images.txt      (COLMAP image poses + 2D features)
│   └── points3D.txt    (COLMAP 3D point cloud)
└── masks/              (optional)
```

### Why flattening is required

3DGS and 2DGS use `os.path.basename()` when reading image paths from COLMAP's `images.txt`. If images are in subdirectories, the code strips the directory and can't find the file. Multi-lens datasets also have duplicate filenames across lenses (e.g. lens 0 and lens 1 both contain `00000006.png`).

The `flatten_scene.py` script solves both problems by:
1. Converting paths like `flight/0/00000/00000006.png` to flat names like `flight_0_00000_00000006.png`
2. Creating symlinks to the original images (no disk space wasted)
3. Rewriting `images.txt` with the flat names
4. Placing sparse files in `sparse/0/` (the layout 3DGS expects)
5. Filtering `points3D.txt` to only include points referenced by the selected images

**gsplat/nerfstudio** handles subdirectories natively and does not need flattening, but flattened datasets work with all three methods.

### Flatten a full dataset

```bash
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-full
```

Output:
```
data/your-dataset-full/
├── images/           (symlinks with flat names)
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt    (rewritten with flat names)
        └── points3D.txt  (filtered to referenced points)
```

### Create a consecutive subset

For large datasets that OOM at the desired resolution, take the first N images. Images in `images.txt` are ordered roughly chronologically by capture, so consecutive images cover contiguous spatial regions. This produces much better results than skipping every Nth image, which creates spatial gaps.

```bash
# First quarter (chunk 0 of 4)
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-q1 \
    --num-chunks 4 --chunk 0

# First half (chunk 0 of 2)
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-half \
    --num-chunks 2 --chunk 0
```

When using `--num-chunks`, output goes into `chunk-0/` inside the output directory:

```bash
# Training path for chunked data
-s /workspace/data/your-dataset-q1/chunk-0
```

You can also create a custom fraction using Python directly:

```python
from flatten_scene import parse_images_txt, parse_points3d_txt, write_chunk
from pathlib import Path

source = Path('production-downloads/your-dataset')
sparse_dir = source / 'sparse'

entries = parse_images_txt(sparse_dir / 'images.txt')
all_points = parse_points3d_txt(sparse_dir / 'points3D.txt')

# Take first 3/4 of images
n = len(entries) * 3 // 4
write_chunk(entries[:n], all_points, source / 'images', 'data/your-dataset-3q', sparse_dir)
```

### Split into multiple chunks for full coverage

To cover a large scene at full resolution, split into chunks and train each independently:

```bash
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-chunks \
    --num-chunks 4
```

This produces `chunk-0/` through `chunk-3/`, each with its own `images/`, `sparse/0/`, and filtered `points3D.txt`. Train each chunk separately to get full scene coverage.

### Add volume mounts

After preparing a dataset, add it to the `volumes:` section of each service in `docker-compose.yml`:

```yaml
volumes:
  - ./output:/workspace/output
  - ./data/your-dataset-full:/workspace/data/your-dataset-full
  - /absolute/path/to/production-downloads:/absolute/path/to/production-downloads:ro
```

The source image mount must use the **exact absolute path** that the symlinks point to, so they resolve inside the container. Check with:

```bash
ls -la data/your-dataset-full/images/ | head -3
```

---

## Method 1: Original 3D Gaussian Splatting (3DGS)

Container: `3dgs`

### Train

```bash
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset \
    -m /workspace/output/your-output-name \
    -r 4 \
    --iterations 30000 \
    --save_iterations 7000 30000
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-s` | — | Path to scene (must contain `images/` and `sparse/0/`) |
| `-m` | — | Output model directory |
| `-r` | -1 (auto) | Resolution downscale factor (2=half, 4=quarter). Omit for full res. |
| `--iterations` | 30000 | Total training iterations |
| `--save_iterations` | 7000 30000 | Save checkpoints at these iterations |
| `--test_iterations` | 7000 30000 | Run eval at these iterations |
| `--data_device` | cuda | Where to store images: `cuda` (VRAM) or `cpu` (system RAM) |
| `--sh_degree` | 3 | Spherical harmonics degree (0–3) |
| `--lambda_dssim` | 0.2 | SSIM loss weight (0.2 = 80% L1 + 20% SSIM) |
| `--densify_grad_threshold` | 0.0002 | Gradient threshold for densification |
| `--densify_until_iter` | 15000 | Stop adding new Gaussians after this iteration |
| `-w` | false | White background |
| `--eval` | false | Hold out test views |
| `--antialiasing` | false | Enable anti-aliased rasterization |

Examples:

```bash
# Quick test — 7k iterations at quarter res
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset -m /workspace/output/test-run \
    -r 4 --iterations 7000

# Full training at half res, 50k iterations
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset -m /workspace/output/full-run \
    -r 2 --iterations 50000 --save_iterations 7000 30000 50000

# Resume from checkpoint
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset -m /workspace/output/full-run \
    --start_checkpoint /workspace/output/full-run/chkpnt30000.pth
```

### Render

```bash
sudo docker compose run --rm 3dgs python gaussian-splatting/render.py \
    -m /workspace/output/your-output-name
```

Renders to `{model_path}/train/ours_{iteration}/` and `test/ours_{iteration}/`.

### Evaluate

```bash
sudo docker compose run --rm 3dgs python gaussian-splatting/metrics.py \
    -m /workspace/output/your-output-name
```

Computes PSNR, SSIM, and LPIPS. Requires `--eval` during training.

### Alpha mask support

3DGS supports per-pixel masking via the alpha channel of RGBA PNG images. Pixels with alpha < 1.0 are excluded from the training loss. This is useful for masking sky, noise, or other artifacts.

If your source images are JPGs with separate mask files, convert them to RGBA PNGs before flattening:
1. Load the JPG (RGB)
2. Load the mask (grayscale, white = keep, black = exclude)
3. Combine into a 4-channel RGBA PNG
4. Place in the images directory

The training code automatically detects 4-channel images and applies the alpha mask.

### Output structure

```
output/your-output-name/
├── cameras.json
├── cfg_args              (training configuration)
├── input.ply             (initial point cloud)
├── exposure.json
└── point_cloud/
    ├── iteration_7000/
    │   └── point_cloud.ply
    ├── iteration_30000/
    │   └── point_cloud.ply
    └── iteration_50000/
        └── point_cloud.ply
```

---

## Method 2: gsplat (Nerfstudio)

Container: `gsplat`

### Train

gsplat uses the nerfstudio `ns-train` CLI. The command structure is subcommand-based: trainer flags go **before** `colmap`, dataparser flags go **after** `colmap`.

```bash
echo "y" | sudo docker compose run --rm gsplat ns-train splatfacto \
    --output-dir /workspace/output/your-output-name \
    --max-num-iterations 30000 \
    colmap \
    --data /workspace/data/your-dataset \
    --colmap-path sparse/0 \
    --downscale-factor 4
```

The `echo "y" |` prefix is required because nerfstudio prompts interactively to confirm image downscaling on the first run.

Key flags before `colmap`:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `outputs/` | Where to save results |
| `--max-num-iterations` | 30000 | Total training iterations |
| `--pipeline.model.sh_degree` | 3 | Spherical harmonics degree |
| `--pipeline.model.num_downscales` | 2 | Start at 1/2^N resolution, then increase |
| `--pipeline.model.resolution_schedule` | 3000 | Double resolution every N steps |
| `--pipeline.model.rasterize_mode` | classic | `classic` or `antialiased` |
| `--pipeline.model.use_scale_regularization` | false | Reduce spikey Gaussians |

Key flags after `colmap`:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | — | Path to dataset |
| `--colmap-path` | `colmap/sparse/0` | Path to sparse dir relative to `--data` |
| `--downscale-factor` | auto | Resolution downscale (2, 4, etc.) |

Notes:
- gsplat handles subdirectory image paths natively — flattening is optional
- If sparse files are in `sparse/` (not `sparse/0/`), use `--colmap-path sparse`
- Nerfstudio uses ffmpeg to downscale images on first run, creating `images_N/` alongside `images/`
- With >500 images, nerfstudio automatically caches images on CPU

### Render

```bash
sudo docker compose run --rm gsplat ns-render spiral \
    --load-config /workspace/output/your-output-name/unnamed/splatfacto/TIMESTAMP/config.yml \
    --output-path /workspace/output/your-output-name/spiral.mp4
```

Other render modes: `interpolate`, `dataset`.

### Export

```bash
# Export as .ply
sudo docker compose run --rm gsplat ns-export gaussian-splat \
    --load-config /workspace/output/.../config.yml \
    --output-dir /workspace/output/exports/

# Export mesh via TSDF
sudo docker compose run --rm gsplat ns-export tsdf \
    --load-config /workspace/output/.../config.yml \
    --output-dir /workspace/output/mesh/
```

### Output structure

```
output/your-output-name/unnamed/splatfacto/TIMESTAMP/
├── config.yml
├── dataparser_transforms.json
└── nerfstudio_models/
    └── step-NNNNN.ckpt
```

---

## Method 3: 2D Gaussian Splatting (2DGS)

Container: `2dgs`

### Train

Uses the same CLI pattern as 3DGS:

```bash
sudo docker compose run --rm 2dgs python 2d-gaussian-splatting/train.py \
    -s /workspace/data/your-dataset \
    -m /workspace/output/your-output-name \
    -r 4 \
    --iterations 30000
```

Same flags as 3DGS, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda_normal` | 0.05 | Normal consistency regularization |
| `--lambda_dist` | 0.0 | Depth distortion regularization |
| `--depth_ratio` | 0.0 | 0 = mean depth (outdoor), 1 = median depth (indoor/objects) |

### Render

```bash
sudo docker compose run --rm 2dgs python 2d-gaussian-splatting/render.py \
    -m /workspace/output/your-output-name \
    -s /workspace/data/your-dataset
```

Note: 2DGS render requires both `-m` (model) **and** `-s` (source data).

### Mesh extraction

This is 2DGS's main advantage over 3DGS:

```bash
# Bounded mesh (TSDF fusion) — best for outdoor scenes
sudo docker compose run --rm 2dgs python 2d-gaussian-splatting/render.py \
    -m /workspace/output/your-output-name \
    -s /workspace/data/your-dataset \
    --depth_ratio 0 --skip_test --skip_train

# Unbounded mesh (Marching Cubes)
sudo docker compose run --rm 2dgs python 2d-gaussian-splatting/render.py \
    -m /workspace/output/your-output-name \
    -s /workspace/data/your-dataset \
    --unbounded --mesh_res 1024 --skip_test --skip_train
```

---

## Running COLMAP from scratch

If you have raw images but no COLMAP reconstruction:

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

---

## VRAM Guidelines

By default, all images are loaded into GPU memory at once (`--data_device cuda`). The main factors affecting VRAM:
- **Number of images** (each stored as an uncompressed tensor)
- **Resolution per image** (pixels x 4 bytes for RGBA)
- **Number of Gaussian points** (grows during densification)

### Choosing a resolution

Based on experiments with a 22 GB GPU (NVIDIA L4):

| Source resolution | `-r 4` | `-r 2` | Full res |
|------------------|--------|--------|----------|
| 504x504 | 50,000+ images | ~12,000 images | ~1,000 images |
| 504x252 | 15,000+ images | 15,000+ images | ~3,700 images |

### Strategies for large datasets

1. **Reduce resolution** (`-r 2` or `-r 4`) — simplest approach, keeps all images
2. **Use consecutive subsets** — take the first N images to cover a contiguous spatial region
3. **Chunk the dataset** — split into chunks, train each independently for full coverage
4. **Do NOT skip every Nth image** — this creates spatial gaps and produces poor results. Always use consecutive subsets.

### Training on chunks

```bash
# Create 4 chunks
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-chunks \
    --num-chunks 4

# Train each chunk
for i in 0 1 2 3; do
    sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
        -s /workspace/data/your-dataset-chunks/chunk-$i \
        -m /workspace/output/your-dataset-chunk-$i \
        --iterations 30000
done
```

---

## Troubleshooting

### CUDA out of memory during image loading

3DGS/2DGS load all images into VRAM by default. Solutions:
- Reduce resolution with `-r 2` or `-r 4`
- Use fewer images (consecutive subset or chunks)
- Use `--data_device cpu` to keep images in system RAM (slower, and may hit RAM limits with very large datasets)

### "Could not recognize scene type!"

The output directory has stale data from a failed run. Delete it and retry:
```bash
sudo rm -rf output/your-output-name
```

### "Image path not found" (3DGS / 2DGS)

Both repos use `os.path.basename()` which strips subdirectory paths. Run `flatten_scene.py` to create a flat image layout with rewritten `images.txt`.

### Nerfstudio prompts for downscaling and hangs

Pipe `yes` to the command: `echo "y" | sudo docker compose run --rm gsplat ns-train ...`

### Nerfstudio: "ffmpeg not found"

The gsplat Docker image must include ffmpeg. Rebuild with `sudo docker compose build gsplat`.

### "Too many open files"

The `docker-compose.yml` sets `ulimits.nofile` to 65536. If running containers manually, add `--ulimit nofile=65536:65536`.

### "Read-only file system" on sparse directory

Training converts `points3D.txt` to `.ply` in-place. The data volume must be writable (no `:ro` flag).

---

## Experimental Results

See `training_results.csv` for the full table of all runs with image counts, resolutions, durations, model sizes, and VRAM outcomes.
