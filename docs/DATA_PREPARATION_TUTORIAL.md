# Data Preparation Tutorial

How to convert a production dataset into a format ready for Gaussian Splatting training.

---

## Overview

Production datasets from the Geocam pipeline arrive in `production-downloads/` with this structure:

```
production-downloads/your-dataset/
├── images/
│   └── flight_folder_name/
│       ├── 0/              (lens 0)
│       │   ├── 00000/      (batch 0)
│       │   │   ├── 00000006.png (or .jpg)
│       │   │   ├── 00000006.xmp
│       │   │   └── ...
│       │   └── 00001/      (batch 1)
│       ├── 1/              (lens 1 — same filenames as lens 0!)
│       │   └── 00000/
│       │       ├── 00000006.png
│       │       └── ...
│       └── ...             (more lenses)
├── sparse/
│   ├── cameras.txt         (COLMAP camera intrinsics)
│   ├── images.txt          (COLMAP image poses + 2D feature points)
│   └── points3D.txt        (COLMAP 3D point cloud)
├── masks/                  (segmentation masks, if generated)
├── flightlog.txt           (GPS coordinates per image — needed for georeferencing)
├── info.json               (processing metadata, cell geometry, export settings)
├── crs.json                (coordinate reference system — needed for georeferencing)
├── Project_GCD/            (RealityCapture project files)
├── image_list.imagelist
├── image_list.rcproj
├── project.ini
└── undistorted_image_list_rel.txt
```

The prepared dataset in `data/` needs to look like this:

```
data/your-dataset/
├── images/
│   ├── flight_folder_name_0_00000_00000006.png   (flat name, symlink)
│   ├── flight_folder_name_1_00000_00000006.png   (different lens, unique name)
│   └── ...
├── sparse/
│   └── 0/
│       ├── cameras.txt
│       ├── images.txt      (rewritten with flat image names)
│       └── points3D.txt    (filtered to referenced points only)
├── flightlog.txt           (copied from source — for georeferencing)
├── info.json               (copied from source — for georeferencing)
└── crs.json                (copied from source — for georeferencing)
```

The key differences:
1. **Flat image directory** — no subdirectories, all images at the top level of `images/`
2. **Unique filenames** — subdirectory path baked into the filename (e.g. `flight_0_00000_00000006.png`)
3. **Symlinks** — images are symlinked, not copied, saving disk space
4. **`sparse/0/`** — COLMAP files inside a `0/` subdirectory (what 3DGS expects)
5. **Rewritten `images.txt`** — image paths updated to match the flat filenames
6. **Metadata carried over** — `flightlog.txt`, `info.json`, `crs.json` for georeferencing

---

## Step-by-step

### Step 1: Identify your source dataset

```bash
cd ~/Documents/gaussian-splatting-projects

# List available datasets
ls production-downloads/
```

Pick the dataset you want to prepare. For this tutorial, we'll use a generic name `your-dataset`.

### Step 2: Inspect the dataset

Before preparing, check what you're working with:

```bash
DATASET="production-downloads/your-dataset"

# How many images?
grep -v "^#" "$DATASET/sparse/images.txt" | grep -cE "\.png|\.jpg"

# How many cameras?
grep -v "^#" "$DATASET/sparse/cameras.txt" | wc -l

# How many 3D points?
grep -v "^#" "$DATASET/sparse/points3D.txt" | wc -l

# What resolution?
file "$(find "$DATASET/images/" -name '*.png' -o -name '*.jpg' | head -1)"

# How much disk space?
du -sh "$DATASET/images/"

# How many lenses / flight folders?
ls "$DATASET/images/"
ls "$DATASET/images/"*/    # lens directories

# Check for duplicate filenames across lenses
ls "$DATASET/images/"*/0/00000/ | head -5
```

### Step 3: Flatten the full dataset

```bash
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-full
```

This will:
- Parse `images.txt` and `points3D.txt` from the sparse directory
- Create symlinks with flattened names in `data/your-dataset-full/images/`
- Write updated COLMAP files to `data/your-dataset-full/sparse/0/`
- Copy `flightlog.txt`, `info.json`, and `crs.json` for georeferencing

Example output:
```
Source: production-downloads/your-dataset
Images: production-downloads/your-dataset/images
Sparse: production-downloads/your-dataset/sparse
Parsing images.txt...
  Found 15060 images
Parsing points3D.txt...
  Found 588148 points

Writing flattened scene -> data/your-dataset-full
  Images: 15060 symlinked, 0 skipped (missing)
  Points: 588148/588148 retained
  Metadata: copied flightlog.txt, info.json, crs.json

Done.
```

### Step 4: Verify the output

```bash
# Check the structure
ls data/your-dataset-full/
# Expected: images/  sparse/  flightlog.txt  info.json  crs.json

# Check images are symlinked
ls -la data/your-dataset-full/images/ | head -5

# Check sparse files exist
ls data/your-dataset-full/sparse/0/
# Expected: cameras.txt  images.txt  points3D.txt

# Verify symlinks resolve
file data/your-dataset-full/images/$(ls data/your-dataset-full/images/ | head -1)
```

### Step 5: Add volume mounts to docker-compose.yml

Edit `docker-compose.yml` and add two lines under the `volumes:` section of each service (`3dgs`, `gsplat`, `2dgs`):

```yaml
volumes:
  - ./output:/workspace/output
  - ./data/your-dataset-full:/workspace/data/your-dataset-full                    # new
  - /absolute/path/to/production-downloads:/absolute/path/to/production-downloads:ro  # new (if not already present)
```

The second mount is required so symlinks resolve inside the container. Check what absolute path your symlinks point to:

```bash
ls -la data/your-dataset-full/images/ | head -3
```

The mount path must match the symlink target prefix.

### Step 6: Train

```bash
# Start with a low resolution to verify everything works
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset-full \
    -m /workspace/output/test-run \
    -r 4 --iterations 7000
```

If that succeeds, scale up resolution and iterations as VRAM allows.

---

## Creating subsets for large datasets

If the full dataset OOMs at your desired resolution, create a consecutive subset. **Always use consecutive images** — skipping every Nth image creates spatial gaps and produces poor results.

### First quarter

```bash
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-q1 \
    --num-chunks 4 --chunk 0
```

Note: with `--num-chunks`, output goes into `chunk-0/` inside the output directory.

```bash
# Training uses the chunk-0 path
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/your-dataset-q1/chunk-0 \
    -m /workspace/output/your-dataset-q1 \
    -r 2 --iterations 50000 --save_iterations 7000 30000 50000
```

### First half

```bash
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-half \
    --num-chunks 2 --chunk 0
```

### Custom fraction (e.g. first 3/4)

For fractions that don't divide evenly into chunks, use Python directly:

```python
cd ~/Documents/gaussian-splatting-projects
python3 -c "
from flatten_scene import parse_images_txt, parse_points3d_txt, write_chunk, copy_metadata
from pathlib import Path

source = Path('production-downloads/your-dataset')
sparse_dir = source / 'sparse'

entries = parse_images_txt(sparse_dir / 'images.txt')
all_points = parse_points3d_txt(sparse_dir / 'points3D.txt')

n = len(entries) * 3 // 4
print(f'Taking first {n} of {len(entries)} images')
write_chunk(entries[:n], all_points, source / 'images', 'data/your-dataset-3q', sparse_dir)
copy_metadata(source, 'data/your-dataset-3q')
"
```

---

## Splitting into chunks for full scene coverage

To train the entire scene at a resolution that doesn't fit in one go, split into chunks. Each chunk covers a contiguous spatial region and produces an independent splat.

```bash
# Split into 4 chunks
python3 flatten_scene.py \
    --source production-downloads/your-dataset \
    --images-dir production-downloads/your-dataset/images \
    --output data/your-dataset-chunks \
    --num-chunks 4
```

This creates:
```
data/your-dataset-chunks/
├── chunk-0/
│   ├── images/           (~N/4 symlinked images)
│   ├── sparse/0/         (filtered COLMAP files)
│   ├── flightlog.txt
│   ├── info.json
│   └── crs.json
├── chunk-1/
│   └── ...
├── chunk-2/
│   └── ...
└── chunk-3/
    └── ...
```

Train each chunk separately:

```bash
for i in 0 1 2 3; do
    sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
        -s /workspace/data/your-dataset-chunks/chunk-$i \
        -m /workspace/output/your-dataset-chunk-$i \
        --iterations 50000 --save_iterations 7000 30000 50000
done
```

---

## How the image ordering works

Images in `images.txt` are ordered roughly by capture time. Multi-lens datasets group images by **capture event** — all lenses that fire simultaneously are grouped together:

```
flight_folder/0/00000/00000006.png    (lens 0, frame 6)
flight_folder/1/00000/00000006.png    (lens 1, frame 6)
flight_folder/2/00000/00000006.png    (lens 2, frame 6)
flight_folder/0/00000/00000007.png    (lens 0, frame 7)
flight_folder/1/00000/00000007.png    (lens 1, frame 7)
...
```

Within a multi-flight dataset, flights appear in sequence. So "the first quarter of images" maps to roughly the first quarter of the physical capture trajectory — a contiguous spatial region.

This is why consecutive subsets produce good results (contiguous coverage) while skipping every Nth image produces poor results (sparse coverage with gaps).

---

## VRAM sizing guide

Use this table to decide what resolution and image count will fit on your GPU:

| Source resolution | `-r 4` | `-r 2` | Full res |
|------------------|--------|--------|----------|
| 504x504 (PNG) | 50,000+ images | ~12,000 images | ~1,000 images |
| 504x252 (JPG) | 15,000+ images | 15,000+ images | ~3,700 images |

*Based on NVIDIA L4 (22 GB VRAM). Scale proportionally for other GPUs.*

**Decision tree:**
1. How many images does your dataset have?
2. What resolution do you want to train at?
3. Look up the limit in the table above
4. If your image count exceeds the limit → create a subset or chunk the dataset

---

## flatten_scene.py reference

```
usage: flatten_scene.py --source SOURCE --images-dir IMAGES_DIR --output OUTPUT
                        [--num-chunks N] [--chunk I] [--every N]

arguments:
  --source        Path to source dataset (containing sparse/ directory)
  --images-dir    Path to images directory within the source
  --output        Output directory for the prepared dataset
  --num-chunks    Split into N consecutive chunks (0 = no chunking, default)
  --chunk         Only create this specific chunk (0-indexed, -1 = all)
  --every         Take every Nth image (1 = all, default). Not recommended
                  for training — use --num-chunks instead.
```

The script automatically:
- Detects whether sparse files are in `sparse/` or `sparse/0/`
- Flattens subdirectory paths into unique flat filenames
- Creates absolute symlinks (work inside Docker containers)
- Rewrites `images.txt` with flattened names
- Filters `points3D.txt` to only points referenced by selected images
- Copies `cameras.txt` unchanged
- Copies `flightlog.txt`, `info.json`, `crs.json` for georeferencing

---

## Complete example: Dublin Station dataset

```bash
cd ~/Documents/gaussian-splatting-projects

# 1. Inspect
DATASET="production-downloads/dublin_station"
grep -v "^#" "$DATASET/sparse/images.txt" | grep -c ".jpg"
# -> 15060 images

file "$(find "$DATASET/images/" -name '*.jpg' | head -1)"
# -> JPEG image data, 504x252

# 2. Flatten the full dataset
python3 flatten_scene.py \
    --source production-downloads/dublin_station \
    --images-dir production-downloads/dublin_station/images \
    --output data/dublin-full

# 3. Verify
ls data/dublin-full/
# -> images/  sparse/  flightlog.txt  info.json  crs.json

ls data/dublin-full/images/ | wc -l
# -> 15060

# 4. Add to docker-compose.yml (under each service's volumes:)
#   - ./data/dublin-full:/workspace/data/dublin-full

# 5. Train at -r 2 (full dataset fits at this resolution)
sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/dublin-full \
    -m /workspace/output/3dgs-dublin-r2 \
    -r 2 --iterations 50000 --save_iterations 7000 30000 50000

# 6. For full res, use the first quarter
python3 flatten_scene.py \
    --source production-downloads/dublin_station \
    --images-dir production-downloads/dublin_station/images \
    --output data/dublin-q1 \
    --num-chunks 4 --chunk 0

sudo docker compose run --rm 3dgs python gaussian-splatting/train.py \
    -s /workspace/data/dublin-q1/chunk-0 \
    -m /workspace/output/3dgs-dublin-q1-fullres \
    --iterations 50000 --save_iterations 7000 30000 50000
```
