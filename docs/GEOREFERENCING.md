# Georeferencing Gaussian Splats

How to apply real-world coordinates to trained Gaussian Splat outputs.

---

## Overview

Gaussian Splatting methods train in COLMAP's arbitrary coordinate system — the output PLY has no real-world position or scale. The `georeference_splat.py` script computes a similarity transform (rotation + scale + translation) between the COLMAP camera positions and GPS positions from the flightlog, then applies it to the output PLY.

The result is a georeferenced PLY in UTM coordinates (or local project coordinates, depending on the dataset).

---

## Prerequisites

The prepared dataset directory must contain:
- `sparse/0/images.txt` — COLMAP image poses (created by `flatten_scene.py`)
- `flightlog.txt` — GPS coordinates per image (copied from source by `flatten_scene.py`)
- `crs.json` — coordinate reference system (copied from source by `flatten_scene.py`)

These files are automatically included when you prepare a dataset with `flatten_scene.py`.

Install Python dependencies on the host:
```bash
pip3 install numpy plyfile
```

---

## Usage

### Basic usage (prepared dataset directory)

```bash
python3 georeference_splat.py \
    --data data/your-dataset \
    --ply output/your-splat/point_cloud/iteration_50000/point_cloud.ply \
    --output output/your-splat/point_cloud_georeferenced.ply
```

The `--data` flag points to your prepared dataset directory. The script automatically finds `sparse/0/images.txt`, `flightlog.txt`, and `crs.json` inside it.

### Explicit file paths

If your files aren't in the standard layout:

```bash
python3 georeference_splat.py \
    --images-txt /path/to/images.txt \
    --flightlog /path/to/flightlog.txt \
    --crs /path/to/crs.json \
    --ply /path/to/point_cloud.ply \
    --output /path/to/point_cloud_georeferenced.ply
```

---

## How it works

### Step 1: Extract COLMAP camera positions

The script reads `images.txt` and converts each camera's quaternion + translation from COLMAP's world-to-camera convention to actual camera position in COLMAP world coordinates.

### Step 2: Read GPS positions from flightlog

The `flightlog.txt` has no header. Each line is:
```
image_path, longitude_or_x, latitude_or_y, elevation
```

If `crs.json` specifies a UTM zone, columns 2/3 are treated as **longitude/latitude** (WGS84) and converted to UTM easting/northing. Otherwise, they are treated as **local project coordinates** and used directly.

### Step 3: Match cameras

The script matches COLMAP image names to flightlog image paths. It handles both:
- **Original paths** (e.g. `flight/0/00000/00000006.png`)
- **Flattened paths** (e.g. `flight_0_00000_00000006.png`) — automatically unflattened by converting `_` back to `/`

### Step 4: Compute similarity transform

Using Umeyama's method, the script finds the best-fit similarity transform:
```
dst = scale * R @ src + t
```
where `src` is the COLMAP position and `dst` is the GPS/UTM position.

The transform accounts for:
- **Scale** — COLMAP's arbitrary scale vs real-world meters
- **Rotation** — COLMAP's arbitrary orientation vs real-world north/east/up
- **Translation** — COLMAP's arbitrary origin vs UTM coordinates

### Step 5: Apply to PLY

The transform is applied to every Gaussian position (x, y, z) in the output PLY.

---

## Output

The script prints diagnostic information:

```
CRS: UTM Zone 10N (EPSG:32610)
  crsPose translation: [600635.00, 4178126.00, 0.00]

Reading COLMAP cameras from data/dublin-full/sparse/0/images.txt...
  Found 15060 COLMAP cameras
Reading flightlog from data/dublin-full/flightlog.txt...
  Found 15060 flightlog positions
Matching cameras...
  Matched 15060 cameras

Computing similarity transform (Umeyama's method)...
  Scale: 1.000000
  Translation: [600635.00, 4178126.00, -0.00]
  Residual mean: 0.0003 m
  Residual max:  0.0005 m
  Residual std:  0.0001 m

Applying transform to point_cloud.ply...
  Transformed 737950 points
  Written to point_cloud_georeferenced.ply

Coordinate system: EPSG:32610
```

Key things to check:
- **Matched cameras** should be close to the total count. If very few match, the image paths don't align.
- **Residual mean** should be small (sub-meter for good reconstructions). Large residuals indicate registration issues.
- **Scale** close to 1.0 means COLMAP already reconstructed at roughly metric scale.

---

## Coordinate systems

### Datasets with lat/lon flightlogs (e.g. Dublin Station)

The flightlog contains WGS84 longitude/latitude. The `crs.json` specifies the UTM zone. The output PLY is in UTM coordinates (easting, northing, elevation in meters).

### Datasets with local coordinate flightlogs (e.g. 9045)

The flightlog contains local project coordinates (small metric values). The `crs.json` has no CRS string. The output PLY stays in the local project frame. The `crsPose` in `crs.json` provides the transform from local to the project's geographic CRS if needed downstream.

---

## Georeferencing chunked datasets

Each chunk can be georeferenced independently. The transform is computed per-chunk using only the cameras in that chunk:

```bash
for i in 0 1 2 3; do
    python3 georeference_splat.py \
        --data data/your-dataset-chunks/chunk-$i \
        --ply output/your-dataset-chunk-$i/point_cloud/iteration_50000/point_cloud.ply \
        --output output/your-dataset-chunk-$i/point_cloud_georeferenced.ply
done
```

Since all chunks are georeferenced to the same coordinate system, their PLYs can be combined in a viewer or downstream tool and they will align correctly.

---

## Troubleshooting

### "Need at least 3 matching cameras"

The image paths in `images.txt` don't match the flightlog paths. This happens when:
- The flightlog uses original subdirectory paths but `images.txt` was rewritten with flattened names
- The script handles this automatically by trying to unflatten names, but check for path format mismatches

### Large residuals (>1m)

- Some images may have poor GPS or incorrect poses
- Try with `--data` pointing to the full (non-chunked) dataset for more matching points
- Check that `crs.json` has the correct UTM zone

### Scale far from 1.0

COLMAP sometimes reconstructs at a different scale than metric. The similarity transform corrects this, but a very large or small scale may indicate issues with the COLMAP reconstruction or GPS data.
