#!/usr/bin/env python3
"""
Georeference a trained Gaussian Splat output.

Computes a similarity transform (rotation + scale + translation) between
COLMAP's arbitrary coordinate system and real-world coordinates by matching
camera positions from the COLMAP reconstruction against GPS positions from
the flightlog.

Supports two modes based on what's in the flightlog:
  1. Lat/lon coordinates — converted to UTM using the zone from crs.json
  2. Local project coordinates — used directly (already in a metric frame)

Usage:
    python georeference_splat.py \
        --data data/dublin-full \
        --ply output/3dgs-dublin-r2-50k/point_cloud/iteration_50000/point_cloud.ply \
        --output output/3dgs-dublin-r2-50k/point_cloud_georeferenced.ply
"""

import argparse
import csv
import json
import math
import re
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def read_colmap_images_txt(path):
    """Extract camera positions from COLMAP images.txt.
    Returns dict of {filename: xyz_position}.

    Handles both original subdirectory paths and flattened paths
    (e.g. 'flight_0_00000_00000006.png' maps back to 'flight/0/00000/00000006.png').
    """
    cameras = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]

            # Convert from COLMAP (world-to-camera) to camera position in world
            R = quat_to_rot(qw, qx, qy, qz)
            cam_pos = -R.T @ np.array([tx, ty, tz])
            cameras[name] = cam_pos

            i += 2  # skip the 2D points line
        else:
            i += 1

    return cameras


def quat_to_rot(qw, qx, qy, qz):
    """Quaternion to rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def latlon_to_utm(lat, lon, zone_number, northern=True):
    """Convert lat/lon (WGS84) to UTM easting/northing in meters.

    Args:
        lat: latitude in degrees
        lon: longitude in degrees
        zone_number: UTM zone number (1-60)
        northern: True for northern hemisphere
    """
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2
    e_prime2 = e2 / (1 - e2)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0_rad = math.radians((zone_number - 1) * 6 - 180 + 3)  # central meridian

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e_prime2 * math.cos(lat_rad) ** 2
    A = (lon_rad - lon0_rad) * math.cos(lat_rad)

    M = a * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*lat_rad)
        + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*lat_rad)
        - (35*e2**3/3072) * math.sin(6*lat_rad)
    )

    k0 = 0.9996

    easting = k0 * N * (
        A + (1 - T + C) * A**3/6
        + (5 - 18*T + T**2 + 72*C - 58*e_prime2) * A**5/120
    ) + 500000.0

    northing = k0 * (
        M + N * math.tan(lat_rad) * (
            A**2/2
            + (5 - T + 9*C + 4*C**2) * A**4/24
            + (61 - 58*T + T**2 + 600*C - 330*e_prime2) * A**6/720
        )
    )

    if not northern:
        northing += 10000000.0

    return easting, northing


def parse_crs_json(crs_path):
    """Parse crs.json to extract UTM zone info and crsPose.

    Returns:
        utm_zone: int or None (None if no CRS / local coordinates)
        northern: bool (hemisphere)
        crs_pose: 3x3 rotation + 3D translation from crsPose array
        epsg: str like 'EPSG:32610' or None
    """
    with open(crs_path) as f:
        data = json.load(f)

    crs_str = data.get("crs", "")
    crs_pose_flat = data.get("crsPose", [1,0,0, 0,1,0, 0,0,1, 0,0,0])

    # Parse crsPose: first 9 values are 3x3 rotation, last 3 are translation
    rot = np.array(crs_pose_flat[:9]).reshape(3, 3)
    trans = np.array(crs_pose_flat[9:12])

    utm_zone = None
    northern = True
    epsg = None

    if crs_str:
        # Try to extract UTM zone from WKT or EPSG
        zone_match = re.search(r'UTM zone (\d+)([NS])', crs_str)
        if zone_match:
            utm_zone = int(zone_match.group(1))
            northern = zone_match.group(2) == 'N'

        # Extract EPSG for the projected CRS (the last AUTHORITY in the WKT)
        epsg_matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', crs_str)
        if epsg_matches:
            epsg = f"EPSG:{epsg_matches[-1]}"  # last one is the projected CRS

    return utm_zone, northern, rot, trans, epsg


def read_flightlog(flightlog_path, utm_zone=None, northern=True):
    """Read flightlog.txt and return dict of {image_path: xyz_position}.

    Format: image_path, longitude_or_x, latitude_or_y, elevation
    No header row.

    If utm_zone is provided, treats columns 2/3 as lon/lat and converts to UTM.
    Otherwise, treats them as direct metric coordinates.
    """
    geo = {}
    with open(flightlog_path, "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if len(row) < 4:
                continue

            image_path = row[0].strip()
            x_or_lon = float(row[1].strip())
            y_or_lat = float(row[2].strip())
            z_or_elev = float(row[3].strip())

            if utm_zone is not None:
                # Columns are lon, lat — convert to UTM
                easting, northing = latlon_to_utm(y_or_lat, x_or_lon, utm_zone, northern)
                geo[image_path] = np.array([easting, northing, z_or_elev])
            else:
                # Columns are already in a metric/local coordinate system
                geo[image_path] = np.array([x_or_lon, y_or_lat, z_or_elev])

    return geo


def match_cameras(colmap_cameras, geo_cameras):
    """Match COLMAP camera names to flightlog names.

    Handles flattened names by converting them back to original paths.
    e.g. 'flight_0_00000_00000006.png' tries to match 'flight/0/00000/00000006.png'
    """
    matched_colmap = []
    matched_geo = []
    matched_names = []

    # Build a lookup of geo cameras with normalized paths
    geo_lookup = {}
    for name, pos in geo_cameras.items():
        normalized = name.replace("\\", "/")
        geo_lookup[normalized] = pos

    for colmap_name, colmap_pos in colmap_cameras.items():
        # Try direct match first
        normalized_colmap = colmap_name.replace("\\", "/")

        if normalized_colmap in geo_lookup:
            matched_colmap.append(colmap_pos)
            matched_geo.append(geo_lookup[normalized_colmap])
            matched_names.append(colmap_name)
            continue

        # Try unflattening: replace _ with / and check all possible splits
        # The flattened name is path/components_joined_by_underscores
        # We need to find the original path with / separators
        # Strategy: try to match the filename part first, then work backwards
        found = False
        for geo_name, geo_pos in geo_lookup.items():
            # Check if the flattened version of geo_name matches colmap_name
            if geo_name.replace("/", "_") == colmap_name:
                matched_colmap.append(colmap_pos)
                matched_geo.append(geo_pos)
                matched_names.append(colmap_name)
                found = True
                break

        if not found:
            # Also try without extension differences
            colmap_stem = colmap_name.rsplit(".", 1)[0] if "." in colmap_name else colmap_name
            for geo_name, geo_pos in geo_lookup.items():
                geo_flat = geo_name.replace("/", "_")
                geo_stem = geo_flat.rsplit(".", 1)[0] if "." in geo_flat else geo_flat
                if geo_stem == colmap_stem:
                    matched_colmap.append(colmap_pos)
                    matched_geo.append(geo_pos)
                    matched_names.append(colmap_name)
                    break

    return np.array(matched_colmap), np.array(matched_geo), matched_names


def compute_similarity_transform(src_points, dst_points):
    """Compute similarity transform using Umeyama's method.
    Returns scale, R, t such that dst ≈ scale * R @ src + t."""
    n = src_points.shape[0]
    assert n >= 3, f"Need at least 3 matching points, got {n}"

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    H = src_centered.T @ dst_centered / n
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T

    src_var = np.sum(src_centered ** 2) / n
    scale = np.sum(S * np.diag(sign_matrix)) / src_var

    t = dst_mean - scale * R @ src_mean

    return scale, R, t


def apply_transform_to_ply(input_ply, output_ply, scale, R, t):
    """Apply similarity transform to all Gaussian positions in a PLY file.
    Also transforms the covariance-related scale if present."""
    ply = PlyData.read(input_ply)
    vertices = ply["vertex"]

    xyz = np.column_stack([
        np.array(vertices["x"]),
        np.array(vertices["y"]),
        np.array(vertices["z"]),
    ])

    xyz_transformed = (scale * (R @ xyz.T)).T + t

    vertices["x"] = xyz_transformed[:, 0].astype(np.float32)
    vertices["y"] = xyz_transformed[:, 1].astype(np.float32)
    vertices["z"] = xyz_transformed[:, 2].astype(np.float32)

    ply.write(output_ply)
    return xyz_transformed.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="Georeference a Gaussian Splat PLY using flightlog GPS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a prepared dataset directory (with flightlog.txt, crs.json, sparse/0/)
  python georeference_splat.py \\
      --data data/dublin-full \\
      --ply output/3dgs-dublin-r2-50k/point_cloud/iteration_50000/point_cloud.ply \\
      --output output/3dgs-dublin-r2-50k/point_cloud_georeferenced.ply

  # Using separate file paths
  python georeference_splat.py \\
      --images-txt data/dublin-full/sparse/0/images.txt \\
      --flightlog data/dublin-full/flightlog.txt \\
      --crs data/dublin-full/crs.json \\
      --ply output/3dgs-dublin-r2-50k/point_cloud/iteration_50000/point_cloud.ply \\
      --output output/3dgs-dublin-r2-50k/point_cloud_georeferenced.ply
        """)

    parser.add_argument("--data", "-d",
                        help="Path to prepared dataset directory (containing sparse/0/, flightlog.txt, crs.json)")
    parser.add_argument("--images-txt",
                        help="Path to COLMAP images.txt (overrides --data)")
    parser.add_argument("--flightlog",
                        help="Path to flightlog.txt (overrides --data)")
    parser.add_argument("--crs",
                        help="Path to crs.json (overrides --data)")
    parser.add_argument("--ply", "-p", required=True,
                        help="Path to trained point_cloud.ply")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to write georeferenced PLY")
    args = parser.parse_args()

    # Resolve file paths
    if args.data:
        data = Path(args.data)
        images_txt = Path(args.images_txt) if args.images_txt else data / "sparse" / "0" / "images.txt"
        flightlog = Path(args.flightlog) if args.flightlog else data / "flightlog.txt"
        crs_path = Path(args.crs) if args.crs else data / "crs.json"
    else:
        if not all([args.images_txt, args.flightlog]):
            parser.error("Either --data or both --images-txt and --flightlog are required")
        images_txt = Path(args.images_txt)
        flightlog = Path(args.flightlog)
        crs_path = Path(args.crs) if args.crs else None

    # Step 1: Parse CRS
    utm_zone = None
    northern = True
    epsg = None
    if crs_path and crs_path.exists():
        utm_zone, northern, crs_rot, crs_trans, epsg = parse_crs_json(crs_path)
        if utm_zone:
            print(f"CRS: UTM Zone {utm_zone}{'N' if northern else 'S'} ({epsg or 'no EPSG'})")
            print(f"  crsPose translation: [{crs_trans[0]:.2f}, {crs_trans[1]:.2f}, {crs_trans[2]:.2f}]")
        else:
            print("CRS: Local/project coordinates (no UTM zone detected)")
    else:
        print("No crs.json found — treating flightlog coordinates as local/metric")

    # Step 2: Read COLMAP camera positions
    print(f"\nReading COLMAP cameras from {images_txt}...")
    colmap_cameras = read_colmap_images_txt(images_txt)
    print(f"  Found {len(colmap_cameras)} COLMAP cameras")

    # Step 3: Read flightlog GPS positions
    print(f"Reading flightlog from {flightlog}...")
    geo_cameras = read_flightlog(flightlog, utm_zone, northern)
    print(f"  Found {len(geo_cameras)} flightlog positions")

    # Step 4: Match cameras
    print("Matching cameras...")
    matched_colmap, matched_geo, matched_names = match_cameras(colmap_cameras, geo_cameras)
    print(f"  Matched {len(matched_names)} cameras")

    if len(matched_names) < 3:
        print("\nERROR: Need at least 3 matching cameras.")
        print("  This usually means the image paths in images.txt don't match the flightlog paths.")
        print("  COLMAP paths (first 3):", list(colmap_cameras.keys())[:3])
        print("  Flightlog paths (first 3):", list(geo_cameras.keys())[:3])
        return

    # Step 5: Compute similarity transform
    print("\nComputing similarity transform (Umeyama's method)...")
    scale, R, t = compute_similarity_transform(matched_colmap, matched_geo)

    # Compute residuals
    transformed = (scale * (R @ matched_colmap.T)).T + t
    residuals = np.linalg.norm(transformed - matched_geo, axis=1)
    print(f"  Scale: {scale:.6f}")
    print(f"  Translation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
    print(f"  Residual mean: {residuals.mean():.4f} m")
    print(f"  Residual max:  {residuals.max():.4f} m")
    print(f"  Residual std:  {residuals.std():.4f} m")

    # Step 6: Apply to PLY
    print(f"\nApplying transform to {args.ply}...")
    n_points = apply_transform_to_ply(args.ply, args.output, scale, R, t)
    print(f"  Transformed {n_points} points")
    print(f"  Written to {args.output}")

    if epsg:
        print(f"\nCoordinate system: {epsg}")
    elif utm_zone:
        print(f"\nCoordinate system: UTM Zone {utm_zone}{'N' if northern else 'S'}")
    else:
        print("\nCoordinate system: Local project coordinates")


if __name__ == "__main__":
    main()
