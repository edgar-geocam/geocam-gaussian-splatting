#!/usr/bin/env python3
"""
Flatten a COLMAP scene with deep subdirectory image paths into a flat layout
compatible with 3DGS and 2DGS (which use os.path.basename() on image paths).

Handles multi-lens datasets where the same filename appears under different
lens/batch subdirectories by converting the full relative path into a flat
name (replacing '/' with '_').

Images are symlinked (not copied) to save disk space. The symlinks use
absolute paths so they resolve correctly inside Docker containers when
the source directory is bind-mounted at the same absolute path.

Optionally splits the scene into consecutive chunks for large datasets.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_images_txt(path):
    """Parse COLMAP images.txt, return list of (line_pair, image_path) tuples.

    Each entry in images.txt is two lines:
      Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      Line 2: POINTS2D[] as (X, Y, POINT3D_ID) ...
    """
    entries = []
    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue
        # This is an image line (line 1 of the pair)
        parts = line.split()
        image_path = parts[-1]  # Last field is the image name/path
        points_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        entries.append((line, points_line, image_path))
        i += 2

    return entries


def parse_points3d_txt(path):
    """Parse COLMAP points3D.txt, return list of lines and a set of point IDs
    referenced by a given set of image entries."""
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            lines.append(line)
    return lines


def get_referenced_point_ids(entries):
    """Get set of POINT3D_IDs referenced by the given image entries."""
    point_ids = set()
    for _, points_line, _ in entries:
        parts = points_line.split()
        # Every 3rd value starting at index 2 is a POINT3D_ID
        for j in range(2, len(parts), 3):
            pid = int(parts[j])
            if pid != -1:
                point_ids.add(pid)
    return point_ids


def flatten_path(image_path):
    """Convert 'flight/lens/batch/file.png' to 'flight_lens_batch_file.png'."""
    return image_path.replace('/', '_')


def copy_metadata(source_dir, output_dir):
    """Copy georeferencing metadata files (flightlog.txt, info.json, crs.json) if present."""
    import shutil
    metadata_files = ['flightlog.txt', 'info.json', 'crs.json']
    copied = []
    for fname in metadata_files:
        src = Path(source_dir) / fname
        if src.exists():
            shutil.copy2(src, Path(output_dir) / fname)
            copied.append(fname)
    if copied:
        print(f'  Metadata: copied {", ".join(copied)}')


def write_chunk(entries, all_points_lines, images_dir, output_dir, sparse_source):
    """Write a single chunk (or full scene) to output_dir."""
    output_dir = Path(output_dir)
    images_out = output_dir / 'images'
    sparse_out = output_dir / 'sparse' / '0'
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    # Copy cameras.txt as-is
    import shutil
    shutil.copy2(sparse_source / 'cameras.txt', sparse_out / 'cameras.txt')

    # Write flattened images.txt and create symlinks
    created = 0
    skipped = 0
    with open(sparse_out / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID) ...\n')
        f.write(f'# Number of images: {len(entries)}\n')
        for img_line, points_line, original_path in entries:
            flat_name = flatten_path(original_path)
            # Rewrite the image line with the flat name
            parts = img_line.split()
            parts[-1] = flat_name
            f.write(' '.join(parts) + '\n')
            f.write(points_line + '\n')

            # Create symlink
            src = Path(images_dir) / original_path
            dst = images_out / flat_name
            if not dst.exists():
                if src.exists():
                    os.symlink(src.resolve(), dst)
                    created += 1
                else:
                    skipped += 1
                    print(f'  WARNING: source image not found: {src}')

    # Filter points3D.txt to only referenced points
    referenced_ids = get_referenced_point_ids(entries)
    kept = 0
    with open(sparse_out / 'points3D.txt', 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...\n')
        for line in all_points_lines:
            point_id = int(line.split()[0])
            if point_id in referenced_ids:
                f.write(line)
                kept += 1

    print(f'  Images: {created} symlinked, {skipped} skipped (missing)')
    print(f'  Points: {kept}/{len(all_points_lines)} retained')


def main():
    parser = argparse.ArgumentParser(description='Flatten a COLMAP scene for 3DGS/2DGS compatibility')
    parser.add_argument('--source', required=True, help='Path to source scene (containing sparse/ directory)')
    parser.add_argument('--images-dir', required=True, help='Path to images directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num-chunks', type=int, default=0, help='Split into N consecutive chunks (0 = no chunking)')
    parser.add_argument('--chunk', type=int, default=-1, help='Only create this specific chunk (0-indexed)')
    parser.add_argument('--every', type=int, default=1, help='Take every Nth image (1 = all images)')
    parser.add_argument('--filter', type=str, default=None, help='CSV file with a Name column of image paths to include')
    args = parser.parse_args()

    source = Path(args.source)
    images_dir = Path(args.images_dir)

    # Find sparse directory
    sparse_dir = source / 'sparse'
    if (sparse_dir / '0').is_dir():
        sparse_dir = sparse_dir / '0'

    print(f'Source: {source}')
    print(f'Images: {images_dir}')
    print(f'Sparse: {sparse_dir}')

    # Parse COLMAP files
    print('Parsing images.txt...')
    entries = parse_images_txt(sparse_dir / 'images.txt')
    print(f'  Found {len(entries)} images')

    print('Parsing points3D.txt...')
    all_points = parse_points3d_txt(sparse_dir / 'points3D.txt')
    print(f'  Found {len(all_points)} points')

    # Filter by CSV file if requested
    if args.filter:
        import csv
        filter_path = Path(args.filter)
        with open(filter_path, 'r') as f:
            reader = csv.DictReader(f)
            allowed = set(row['Name'] for row in reader)
        before = len(entries)
        entries = [e for e in entries if e[2] in allowed]
        print(f'  Filtered to {len(entries)}/{before} images (from {filter_path.name})')

    # Subsample if requested
    if args.every > 1:
        entries = entries[::args.every]
        print(f'  Subsampled to {len(entries)} images (every {args.every}th)')

    if args.num_chunks > 0:
        # Split into chunks
        chunk_size = len(entries) // args.num_chunks
        remainder = len(entries) % args.num_chunks

        chunks = []
        start = 0
        for i in range(args.num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(entries[start:end])
            start = end

        indices = range(args.num_chunks) if args.chunk < 0 else [args.chunk]
        for i in indices:
            chunk_dir = Path(args.output) / f'chunk-{i}'
            print(f'\nWriting chunk {i} ({len(chunks[i])} images) -> {chunk_dir}')
            write_chunk(chunks[i], all_points, images_dir, chunk_dir, sparse_dir)
            copy_metadata(source, chunk_dir)
    else:
        # Single output
        print(f'\nWriting flattened scene -> {args.output}')
        write_chunk(entries, all_points, images_dir, args.output, sparse_dir)
        copy_metadata(source, args.output)

    print('\nDone.')


if __name__ == '__main__':
    main()
