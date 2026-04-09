#!/usr/bin/env python3
"""
VRAM Calculator for Gaussian Splatting training.

Estimates GPU memory requirements based on image count, resolution, and point count.
Uses empirical data from actual training runs on an NVIDIA L4 (22 GB VRAM).

Usage:
    python3 vram_calculator.py
    python3 vram_calculator.py --images 3765 --width 3000 --height 4096 --points 568000
    python3 vram_calculator.py --images 3765 --width 3000 --height 4096 --points 568000 --vram 22
"""

import argparse
import math


def estimate_vram(num_images, width, height, num_points, downscale=1):
    """Estimate VRAM usage in GB.

    Based on empirical measurements:
    - Images are stored as RGBA float32 tensors on GPU (4 bytes/channel x 4 channels = 16 bytes/pixel)
      Actually stored as uint8 initially but converted to float32 during training
    - Each image takes width * height * 4 * 4 bytes (RGBA float32) = 16 bytes/pixel
    - Gaussian model: ~200 bytes per point (position, covariance, SH coefficients, opacity)
    - Optimizer state: ~2x model size (Adam stores m and v per parameter)
    - Points typically grow 2-5x during densification
    """
    rw = width // downscale
    rh = height // downscale
    pixels_per_image = rw * rh

    # Image storage (RGBA float32 on GPU)
    image_bytes = num_images * pixels_per_image * 16  # 4 channels x 4 bytes
    image_gb = image_bytes / (1024**3)

    # Gaussian model (grows during training, estimate 3x initial points)
    estimated_final_points = num_points * 3
    model_bytes = estimated_final_points * 200  # ~200 bytes per Gaussian
    model_gb = model_bytes / (1024**3)

    # Optimizer state (~2x model)
    optimizer_gb = model_gb * 2

    # PyTorch overhead, CUDA context, etc.
    overhead_gb = 1.5

    total_gb = image_gb + model_gb + optimizer_gb + overhead_gb

    return {
        'downscale': downscale,
        'res': f'{rw}x{rh}',
        'pixels_per_image': pixels_per_image,
        'image_gb': image_gb,
        'model_gb': model_gb,
        'optimizer_gb': optimizer_gb,
        'overhead_gb': overhead_gb,
        'total_gb': total_gb,
    }


def max_images_for_vram(width, height, num_points, vram_gb, downscale=1):
    """Calculate max images that fit in given VRAM."""
    rw = width // downscale
    rh = height // downscale
    pixels_per_image = rw * rh

    # Reserve space for model, optimizer, overhead
    estimated_final_points = num_points * 3
    model_gb = (estimated_final_points * 200) / (1024**3)
    optimizer_gb = model_gb * 2
    overhead_gb = 1.5

    available_for_images = vram_gb - model_gb - optimizer_gb - overhead_gb
    if available_for_images <= 0:
        return 0

    bytes_per_image = pixels_per_image * 16
    max_imgs = int(available_for_images * (1024**3) / bytes_per_image)
    return max(0, max_imgs)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate VRAM requirements for Gaussian Splatting training',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=int, help='Number of images')
    parser.add_argument('--width', type=int, help='Source image width in pixels')
    parser.add_argument('--height', type=int, help='Source image height in pixels')
    parser.add_argument('--points', type=int, default=500000, help='Initial 3D points (default: 500000)')
    parser.add_argument('--vram', type=float, default=22, help='Available GPU VRAM in GB (default: 22)')
    args = parser.parse_args()

    if args.images and args.width and args.height:
        # Specific calculation
        print(f'\nDataset: {args.images} images at {args.width}x{args.height}, {args.points:,} initial points')
        print(f'GPU VRAM: {args.vram} GB')
        print()
        print(f'{"Downscale":<12} {"Resolution":<14} {"Images (GB)":<14} {"Model (GB)":<13} {"Optim (GB)":<13} {"Total (GB)":<13} {"Fits?"}')
        print('-' * 95)

        for r in [1, 2, 4, 8]:
            if args.width // r < 1 or args.height // r < 1:
                continue
            est = estimate_vram(args.images, args.width, args.height, args.points, r)
            fits = 'YES' if est['total_gb'] <= args.vram else 'NO'
            marker = '  <--' if fits == 'YES' else ''
            print(f'-r {r:<9} {est["res"]:<14} {est["image_gb"]:<14.1f} {est["model_gb"]:<13.1f} {est["optimizer_gb"]:<13.1f} {est["total_gb"]:<13.1f} {fits}{marker}')

        print()
        print(f'Max images at each resolution (for {args.vram} GB VRAM):')
        print()
        for r in [1, 2, 4, 8]:
            if args.width // r < 1 or args.height // r < 1:
                continue
            rw, rh = args.width // r, args.height // r
            max_imgs = max_images_for_vram(args.width, args.height, args.points, args.vram, r)
            print(f'  -r {r} ({rw}x{rh}): {max_imgs:,} images')

    else:
        # Interactive / example mode
        print('\nVRAM Calculator for Gaussian Splatting')
        print('=' * 60)

        datasets = [
            ('9045 (504x504 PNG)', 50420, 504, 504, 2500881),
            ('Dublin (504x252 JPG)', 15060, 504, 252, 588148),
            ('Dublin old-pos (3000x4096 JPG)', 3765, 3000, 4096, 568127),
        ]

        for name, images, w, h, points in datasets:
            print(f'\n--- {name}: {images} images, {points:,} points ---')
            print(f'{"Downscale":<12} {"Resolution":<14} {"Est. VRAM":<14} {"Fits 22GB?":<12} {"Max images (22GB)"}')
            print('-' * 75)
            for r in [1, 2, 4, 8]:
                rw, rh = w // r, h // r
                if rw < 1 or rh < 1:
                    continue
                est = estimate_vram(images, w, h, points, r)
                fits = 'YES' if est['total_gb'] <= 22 else 'NO'
                max_imgs = max_images_for_vram(w, h, points, 22, r)
                print(f'-r {r:<9} {rw}x{rh:<12} {est["total_gb"]:<14.1f} {fits:<12} {max_imgs:,}')

        print('\n\nUsage with specific dataset:')
        print('  python3 vram_calculator.py --images 3765 --width 3000 --height 4096 --points 568000')
        print('  python3 vram_calculator.py --images 3765 --width 3000 --height 4096 --points 568000 --vram 48')


if __name__ == '__main__':
    main()
