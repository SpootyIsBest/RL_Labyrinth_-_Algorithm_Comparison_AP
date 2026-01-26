import argparse
import glob
import os
import re
import sys
from typing import List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to console."""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


def load_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        raise ValueError(f"CSV file is empty: {path}")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    return data


def render_heatmap(data: np.ndarray, out_path: str, vmin: float, vmax: float, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_heatmap_3d(data: np.ndarray, out_path: str, vmin: float, vmax: float, cmap: str) -> None:
    """Optimized 3D rendering with lower DPI for faster processing."""
    # Reduce DPI for faster rendering (100 instead of 150)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    y_size, x_size = data.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    xx, yy = np.meshgrid(x, y)
    zz = data
    dx = dy = 1.0
    z0 = np.zeros_like(zz)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.get_cmap(cmap)(norm(zz))
    
    # Faster rendering with simplified shading
    ax.bar3d(xx.ravel(), yy.ravel(), z0.ravel(), dx, dy, zz.ravel(), 
             color=colors.reshape(-1, 4), shade=False, zsort="average")
    
    # Set 1:1 aspect ratio for x:y
    ax.set_box_aspect([1, 1, 0.5])  # 1:1 base, z is half for better viewing
    ax.set_xlim(0, x_size - 1)
    ax.set_ylim(0, y_size - 1)
    ax.set_zlim(vmin, vmax)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_video(frame_paths: List[str], out_path: str, fps: int) -> None:
    with imageio.get_writer(out_path, fps=max(fps, 1)) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))


def natural_key(path: str) -> List[object]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HeatMap CSV files to images and build a GIF/MP4.")
    parser.add_argument("--pattern", default="HeatMap*.csv", help="Glob pattern for CSV files.")
    parser.add_argument("--out-video", default="heatmap.mp4", help="Output video path (mp4).")
    parser.add_argument("--out-dir", default="heatmap_frames", help="Directory to write 2D frame images.")
    parser.add_argument("--out-dir-3d", default="heatmap_frames_3d", help="Directory to write 3D frame images.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for video.")
    parser.add_argument("--cmap", default="hot", help="Matplotlib colormap name.")
    args = parser.parse_args()

    print("="*60)
    print("HeatMap Video Generator")
    print("NOTE: 3D rendering uses matplotlib (CPU-based).")
    print("      For GPU-accelerated viewing, use heatmap_viewer.py")
    print("="*60)
    print()

    csv_paths = sorted(glob.glob(args.pattern), key=natural_key)
    if not csv_paths:
        raise SystemExit(f"No CSV files matched pattern: {args.pattern}")

    print(f"Found {len(csv_paths)} CSV files")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir_3d, exist_ok=True)

    print("\nLoading CSV files...")
    data_list = []
    for i, p in enumerate(csv_paths, 1):
        data_list.append(load_csv(p))
        print_progress_bar(i, len(csv_paths), prefix='Loading:', suffix=f'{i}/{len(csv_paths)} files')
    
    print("\nComputing value ranges...")
    vmin = min(np.nanmin(d) for d in data_list)
    vmax = max(np.nanmax(d) for d in data_list)
    print(f"Value range: {vmin:.2f} to {vmax:.2f}")

    frame_paths: List[str] = []
    frame_paths_3d: List[str] = []
    
    print("\nRendering 2D frames...")
    for idx, (path, data) in enumerate(zip(csv_paths, data_list), start=1):
        frame_name = f"frame_{idx:04d}.png"
        out_path_2d = os.path.join(args.out_dir, frame_name)
        render_heatmap(data, out_path_2d, vmin, vmax, args.cmap)
        frame_paths.append(out_path_2d)
        print_progress_bar(idx, len(csv_paths), prefix='2D Render:', suffix=f'{idx}/{len(csv_paths)} frames')
    
    print("\nRendering 3D frames...")
    for idx, (path, data) in enumerate(zip(csv_paths, data_list), start=1):
        frame_name = f"frame_{idx:04d}.png"
        out_path_3d = os.path.join(args.out_dir_3d, frame_name)
        render_heatmap_3d(data, out_path_3d, vmin, vmax, args.cmap)
        frame_paths_3d.append(out_path_3d)
        print_progress_bar(idx, len(csv_paths), prefix='3D Render:', suffix=f'{idx}/{len(csv_paths)} frames')

    base, ext = os.path.splitext(args.out_video)
    out_video_2d = args.out_video
    out_video_3d = f"{base}_3d{ext or '.mp4'}"
    
    print("\nBuilding 2D video...")
    build_video(frame_paths, out_video_2d, args.fps)
    print(f"✓ Created 2D video: {out_video_2d}")
    
    print("\nBuilding 3D video...")
    build_video(frame_paths_3d, out_video_3d, args.fps)
    print(f"✓ Created 3D video: {out_video_3d}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(frame_paths)} frames rendered")
    print(f"  2D frames: {args.out_dir}")
    print(f"  3D frames: {args.out_dir_3d}")
    print(f"{'='*60}")
    print(f"Created {len(frame_paths_3d)} 3D frames in '{args.out_dir_3d}'")
    print(f"2D video saved to '{out_video_2d}'")
    print(f"3D video saved to '{out_video_3d}'")


if __name__ == "__main__":
    main()
