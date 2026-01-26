import argparse
import glob
import re
from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force browser renderer for GPU acceleration via WebGL
pio.renderers.default = "browser"

# Performance threshold - use surface plot for grids larger than this (if --fast mode)
SURFACE_THRESHOLD = 100  # Only switch to surface if explicitly large


def load_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        raise ValueError(f"CSV file is empty: {path}")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    return data


def natural_key(path: str) -> List[object]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path)]


def create_bar_mesh(data: np.ndarray, cmap_name: str, vmin: float, vmax: float):
    """Create 3D bar mesh data efficiently for Plotly - each cell is a 1x1 square with height = value."""
    y_size, x_size = data.shape
    
    # Small gap between blocks so they're visually distinct
    gap = 0.02  # 2% gap between blocks
    
    # Create lists for vertices and faces
    vertices = []
    faces_i = []
    faces_j = []
    faces_k = []
    colors = []
    
    # Get colormap
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    vertex_count = 0
    
    for i in range(y_size):
        for j in range(x_size):
            val = data[i, j]
            if np.isnan(val):
                val = 0
            
            # Clamp negative values to small positive to show as flat squares
            if val < 0:
                val = 0.01
            
            # 8 vertices for a rectangular box (bar/block)
            # Each grid cell is EXACTLY 1x1 square with small gaps for separation
            x0, x1 = float(j) + gap, float(j + 1) - gap  # 1 unit wide with gap
            y0, y1 = float(i) + gap, float(i + 1) - gap  # 1 unit deep with gap
            z0, z1 = 0.0, float(val)                      # Height = cell value
            
            box_vertices = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom square
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top square
            ]
            vertices.extend(box_vertices)
            
            # 12 triangles (2 per face, 6 faces) - forms a complete rectangular block
            base = vertex_count
            box_faces = [
                # bottom square
                [base+0, base+1, base+2], [base+0, base+2, base+3],
                # top square
                [base+4, base+5, base+6], [base+4, base+6, base+7],
                # 4 sides
                [base+0, base+1, base+5], [base+0, base+5, base+4],  # front
                [base+2, base+3, base+7], [base+2, base+7, base+6],  # back
                [base+0, base+3, base+7], [base+0, base+7, base+4],  # left
                [base+1, base+2, base+6], [base+1, base+6, base+5],  # right
            ]
            
            for face in box_faces:
                faces_i.append(face[0])
                faces_j.append(face[1])
                faces_k.append(face[2])
            
            # Color based on value (value 56 gets color for 56, etc.)
            rgba = cmap(norm(val))
            color_str = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
            colors.extend([color_str] * 12)  # 12 triangular faces per block
            
            vertex_count += 8
    
    vertices = np.array(vertices)
    return vertices, faces_i, faces_j, faces_k, colors


def create_surface_plot(data: np.ndarray, vmin: float, vmax: float, cmap: str):
    """Create efficient 3D surface plot for large grids."""
    y_size, x_size = data.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    
    return go.Surface(
        z=data,
        x=x,
        y=y,
        colorscale=cmap,
        cmin=vmin,
        cmax=vmax,
        colorbar=dict(title="Value"),
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5, specular=0.2, fresnel=0.2),
    )


def create_figure(data: np.ndarray, vmin: float, vmax: float, cmap: str, frame_idx: int, total_frames: int, use_bars: bool = True):
    """Create Plotly figure - uses 3D rectangular bars (skycrapers) by default."""
    y_size, x_size = data.shape
    max_dim = max(y_size, x_size)
    
    # Use bars unless explicitly disabled or grid is extremely large
    if use_bars and max_dim <= SURFACE_THRESHOLD:
        plot_type = "3D Bars"
        vertices, faces_i, faces_j, faces_k, colors = create_bar_mesh(data, cmap, vmin, vmax)
        
        # Create mesh with visible edges (like Minecraft blocks)
        trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces_i,
            j=faces_j,
            k=faces_k,
            facecolor=colors,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.3, fresnel=0.2),
            lightposition=dict(x=100, y=100, z=1000),
            # Add visible edges to each block
            contour=dict(
                show=True,
                color='rgba(50, 50, 50, 0.8)',  # Dark grey edges
                width=2
            )
        )
    else:
        plot_type = "Surface"
        trace = create_surface_plot(data, vmin, vmax, cmap)
    
    fig = go.Figure(data=[trace])
    
    # Calculate aspect ratio to ensure 1:1 for x:y plane (square grid cells)
    # This makes each bar look like a proper rectangular block
    max_z = max(vmax - vmin, 1)
    z_ratio = min(0.6, max_z / max(x_size, y_size))  # Scale Z for better viewing
    aspectratio = dict(x=1, y=1, z=z_ratio)
    
    fig.update_layout(
        title=f"Interactive 3D Heatmap - Frame {frame_idx + 1}/{total_frames}<br><sub>{plot_type} | {x_size}×{y_size} grid | WebGL GPU Rendering</sub>",
        scene=dict(
            xaxis=dict(title="X", range=[0, x_size]),
            yaxis=dict(title="Y", range=[0, y_size]),
            zaxis=dict(title="Value", range=[vmin, vmax]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='manual',
            aspectratio=aspectratio
        ),
        width=1200,
        height=900,
        template="plotly_dark",
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
    )
    
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D HeatMap viewer with rotation.")
    parser.add_argument("--pattern", default="HeatMap*.csv", help="Glob pattern for CSV files.")
    parser.add_argument("--cmap", default="hot", help="Matplotlib colormap name or Plotly colorscale (hot, viridis, plasma, etc.).")
    parser.add_argument("--initial", type=int, default=0, help="Initial frame index to display.")
    parser.add_argument("--frame", type=int, default=None, help="Show only a specific frame (0-indexed).")
    parser.add_argument("--fast", action="store_true", help="Use fast surface mode instead of 3D bars for large grids.")
    args = parser.parse_args()

    print("=" * 60)
    print("3D HeatMap Viewer - Optimized for Large Grids")
    print("=" * 60)
    
    print(f"\n[1/4] Searching for CSV files matching: {args.pattern}")
    csv_paths = sorted(glob.glob(args.pattern), key=natural_key)
    if not csv_paths:
        raise SystemExit(f"ERROR: No CSV files matched pattern: {args.pattern}")

    print(f"      Found {len(csv_paths)} CSV files")
    
    # Lazy loading - only load what we need
    print(f"\n[2/4] Sampling data to compute value ranges...")
    # Sample a few files to get value range (much faster than loading all)
    sample_indices = [0, len(csv_paths)//2, len(csv_paths)-1] if len(csv_paths) > 1 else [0]
    sample_data = [load_csv(csv_paths[i]) for i in sample_indices]
    vmin = min(np.nanmin(d) for d in sample_data)
    vmax = max(np.nanmax(d) for d in sample_data)
    print(f"      Estimated value range: {vmin:.2f} to {vmax:.2f}")
    print(f"      Grid size: {sample_data[0].shape}")

    # Determine which frame to show
    if args.frame is not None:
        current_idx = min(args.frame, len(csv_paths) - 1)
    else:
        current_idx = min(args.initial, len(csv_paths) - 1)

    # Load only the current frame
    print(f"\n[3/4] Loading frame {current_idx + 1}/{len(csv_paths)}...")
    print(f"      File: {csv_paths[current_idx]}")
    current_data = load_csv(csv_paths[current_idx])
    print(f"      ✓ Loaded {current_data.shape[0]}×{current_data.shape[1]} grid")
    
    # Create initial figure
    print(f"\n[4/4] Generating 3D visualization...")
    y_size, x_size = current_data.shape
    max_dim = max(y_size, x_size)
    
    use_bars = not args.fast or max_dim <= SURFACE_THRESHOLD
    
    if use_bars:
        print(f"      Using 3D Rectangular Bars (skycraper style: {x_size}×{y_size} = {x_size * y_size} blocks)")
    else:
        print(f"      Using Fast Surface Plot (grid size: {x_size}×{y_size})")
    
    fig = create_figure(current_data, vmin, vmax, args.cmap, current_idx, len(csv_paths), use_bars=use_bars)
    print(f"      Visualization generated successfully!")
    
    # Add buttons for frame navigation if multiple files
    if len(csv_paths) > 1 and args.frame is None:
        print(f"\n      NOTE: With {len(csv_paths)} frames, use --frame <N> to view specific frames.")
        print(f"            Example: python heatmap_viewer.py --frame 0")
        print(f"            Frame range: 0 to {len(csv_paths) - 1}")
    else:
        print(f"\n      Single frame mode")
    
    # Show in browser with GPU acceleration via WebGL
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'heatmap_3d_frame_{current_idx}',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    print("\n" + "=" * 60)
    print("Opening in browser with GPU-accelerated WebGL rendering...")
    print("=" * 60)
    print("\nINSTRUCTIONS:")
    print("  • Click and drag to ROTATE the view")
    print("  • Scroll to ZOOM in/out")
    print("  • Right-click and drag to PAN")
    if len(csv_paths) > 1:
        print("\nNAVIGATION:")
        print(f"  • Currently viewing frame {current_idx + 1}/{len(csv_paths)}")
        print(f"  • To view other frames, use: --frame <N> (0 to {len(csv_paths) - 1})")
        print(f"  • Example: python heatmap_viewer.py --frame {min(current_idx + 1, len(csv_paths) - 1)}")
    print("\nIf the browser doesn't open automatically, check the terminal")
    print("for a URL and open it manually in your browser.")
    print("=" * 60 + "\n")
    
    fig.show(config=config)
    
    print("\nViewer opened in browser. Close the browser tab when done.")


if __name__ == "__main__":
    main()
