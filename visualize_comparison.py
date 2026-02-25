"""
Comparison Visualization Script
================================
Reads a comparison results JSON (from default_comparison_results structure)
placed in the ComparisonVisualization/ folder and generates charts.

Usage:
    python visualize_comparison.py                     # picks the newest JSON in ComparisonVisualization/
    python visualize_comparison.py my_results.json     # specific file inside the folder
    python visualize_comparison.py --save              # save PNGs instead of showing interactively
    python visualize_comparison.py --all               # generate every chart type
"""

import json
import sys
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

INPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComparisonVisualization")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComparisonVisualization", "charts")

# Color palette for algorithms (consistent across all charts)
ALGO_COLORS = {
    "Q-Learning":    "#1565C0",   # deep blue
    "SARSA":         "#E65100",   # deep orange
    "BFS":           "#2E7D32",   # deep green
    "Wall Follower": "#6A1B9A",   # deep purple
    "Random Walk":   "#C62828",   # deep red
    "Greedy":        "#00838F",   # deep teal
}

TYPE_COLORS = {
    "RL":    "#1565C0",
    "NonRL": "#2E7D32",
}

BACKGROUND_COLOR = "#FFFFFF"
PANEL_COLOR      = "#F5F5F5"
TEXT_COLOR        = "#212121"
GRID_COLOR       = "#BDBDBD"
ACCENT_COLOR     = "#D32F2F"
BAR_EDGE_COLOR   = "#424242"   # dark edge for bars on white background

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_comparison_json(filepath=None):
    """Load comparison JSON. If no filepath given, pick the newest file in INPUT_FOLDER."""
    if filepath and os.path.isabs(filepath):
        path = filepath
    elif filepath:
        path = os.path.join(INPUT_FOLDER, filepath)
    else:
        # Find newest JSON in folder
        json_files = list(Path(INPUT_FOLDER).glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {INPUT_FOLDER}")
            print("Place a comparison results JSON in that folder and try again.")
            sys.exit(1)
        path = str(max(json_files, key=os.path.getmtime))
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    
    with open(path, "r") as f:
        data = json.load(f)
    
    print(f"Loaded: {path}")
    return data, path


def get_completed_algorithms(data):
    """Return dict of {name: algo_data} for algorithms that actually ran."""
    algos = {}
    for name, info in data.get("algorithms", {}).items():
        if info.get("status") == "completed":
            algos[name] = info
    return algos


def get_color(name):
    return ALGO_COLORS.get(name, "#888888")


def apply_style(fig, axes):
    """Apply clean light theme to figure and all axes."""
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in np.array(axes).flat:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.4, linestyle="-", linewidth=0.5)


# ─────────────────────────────────────────────
# Chart 1: Path Length Comparison (all algos)
# ─────────────────────────────────────────────

def chart_path_lengths(data, algos, save=False):
    """Bar chart comparing path lengths across all algorithms with optimal baseline."""
    optimal = data["metadata"].get("optimal_path_length", 0)
    
    names = []
    lengths = []
    colors = []
    for name, info in algos.items():
        cm = info["common_metrics"]
        if cm["path_found"]:
            names.append(name)
            lengths.append(cm["path_length"])
            colors.append(get_color(name))
    
    if not names:
        print("  [Skip] No algorithms found a path — skipping path length chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    
    bars = ax.bar(names, lengths, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    
    # Optimal path line
    if optimal > 0:
        ax.axhline(y=optimal, color=ACCENT_COLOR, linestyle="--", linewidth=2, label=f"Optimal ({optimal} steps)", zorder=4)
        ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    
    # Value labels on bars
    for bar, val in zip(bars, lengths):
        extra = val - optimal if optimal > 0 else 0
        label = f"{val}"
        if extra > 0:
            label += f"\n(+{extra})"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", color=TEXT_COLOR, fontsize=9, fontweight="bold")
    
    ax.set_ylabel("Path Length (steps)", fontsize=12)
    ax.set_title("Path Length Comparison — All Algorithms", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(lengths) * 1.25)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "path_lengths", save)


# ─────────────────────────────────────────────
# Chart 2: Execution Time Comparison
# ─────────────────────────────────────────────

def chart_execution_times(data, algos, save=False):
    """Bar chart comparing execution/training times."""
    names = []
    times = []
    colors = []
    for name, info in algos.items():
        cm = info["common_metrics"]
        if cm["path_found"]:
            names.append(name)
            times.append(cm["time_to_solution_seconds"])
            colors.append(get_color(name))
    
    if not names:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    
    bars = ax.bar(names, times, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    
    for bar, val in zip(bars, times):
        label = f"{val:.4f}s" if val < 1 else f"{val:.2f}s"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                label, ha="center", va="bottom", color=TEXT_COLOR, fontsize=9, fontweight="bold")
    
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Execution / Training Time — All Algorithms", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(times) * 1.3)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "execution_times", save)


# ─────────────────────────────────────────────
# Chart 3: RL Success Rate Comparison
# ─────────────────────────────────────────────

def chart_rl_success_rates(data, algos, save=False):
    """Side-by-side bars for RL algorithm success rates."""
    rl_algos = {n: a for n, a in algos.items() if a["type"] == "RL"}
    if not rl_algos:
        return
    
    names = list(rl_algos.keys())
    success_rates = [rl_algos[n]["performance_metrics"]["success_rate_percent"] for n in names]
    first_finds = [rl_algos[n]["performance_metrics"]["first_find_episode"] for n in names]
    colors = [get_color(n) for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    apply_style(fig, [ax1, ax2])
    
    # Success rate bars
    bars1 = ax1.bar(names, success_rates, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    for bar, val in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_title("Success Rate", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 110)
    
    # First find episode bars
    display_finds = [f if f > 0 else 0 for f in first_finds]
    bars2 = ax2.bar(names, display_finds, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    for bar, val in zip(bars2, first_finds):
        label = str(val) if val > 0 else "Never"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(display_finds)*0.02,
                label, ha="center", va="bottom", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Episode #", fontsize=12)
    ax2.set_title("First Successful Episode", fontsize=13, fontweight="bold")
    
    fig.suptitle("RL Algorithm Performance", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, "rl_success_rates", save)


# ─────────────────────────────────────────────
# Chart 4: Learning Curves (RL)
# ─────────────────────────────────────────────

def chart_learning_curves(data, algos, save=False):
    """Line chart of RL learning curves (success rate over episode windows)."""
    rl_algos = {n: a for n, a in algos.items() if a["type"] == "RL"}
    if not rl_algos:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)
    
    has_data = False
    for name, info in rl_algos.items():
        curve = info.get("learning_curve", {})
        points = curve.get("data_points", [])
        if not points:
            continue
        has_data = True
        
        episodes = [(p["episode_range"][0] + p["episode_range"][1]) / 2 for p in points]
        rates = [p["success_rate_percent"] for p in points]
        
        ax.plot(episodes, rates, color=get_color(name), linewidth=2, label=name, marker="o", markersize=3, zorder=3)
        
        # Mark convergence point
        conv_ep = curve.get("convergence_episode", -1)
        if conv_ep > 0:
            # Find the rate at convergence
            conv_rate = None
            for p in points:
                if p["episode_range"][0] <= conv_ep <= p["episode_range"][1]:
                    conv_rate = p["success_rate_percent"]
                    break
            if conv_rate is not None:
                ax.scatter([conv_ep], [conv_rate], color=get_color(name), s=120, zorder=5,
                          edgecolors=BAR_EDGE_COLOR, linewidths=2, marker="D")
                ax.annotate(f"Converged\nEp. {conv_ep}", (conv_ep, conv_rate),
                           textcoords="offset points", xytext=(15, 10),
                           color=get_color(name), fontsize=9, fontweight="bold",
                           arrowprops=dict(arrowstyle="->", color=get_color(name), lw=1.5))
    
    if not has_data:
        plt.close(fig)
        return
    
    # Threshold line
    threshold = 70.0
    ax.axhline(y=threshold, color=ACCENT_COLOR, linestyle=":", linewidth=1.5,
              label=f"Convergence threshold ({threshold}%)", alpha=0.7)
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("RL Learning Curves — Success Rate Over Training", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(-5, 105)
    ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    
    plt.tight_layout()
    _save_or_show(fig, "learning_curves", save)


# ─────────────────────────────────────────────
# Chart 5: Exploration Heatmaps (RL)
# ─────────────────────────────────────────────

def chart_heatmaps(data, algos, save=False):
    """Side-by-side exploration heatmaps for RL algorithms."""
    rl_algos = {n: a for n, a in algos.items() if a["type"] == "RL" and a.get("exploration_data", {}).get("heatmap")}
    if not rl_algos:
        return
    
    n = len(rl_algos)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    apply_style(fig, axes)
    
    for ax, (name, info) in zip(axes, rl_algos.items()):
        heatmap = np.array(info["exploration_data"]["heatmap"])
        
        # Log scale for better visibility
        heatmap_log = np.log1p(heatmap)
        
        im = ax.imshow(heatmap_log, cmap="YlOrRd", aspect="equal", interpolation="nearest")
        ax.set_title(name, fontsize=13, fontweight="bold", color=get_color(name))
        
        # Mark start and goal
        meta = data["metadata"]
        start = meta.get("start_position", [0, 0])
        goal = meta.get("goal_position", [0, 0])
        ax.plot(start[0], start[1], "ks", markersize=10, label="Start")
        ax.plot(goal[0], goal[1], "k*", markersize=14, label="Goal")
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("log(visits + 1)", color=TEXT_COLOR, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
        
        ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                 fontsize=8, loc="upper right")
        
        # Stats annotation
        expl = info["exploration_data"]
        stats_text = f"Coverage: {expl['exploration_coverage_percent']:.0f}%\nTotal visits: {expl['total_state_visits']}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
               color=TEXT_COLOR, verticalalignment="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor=BACKGROUND_COLOR, alpha=0.85, edgecolor=GRID_COLOR))
    
    fig.suptitle("Exploration Heatmaps — State Visit Frequency", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, "heatmaps", save)


# ─────────────────────────────────────────────
# Chart 6: Path Efficiency & Extra Steps
# ─────────────────────────────────────────────

def chart_efficiency(data, algos, save=False):
    """Grouped bar chart: efficiency ratio + extra steps for all algorithms."""
    optimal = data["metadata"].get("optimal_path_length", 1)
    
    names = []
    efficiencies = []
    extra_steps = []
    colors = []
    
    for name, info in algos.items():
        cm = info["common_metrics"]
        if cm["path_found"]:
            names.append(name)
            efficiencies.append(cm["path_efficiency"] * 100)  # as percentage
            extra_steps.append(cm["extra_steps_vs_optimal"])
            colors.append(get_color(name))
    
    if not names:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    apply_style(fig, [ax1, ax2])
    
    # Efficiency (higher = better, 100% = optimal)
    bars1 = ax1.bar(names, efficiencies, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    ax1.axhline(y=100, color=ACCENT_COLOR, linestyle="--", linewidth=2, label="Optimal (100%)", zorder=4)
    for bar, val in zip(bars1, efficiencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Efficiency (%)", fontsize=12)
    ax1.set_title("Path Efficiency (100% = optimal)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(max(efficiencies) * 1.2, 110))
    ax1.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    plt.sca(ax1)
    plt.xticks(rotation=15, ha="right")
    
    # Extra steps (lower = better)
    bars2 = ax2.bar(names, extra_steps, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    for bar, val in zip(bars2, extra_steps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(extra_steps)*0.02 + 0.2,
                str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Extra Steps", fontsize=12)
    ax2.set_title(f"Extra Steps Beyond Optimal ({optimal})", fontsize=13, fontweight="bold")
    plt.sca(ax2)
    plt.xticks(rotation=15, ha="right")
    
    fig.suptitle("Path Efficiency Analysis", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, "efficiency", save)


# ─────────────────────────────────────────────
# Chart 7: Exploration Coverage
# ─────────────────────────────────────────────

def chart_exploration_coverage(data, algos, save=False):
    """Bar chart of exploration coverage percentage for all algorithms."""
    names = []
    coverages = []
    colors = []
    
    for name, info in algos.items():
        if info.get("status") != "completed":
            continue
        expl = info.get("exploration_data", {})
        cov = expl.get("exploration_coverage_percent", 0)
        names.append(name)
        coverages.append(cov)
        colors.append(get_color(name))
    
    if not names:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    
    bars = ax.barh(names, coverages, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Exploration Coverage (%)", fontsize=12)
    ax.set_title("Maze Exploration Coverage — All Algorithms", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 110)
    ax.axvline(x=100, color=ACCENT_COLOR, linestyle=":", linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    _save_or_show(fig, "exploration_coverage", save)


# ─────────────────────────────────────────────
# Chart 8: Summary Dashboard
# ─────────────────────────────────────────────

def chart_dashboard(data, algos, save=False):
    """Multi-panel dashboard with key metrics at a glance."""
    meta = data["metadata"]
    summary = data.get("summary", {})
    
    completed = {n: a for n, a in algos.items() if a["common_metrics"]["path_found"]}
    if not completed:
        return
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)
    
    # ── Title area ──
    fig.suptitle(
        f"Algorithm Comparison Dashboard — {meta.get('maze_name', '?')} "
        f"({meta['maze_size'][0]}×{meta['maze_size'][1]})",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98
    )
    
    # ── Panel 1: Path lengths bar chart (top-left) ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    apply_style(fig, ax1)
    optimal = meta.get("optimal_path_length", 0)
    names = list(completed.keys())
    lengths = [completed[n]["common_metrics"]["path_length"] for n in names]
    colors = [get_color(n) for n in names]
    bars = ax1.bar(names, lengths, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    if optimal > 0:
        ax1.axhline(y=optimal, color=ACCENT_COLOR, linestyle="--", linewidth=2, zorder=4)
    for bar, val in zip(bars, lengths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=8, fontweight="bold")
    ax1.set_title("Path Lengths", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8, rotation=15)
    
    # ── Panel 2: Summary text box (top-right) ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_COLOR)
    ax2.axis("off")
    
    info_lines = [
        f"Maze: {meta.get('maze_name', '?')}",
        f"Size: {meta['maze_size'][0]}×{meta['maze_size'][1]} ({meta['total_maze_cells']} cells)",
        f"Optimal path: {optimal} steps",
        f"Date: {meta.get('comparison_date', '?')}",
        "",
        f"Best RL: {summary.get('best_rl_by_success_rate', {}).get('name', 'N/A')}",
        f"  Success: {summary.get('best_rl_by_success_rate', {}).get('success_rate_percent', 0):.1f}%",
        f"Fastest: {summary.get('fastest_overall', {}).get('name', 'N/A')}",
        f"  Time: {summary.get('fastest_overall', {}).get('time_seconds', 0):.4f}s",
        f"Shortest path: {summary.get('shortest_path_found', {}).get('name', 'N/A')}",
        f"  Length: {summary.get('shortest_path_found', {}).get('path_length', 0)} steps",
    ]
    text = "\n".join(info_lines)
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=9, color=TEXT_COLOR,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR))
    
    # ── Panel 3: Learning curves (middle row, spans 2 cols) ──
    ax3 = fig.add_subplot(gs[1, 0:2])
    apply_style(fig, ax3)
    rl_algos = {n: a for n, a in algos.items() if a["type"] == "RL"}
    has_curve = False
    for name, info in rl_algos.items():
        points = info.get("learning_curve", {}).get("data_points", [])
        if points:
            episodes = [(p["episode_range"][0] + p["episode_range"][1]) / 2 for p in points]
            rates = [p["success_rate_percent"] for p in points]
            ax3.plot(episodes, rates, color=get_color(name), linewidth=2, label=name, zorder=3)
            has_curve = True
    if has_curve:
        ax3.axhline(y=70, color=ACCENT_COLOR, linestyle=":", linewidth=1, alpha=0.5)
        ax3.set_title("RL Learning Curves", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Episode", fontsize=9)
        ax3.set_ylabel("Success %", fontsize=9)
        ax3.set_ylim(-5, 105)
        ax3.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No RL learning curve data", ha="center", va="center",
                color=TEXT_COLOR, fontsize=11, transform=ax3.transAxes)
    
    # ── Panel 4: Efficiency (middle-right) ──
    ax4 = fig.add_subplot(gs[1, 2])
    apply_style(fig, ax4)
    effs = [completed[n]["common_metrics"]["path_efficiency"] * 100 for n in names]
    bars4 = ax4.barh(names, effs, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    ax4.axvline(x=100, color=ACCENT_COLOR, linestyle="--", linewidth=1.5)
    ax4.set_title("Path Efficiency %", fontsize=11, fontweight="bold")
    ax4.set_xlim(0, max(max(effs) * 1.15, 110))
    ax4.tick_params(axis="y", labelsize=8)
    
    # ── Panel 5: Execution times (bottom-left) ──
    ax5 = fig.add_subplot(gs[2, 0])
    apply_style(fig, ax5)
    times = [completed[n]["common_metrics"]["time_to_solution_seconds"] for n in names]
    bars5 = ax5.bar(names, times, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    ax5.set_title("Execution Time", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Seconds", fontsize=9)
    ax5.tick_params(axis="x", labelsize=7, rotation=25)
    
    # ── Panel 6: NonRL nodes explored (bottom-middle) ──
    ax6 = fig.add_subplot(gs[2, 1])
    apply_style(fig, ax6)
    nonrl_algos = {n: a for n, a in algos.items() if a["type"] == "NonRL" and a.get("status") == "completed"}
    if nonrl_algos:
        nn = list(nonrl_algos.keys())
        nodes = [nonrl_algos[n].get("execution_metrics", {}).get("nodes_explored", 0) for n in nn]
        nc = [get_color(n) for n in nn]
        ax6.bar(nn, nodes, color=nc, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
        ax6.set_title("NonRL: Nodes Explored", fontsize=11, fontweight="bold")
        ax6.tick_params(axis="x", labelsize=7, rotation=25)
    else:
        ax6.text(0.5, 0.5, "No NonRL data", ha="center", va="center", color=TEXT_COLOR, transform=ax6.transAxes)
    
    # ── Panel 7: RL success rates (bottom-right) ──
    ax7 = fig.add_subplot(gs[2, 2])
    apply_style(fig, ax7)
    if rl_algos:
        rn = list(rl_algos.keys())
        sr = [rl_algos[n]["performance_metrics"]["success_rate_percent"] for n in rn]
        rc = [get_color(n) for n in rn]
        ax7.bar(rn, sr, color=rc, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
        ax7.set_title("RL Success Rate %", fontsize=11, fontweight="bold")
        ax7.set_ylim(0, 110)
        ax7.tick_params(axis="x", labelsize=8)
    else:
        ax7.text(0.5, 0.5, "No RL data", ha="center", va="center", color=TEXT_COLOR, transform=ax7.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_or_show(fig, "dashboard", save)


# ─────────────────────────────────────────────
# Chart 9: NonRL Algorithm Characteristics Table
# ─────────────────────────────────────────────

def chart_nonrl_comparison_table(data, algos, save=False):
    """Visual table comparing NonRL algorithm properties and results."""
    nonrl = {n: a for n, a in algos.items() if a["type"] == "NonRL"}
    if not nonrl:
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")
    
    # Build table data
    headers = ["Algorithm", "Path\nLength", "Time (s)", "Nodes\nExplored", "Coverage %",
               "Optimal?", "Deterministic?", "Uses\nHeuristic?"]
    
    rows = []
    row_colors = []
    for name, info in nonrl.items():
        cm = info["common_metrics"]
        em = info.get("execution_metrics", {})
        pa = info.get("path_analysis", {})
        ex = info.get("exploration_data", {})
        ch = info.get("algorithm_characteristics", {})
        
        rows.append([
            name,
            str(cm["path_length"]) if cm["path_found"] else "—",
            f"{cm['time_to_solution_seconds']:.5f}" if cm["path_found"] else "—",
            str(em.get("nodes_explored", "?")),
            f"{ex.get('exploration_coverage_percent', 0):.0f}%",
            "✓" if pa.get("is_optimal_path", False) else "✗",
            "✓" if ch.get("is_deterministic", False) else "✗",
            "✓" if ch.get("uses_heuristic", False) else "✗",
        ])
        row_colors.append(get_color(name))
    
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
        else:
            cell.set_facecolor(BACKGROUND_COLOR)
            cell.set_text_props(color=TEXT_COLOR)
            if col == 0:
                cell.set_text_props(color=row_colors[row - 1], fontweight="bold")
    
    ax.set_title("Non-RL Algorithm Comparison", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)
    plt.tight_layout()
    _save_or_show(fig, "nonrl_table", save)


# ─────────────────────────────────────────────
# Chart 10: Radar Chart — Multi-Metric Overview
# ─────────────────────────────────────────────

def chart_radar(data, algos, save=False):
    """Radar/spider chart comparing normalized metrics across all successful algorithms."""
    completed = {n: a for n, a in algos.items() if a["common_metrics"]["path_found"]}
    if len(completed) < 2:
        return
    
    optimal = max(data["metadata"].get("optimal_path_length", 1), 1)
    
    # Metrics to compare (name, extractor, higher_is_better)
    metric_defs = [
        ("Path\nEfficiency", lambda n, a: a["common_metrics"]["path_efficiency"] * 100, True),
        ("Speed\n(inv. time)", lambda n, a: 1.0 / max(a["common_metrics"]["time_to_solution_seconds"], 0.0001), True),
        ("Low Extra\nSteps", lambda n, a: max(0, 100 - a["common_metrics"]["extra_steps_vs_optimal"] * 5), True),
    ]
    
    # Add RL-specific metric (normalized so NonRL get 100%)
    has_rl = any(a["type"] == "RL" for a in completed.values())
    if has_rl:
        metric_defs.append(
            ("RL Success\nRate", lambda n, a: a.get("performance_metrics", {}).get("success_rate_percent", 100) if a["type"] == "RL" else 100, True)
        )
    
    # Add exploration coverage
    metric_defs.append(
        ("Exploration\nCoverage", lambda n, a: a.get("exploration_data", {}).get("exploration_coverage_percent", 0), True)
    )
    
    labels = [m[0] for m in metric_defs]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    for name, info in completed.items():
        raw_values = []
        for _, extractor, _ in metric_defs:
            try:
                raw_values.append(extractor(name, info))
            except:
                raw_values.append(0)
        
        # Normalize to 0-100 range
        values = [max(0, min(100, v)) for v in raw_values]
        values += values[:1]
        
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=get_color(name), markersize=5)
        ax.fill(angles, values, alpha=0.15, color=get_color(name))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color=TEXT_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.spines["polar"].set_color(GRID_COLOR)
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
             facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    
    ax.set_title("Multi-Metric Radar Comparison", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=25)
    plt.tight_layout()
    _save_or_show(fig, "radar", save)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _save_or_show(fig, name, save):
    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def print_text_summary(data, algos):
    """Print a formatted text summary to the console."""
    meta = data["metadata"]
    summary = data.get("summary", {})
    rankings = data.get("rankings", {})
    
    print("\n" + "=" * 60)
    print(f"  COMPARISON RESULTS — {meta.get('maze_name', '?')}")
    print(f"  {meta['maze_size'][0]}×{meta['maze_size'][1]} maze | "
          f"Optimal: {meta.get('optimal_path_length', '?')} steps | "
          f"{meta.get('comparison_date', '?')}")
    print("=" * 60)
    
    # Rankings by path length
    by_path = rankings.get("by_path_length", [])
    if by_path:
        print("\n  📏 Rankings by Path Length:")
        for i, r in enumerate(by_path, 1):
            medal = "🥇🥈🥉"[i-1] if i <= 3 else f" {i}."
            print(f"    {medal} {r['algorithm']}: {r['path_length']} steps")
    
    # Rankings by time
    by_time = rankings.get("by_execution_time", [])
    if by_time:
        print("\n  ⏱️  Rankings by Speed:")
        for i, r in enumerate(by_time, 1):
            medal = "🥇🥈🥉"[i-1] if i <= 3 else f" {i}."
            t = r["time_seconds"]
            label = f"{t:.5f}s" if t < 1 else f"{t:.2f}s"
            print(f"    {medal} {r['algorithm']}: {label}")
    
    # RL success rates
    rl_sr = rankings.get("rl_by_success_rate", [])
    if rl_sr:
        print("\n  🎯 RL Success Rates:")
        for r in rl_sr:
            print(f"    • {r['algorithm']}: {r['success_rate_percent']:.1f}%")
    
    print("\n" + "=" * 60 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Parse arguments
    save_mode = "--save" in sys.argv
    show_all = "--all" in sys.argv
    
    # Find JSON file argument (not a flag)
    json_arg = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            json_arg = arg
            break
    
    # Load data
    data, filepath = load_comparison_json(json_arg)
    algos = get_completed_algorithms(data)
    
    if not algos:
        print("No completed algorithms found in the JSON.")
        print("Make sure the algorithms have 'status': 'completed'.")
        sys.exit(1)
    
    print(f"Found {len(algos)} completed algorithms: {', '.join(algos.keys())}")
    
    # Print text summary
    print_text_summary(data, algos)
    
    if save_mode:
        print(f"Saving charts to: {OUTPUT_FOLDER}\n")
    
    # Generate all charts
    print("Generating charts...")
    
    chart_dashboard(data, algos, save_mode)
    print("  ✓ Dashboard")
    
    chart_path_lengths(data, algos, save_mode)
    print("  ✓ Path lengths")
    
    chart_execution_times(data, algos, save_mode)
    print("  ✓ Execution times")
    
    chart_rl_success_rates(data, algos, save_mode)
    print("  ✓ RL success rates")
    
    chart_learning_curves(data, algos, save_mode)
    print("  ✓ Learning curves")
    
    chart_heatmaps(data, algos, save_mode)
    print("  ✓ Heatmaps")
    
    chart_efficiency(data, algos, save_mode)
    print("  ✓ Efficiency analysis")
    
    chart_exploration_coverage(data, algos, save_mode)
    print("  ✓ Exploration coverage")
    
    chart_nonrl_comparison_table(data, algos, save_mode)
    print("  ✓ NonRL comparison table")
    
    chart_radar(data, algos, save_mode)
    print("  ✓ Radar chart")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
