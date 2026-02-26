# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     COMPARISON VISUALIZATION SCRIPT                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Reads a comparison results JSON (from default_comparison_results structure)
# placed in the ComparisonVisualization/ folder and generates charts.
#
# Usage:
#   python visualize_comparison.py                     # picks the newest JSON
#   python visualize_comparison.py my_results.json     # specific file
#   python visualize_comparison.py --save              # save PNGs
#   python visualize_comparison.py --all               # generate every chart type

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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              DATA LOADING                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Load comparison JSON — if no filepath given, pick the newest file in INPUT_FOLDER
def load_comparison_json(filepath=None):
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


# Return dict of {name: algo_data} for algorithms that actually ran
def get_completed_algorithms(data):
    algos = {}
    for name, info in data.get("algorithms", {}).items():
        if info.get("status") == "completed":
            algos[name] = info
    return algos


def get_color(name):
    return ALGO_COLORS.get(name, "#888888")


# Detect whether the loaded JSON is a multi-run batch
def is_multi_run(data):
    return "batch_metadata" in data and "runs" in data


# Extract per-algorithm, per-run metrics from a multi-run JSON
# Returns a dict: { algo_name: { 'path_lengths': [...], 'times': [...], ... } }
def extract_multi_run_metrics(data):
    metrics = {}
    for run in data.get("runs", []):
        algos = run.get("algorithms", {})
        for algo_name, algo_data in algos.items():
            if algo_name not in metrics:
                metrics[algo_name] = {
                    "type": algo_data.get("type", "?"),
                    "path_lengths": [],
                    "times": [],
                    "efficiencies": [],
                    "extra_steps": [],
                    "success_rates": [],       # RL only
                    "first_find_episodes": [],  # RL only
                    "coverages": [],
                    "path_found_flags": [],
                    "optimal_path_lengths": [],
                    "learning_curves": [],      # RL only: list of per-run data_points
                }
            m = metrics[algo_name]
            cm = algo_data.get("common_metrics", {})
            m["path_found_flags"].append(cm.get("path_found", False))
            if cm.get("path_found", False):
                m["path_lengths"].append(cm.get("path_length", 0))
                m["efficiencies"].append(cm.get("path_efficiency", 0))
                m["extra_steps"].append(cm.get("extra_steps_vs_optimal", 0))
            m["times"].append(cm.get("time_to_solution_seconds", 0))

            # Exploration coverage
            expl = algo_data.get("exploration_data", {})
            cov = expl.get("exploration_coverage_percent", 0)
            m["coverages"].append(cov)

            # RL-specific
            if algo_data.get("type") == "RL":
                pm = algo_data.get("performance_metrics", {})
                m["success_rates"].append(pm.get("success_rate_percent", 0))
                m["first_find_episodes"].append(pm.get("first_find_episode", -1))
                lc = algo_data.get("learning_curve", {})
                pts = lc.get("data_points", [])
                if pts:
                    m["learning_curves"].append(pts)

            # Optimal path length from run metadata
            run_meta = run.get("metadata", {})
            m["optimal_path_lengths"].append(run_meta.get("optimal_path_length", 0))
    return metrics


# Apply clean light theme to figure and all axes
def apply_style(fig, axes):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    CHART 1 — PATH LENGTH COMPARISON                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Bar chart comparing path lengths across all algorithms with optimal baseline
def chart_path_lengths(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   CHART 2 — EXECUTION TIME COMPARISON                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Bar chart comparing execution/training times
def chart_execution_times(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   CHART 3 — RL SUCCESS RATE COMPARISON                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Side-by-side bars for RL algorithm success rates
def chart_rl_success_rates(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      CHART 4 — LEARNING CURVES (RL)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Line chart of RL learning curves (success rate over episode windows)
def chart_learning_curves(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    CHART 5 — EXPLORATION HEATMAPS (RL)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Side-by-side exploration heatmaps for RL algorithms
def chart_heatmaps(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  CHART 6 — PATH EFFICIENCY & EXTRA STEPS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Grouped bar chart: efficiency ratio + extra steps for all algorithms
def chart_efficiency(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CHART 7 — EXPLORATION COVERAGE                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Bar chart of exploration coverage percentage for all algorithms
def chart_exploration_coverage(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CHART 8 — SUMMARY DASHBOARD                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Multi-panel dashboard with key metrics at a glance
def chart_dashboard(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║               CHART 9 — NONRL ALGORITHM CHARACTERISTICS TABLE                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Visual table comparing NonRL algorithm properties and results
def chart_nonrl_comparison_table(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║               CHART 10 — RADAR CHART (MULTI-METRIC OVERVIEW)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Radar/spider chart comparing normalized metrics across all successful algorithms
def chart_radar(data, algos, save=False):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MULTI-RUN CHARTS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Box plot of path lengths per algorithm across all runs
def chart_multi_path_lengths_box(data, metrics, save=False):
    algo_names = sorted(metrics.keys(), key=lambda n: np.median(metrics[n]["path_lengths"]) if metrics[n]["path_lengths"] else 1e9)
    plot_data = [metrics[n]["path_lengths"] for n in algo_names]
    colors = [get_color(n) for n in algo_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    bp = ax.boxplot(plot_data, tick_labels=algo_names, patch_artist=True, widths=0.5,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(markerfacecolor=ACCENT_COLOR, markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    # Annotate median values
    for i, d in enumerate(plot_data):
        if d:
            med = np.median(d)
            ax.text(i + 1, med, f" {med:.0f}", va="center", ha="left", fontsize=8,
                    color=TEXT_COLOR, fontweight="bold")

    # Optimal line (average across runs)
    all_opts = []
    for n in algo_names:
        all_opts.extend(metrics[n]["optimal_path_lengths"])
    if all_opts:
        avg_opt = np.mean(all_opts)
        ax.axhline(y=avg_opt, color=ACCENT_COLOR, linestyle="--", linewidth=2,
                   label=f"Avg optimal ({avg_opt:.0f})")
        ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_ylabel("Path Length (steps)", fontsize=12)
    ax.set_title(f"Path Length Distribution Across {total_runs} Runs", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "multi_path_lengths_box", save)


# Box plot of execution/training times per algorithm across all runs
def chart_multi_execution_times_box(data, metrics, save=False):
    algo_names = sorted(metrics.keys(), key=lambda n: np.median(metrics[n]["times"]) if metrics[n]["times"] else 1e9)
    plot_data = [metrics[n]["times"] for n in algo_names]
    colors = [get_color(n) for n in algo_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    bp = ax.boxplot(plot_data, tick_labels=algo_names, patch_artist=True, widths=0.5,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(markerfacecolor=ACCENT_COLOR, markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    for i, d in enumerate(plot_data):
        if d:
            med = np.median(d)
            label = f" {med:.4f}s" if med < 1 else f" {med:.2f}s"
            ax.text(i + 1, med, label, va="center", ha="left", fontsize=8,
                    color=TEXT_COLOR, fontweight="bold")

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"Execution Time Distribution Across {total_runs} Runs", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "multi_execution_times_box", save)


# Box plot of RL success rates across runs
def chart_multi_success_rates_box(data, metrics, save=False):
    rl_names = [n for n in sorted(metrics.keys()) if metrics[n]["type"] == "RL" and metrics[n]["success_rates"]]
    if not rl_names:
        return

    plot_data = [metrics[n]["success_rates"] for n in rl_names]
    colors = [get_color(n) for n in rl_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)

    bp = ax.boxplot(plot_data, tick_labels=rl_names, patch_artist=True, widths=0.45,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(markerfacecolor=ACCENT_COLOR, markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    # Individual data points (jittered)
    for i, (name, d) in enumerate(zip(rl_names, plot_data)):
        jitter = np.random.normal(0, 0.04, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, color=get_color(name),
                   alpha=0.4, s=15, zorder=4, edgecolors="none")
        med = np.median(d)
        ax.text(i + 1, med, f"  {med:.1f}%", va="center", ha="left", fontsize=9,
                color=TEXT_COLOR, fontweight="bold")

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(f"RL Success Rate Distribution Across {total_runs} Runs", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(max(max(d) for d in plot_data) * 1.15, 110))
    plt.tight_layout()
    _save_or_show(fig, "multi_success_rates_box", save)


# Bar chart showing how many runs each algorithm had the shortest path
def chart_multi_win_rate(data, metrics, save=False):
    runs = data.get("runs", [])
    win_counts = {n: 0 for n in metrics}
    tie_counts = {n: 0 for n in metrics}

    for run in runs:
        algos = run.get("algorithms", {})
        best_len = float("inf")
        best_names = []
        for name, algo_data in algos.items():
            cm = algo_data.get("common_metrics", {})
            if cm.get("path_found", False):
                pl = cm["path_length"]
                if pl < best_len:
                    best_len = pl
                    best_names = [name]
                elif pl == best_len:
                    best_names.append(name)
        if len(best_names) == 1:
            win_counts[best_names[0]] += 1
        elif len(best_names) > 1:
            for n in best_names:
                tie_counts[n] += 1

    algo_names = sorted(metrics.keys())
    wins = [win_counts.get(n, 0) for n in algo_names]
    ties = [tie_counts.get(n, 0) for n in algo_names]
    colors = [get_color(n) for n in algo_names]
    total_runs = data["batch_metadata"]["total_runs"]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)

    x = np.arange(len(algo_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, wins, width, label="Wins (sole shortest)",
                   color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, ties, width, label="Ties (shared shortest)",
                   color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3, alpha=0.45)

    for bar, val in zip(bars1, wins):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, ties):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=15, ha="right")
    ax.set_ylabel("Number of Runs", fontsize=12)
    ax.set_title(f"Win Rate — Shortest Path Across {total_runs} Runs", fontsize=14, fontweight="bold", pad=15)
    ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.set_ylim(0, max(max(wins + ties) * 1.25, 2))
    plt.tight_layout()
    _save_or_show(fig, "multi_win_rate", save)


# Grouped bar chart of average path length and time with min/max error bars
def chart_multi_aggregate_bars(data, metrics, save=False):
    agg = data.get("aggregate_statistics", {})
    if not agg:
        return

    algo_names = sorted(agg.keys())
    colors = [get_color(n) for n in algo_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    apply_style(fig, [ax1, ax2])

    # ── Left: Average path length with min/max bars ──
    avgs = [agg[n].get("avg_path_length", 0) for n in algo_names]
    mins = [agg[n].get("min_path_length", 0) for n in algo_names]
    maxs = [agg[n].get("max_path_length", 0) for n in algo_names]
    err_low = [a - mn for a, mn in zip(avgs, mins)]
    err_high = [mx - a for a, mx in zip(avgs, maxs)]

    bars1 = ax1.bar(algo_names, avgs, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    ax1.errorbar(algo_names, avgs, yerr=[err_low, err_high], fmt="none",
                 ecolor=TEXT_COLOR, capsize=5, capthick=1.5, zorder=4)
    for bar, val in zip(bars1, avgs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avgs)*0.02,
                 f"{val:.1f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Path Length", fontsize=12)
    ax1.set_title("Average Path Length (min/max bars)", fontsize=13, fontweight="bold")
    plt.sca(ax1)
    plt.xticks(rotation=15, ha="right")

    # ── Right: Average execution time ──
    avg_times = [agg[n].get("avg_time_seconds", 0) for n in algo_names]
    bars2 = ax2.bar(algo_names, avg_times, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    for bar, val in zip(bars2, avg_times):
        label = f"{val:.4f}s" if val < 1 else f"{val:.2f}s"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.02,
                 label, ha="center", va="bottom", color=TEXT_COLOR, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Time (seconds)", fontsize=12)
    ax2.set_title("Average Execution Time", fontsize=13, fontweight="bold")
    plt.sca(ax2)
    plt.xticks(rotation=15, ha="right")

    total_runs = data["batch_metadata"]["total_runs"]
    fig.suptitle(f"Aggregate Statistics — {total_runs} Runs", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, "multi_aggregate_bars", save)


# Averaged RL learning curves with ±std shaded bands across runs
def chart_multi_learning_curves_avg(data, metrics, save=False):
    rl_names = [n for n in sorted(metrics.keys()) if metrics[n]["type"] == "RL" and metrics[n]["learning_curves"]]
    if not rl_names:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    for name in rl_names:
        all_curves = metrics[name]["learning_curves"]  # list of lists of data_points
        # Align curves by episode midpoints from the first curve
        if not all_curves:
            continue

        # Use the first curve's episode structure as reference
        ref = all_curves[0]
        episodes = [(p["episode_range"][0] + p["episode_range"][1]) / 2 for p in ref]
        n_points = len(episodes)

        # Gather rates across runs (pad/truncate to same length)
        all_rates = []
        for curve in all_curves:
            rates = [p["success_rate_percent"] for p in curve]
            # Pad or truncate
            if len(rates) >= n_points:
                all_rates.append(rates[:n_points])
            else:
                padded = rates + [rates[-1]] * (n_points - len(rates))
                all_rates.append(padded)

        rates_arr = np.array(all_rates)
        mean_rates = rates_arr.mean(axis=0)
        std_rates = rates_arr.std(axis=0)

        color = get_color(name)
        ax.plot(episodes, mean_rates, color=color, linewidth=2, label=f"{name} (avg)", zorder=3)
        ax.fill_between(episodes, mean_rates - std_rates, mean_rates + std_rates,
                        alpha=0.2, color=color, zorder=2)

    # Threshold line
    ax.axhline(y=70, color=ACCENT_COLOR, linestyle=":", linewidth=1.5,
               label="Convergence threshold (70%)", alpha=0.7)

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(f"RL Learning Curves — Average ± Std Across {total_runs} Runs",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(-5, 105)
    ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, "multi_learning_curves_avg", save)


# Bar chart of coefficient of variation (std/mean) for path lengths — lower = more consistent
def chart_multi_consistency(data, metrics, save=False):
    algo_names = sorted(n for n in metrics if len(metrics[n]["path_lengths"]) >= 2)
    if not algo_names:
        return

    cvs = []
    colors = []
    for n in algo_names:
        vals = metrics[n]["path_lengths"]
        mean = np.mean(vals)
        std = np.std(vals)
        cv = (std / mean * 100) if mean > 0 else 0
        cvs.append(cv)
        colors.append(get_color(n))

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)

    bars = ax.bar(algo_names, cvs, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    for bar, val in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cvs)*0.02 + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_ylabel("Coefficient of Variation (%)", fontsize=12)
    ax.set_title(f"Path Length Consistency Across {total_runs} Runs (lower = more consistent)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "multi_consistency", save)


# Multi-run overview dashboard with batch metadata and aggregate stats
def chart_multi_dashboard(data, metrics, save=False):
    batch = data["batch_metadata"]
    agg = data.get("aggregate_statistics", {})
    total_runs = batch["total_runs"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f"Multi-Run Dashboard — {total_runs} Runs | {batch.get('batch_date', '?')}",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98
    )

    # ── Panel 1: Path length box plot (top-left, 2 cols) ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    apply_style(fig, ax1)
    algo_names = sorted(metrics.keys(), key=lambda n: np.median(metrics[n]["path_lengths"]) if metrics[n]["path_lengths"] else 1e9)
    plot_data = [metrics[n]["path_lengths"] for n in algo_names]
    colors = [get_color(n) for n in algo_names]
    if plot_data and any(plot_data):
        bp = ax1.boxplot(plot_data, tick_labels=algo_names, patch_artist=True, widths=0.5,
                         medianprops=dict(color=TEXT_COLOR, linewidth=2),
                         whiskerprops=dict(color=TEXT_COLOR),
                         capprops=dict(color=TEXT_COLOR))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(BAR_EDGE_COLOR)
    ax1.set_title("Path Length Distribution", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8, rotation=15)

    # ── Panel 2: Info text (top-right) ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_COLOR)
    ax2.axis("off")
    config = data.get("shared_config", {})
    training = config.get("training", {})
    rl_hp = config.get("rl_hyperparameters", {})
    info_lines = [
        f"Total runs: {total_runs}",
        f"Batch time: {batch.get('total_batch_time_seconds', 0):.1f}s",
        f"Avg per run: {batch.get('average_run_time_seconds', 0):.1f}s",
        f"Config: {batch.get('maze_config_file', '?')}",
        "",
        f"Episodes: {training.get('episodes', '?')}",
        f"Max steps: {training.get('max_steps_per_episode', '?')}",
        f"Gamma: {rl_hp.get('gamma', '?')}",
        f"Epsilon: {rl_hp.get('epsilon_start', '?')} → {rl_hp.get('epsilon_min', '?')}",
        f"Alpha: {rl_hp.get('alpha_start', '?')} → {rl_hp.get('alpha_min', '?')}",
    ]
    text = "\n".join(info_lines)
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=9, color=TEXT_COLOR,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR))

    # ── Panel 3: Averaged learning curves (middle, 2 cols) ──
    ax3 = fig.add_subplot(gs[1, 0:2])
    apply_style(fig, ax3)
    has_curve = False
    rl_names = [n for n in sorted(metrics.keys()) if metrics[n]["type"] == "RL"]
    for name in rl_names:
        all_curves = metrics[name]["learning_curves"]
        if not all_curves:
            continue
        has_curve = True
        ref = all_curves[0]
        episodes = [(p["episode_range"][0] + p["episode_range"][1]) / 2 for p in ref]
        n_points = len(episodes)
        all_rates = []
        for curve in all_curves:
            rates = [p["success_rate_percent"] for p in curve]
            if len(rates) >= n_points:
                all_rates.append(rates[:n_points])
            else:
                padded = rates + [rates[-1]] * (n_points - len(rates))
                all_rates.append(padded)
        rates_arr = np.array(all_rates)
        mean_rates = rates_arr.mean(axis=0)
        std_rates = rates_arr.std(axis=0)
        color = get_color(name)
        ax3.plot(episodes, mean_rates, color=color, linewidth=2, label=name, zorder=3)
        ax3.fill_between(episodes, mean_rates - std_rates, mean_rates + std_rates,
                         alpha=0.15, color=color, zorder=2)
    if has_curve:
        ax3.axhline(y=70, color=ACCENT_COLOR, linestyle=":", linewidth=1, alpha=0.5)
        ax3.set_title("RL Learning Curves (avg ± std)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Episode", fontsize=9)
        ax3.set_ylabel("Success %", fontsize=9)
        ax3.set_ylim(-5, 105)
        ax3.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No RL learning curve data", ha="center", va="center",
                 color=TEXT_COLOR, fontsize=11, transform=ax3.transAxes)

    # ── Panel 4: Aggregate times bar (middle-right) ──
    ax4 = fig.add_subplot(gs[1, 2])
    apply_style(fig, ax4)
    if agg:
        an = sorted(agg.keys())
        avg_t = [agg[n].get("avg_time_seconds", 0) for n in an]
        ac = [get_color(n) for n in an]
        ax4.barh(an, avg_t, color=ac, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
        ax4.set_title("Avg Time (s)", fontsize=11, fontweight="bold")
        ax4.tick_params(axis="y", labelsize=8)

    # ── Panel 5: Win rates (bottom-left) ──
    ax5 = fig.add_subplot(gs[2, 0])
    apply_style(fig, ax5)
    runs = data.get("runs", [])
    win_counts = {n: 0 for n in metrics}
    for run in runs:
        algos_run = run.get("algorithms", {})
        best_len = float("inf")
        best_names = []
        for nm, ad in algos_run.items():
            cm = ad.get("common_metrics", {})
            if cm.get("path_found", False):
                pl = cm["path_length"]
                if pl < best_len:
                    best_len = pl
                    best_names = [nm]
                elif pl == best_len:
                    best_names.append(nm)
        for nm in best_names:
            win_counts[nm] = win_counts.get(nm, 0) + 1
    wn = sorted(win_counts.keys())
    wv = [win_counts[n] for n in wn]
    wc = [get_color(n) for n in wn]
    ax5.bar(wn, wv, color=wc, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
    ax5.set_title("Win Rate (shortest path)", fontsize=11, fontweight="bold")
    ax5.tick_params(axis="x", labelsize=7, rotation=25)

    # ── Panel 6: Consistency (CV%) (bottom-middle) ──
    ax6 = fig.add_subplot(gs[2, 1])
    apply_style(fig, ax6)
    cv_names = [n for n in sorted(metrics.keys()) if len(metrics[n]["path_lengths"]) >= 2]
    if cv_names:
        cvs = []
        cvc = []
        for n in cv_names:
            vals = metrics[n]["path_lengths"]
            mean = np.mean(vals)
            std = np.std(vals)
            cvs.append((std / mean * 100) if mean > 0 else 0)
            cvc.append(get_color(n))
        ax6.bar(cv_names, cvs, color=cvc, edgecolor=BAR_EDGE_COLOR, linewidth=0.5, zorder=3)
        ax6.set_title("Consistency (CV%)", fontsize=11, fontweight="bold")
        ax6.tick_params(axis="x", labelsize=7, rotation=25)
    else:
        ax6.text(0.5, 0.5, "Not enough data", ha="center", va="center", color=TEXT_COLOR, transform=ax6.transAxes)

    # ── Panel 7: RL success rate box (bottom-right) ──
    ax7 = fig.add_subplot(gs[2, 2])
    apply_style(fig, ax7)
    rl_with_sr = [n for n in rl_names if metrics[n]["success_rates"]]
    if rl_with_sr:
        sr_data = [metrics[n]["success_rates"] for n in rl_with_sr]
        sr_colors = [get_color(n) for n in rl_with_sr]
        bp7 = ax7.boxplot(sr_data, tick_labels=rl_with_sr, patch_artist=True, widths=0.4,
                          medianprops=dict(color=TEXT_COLOR, linewidth=2),
                          whiskerprops=dict(color=TEXT_COLOR),
                          capprops=dict(color=TEXT_COLOR))
        for patch, color in zip(bp7["boxes"], sr_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(BAR_EDGE_COLOR)
        ax7.set_title("RL Success Rate %", fontsize=11, fontweight="bold")
        ax7.set_ylim(0, 110)
        ax7.tick_params(axis="x", labelsize=8)
    else:
        ax7.text(0.5, 0.5, "No RL data", ha="center", va="center", color=TEXT_COLOR, transform=ax7.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_or_show(fig, "multi_dashboard", save)


# Box plot of path efficiency across runs per algorithm
def chart_multi_efficiency_box(data, metrics, save=False):
    algo_names = sorted(n for n in metrics if metrics[n]["efficiencies"])
    if not algo_names:
        return

    plot_data = [[e * 100 for e in metrics[n]["efficiencies"]] for n in algo_names]
    colors = [get_color(n) for n in algo_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    bp = ax.boxplot(plot_data, tick_labels=algo_names, patch_artist=True, widths=0.5,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(markerfacecolor=ACCENT_COLOR, markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    ax.axhline(y=100, color=ACCENT_COLOR, linestyle="--", linewidth=2, label="Optimal (100%)")
    ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    for i, d in enumerate(plot_data):
        if d:
            med = np.median(d)
            ax.text(i + 1, med, f" {med:.0f}%", va="center", ha="left", fontsize=8,
                    color=TEXT_COLOR, fontweight="bold")

    total_runs = data["batch_metadata"]["total_runs"]
    ax.set_ylabel("Path Efficiency (%)", fontsize=12)
    ax.set_title(f"Path Efficiency Distribution Across {total_runs} Runs",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "multi_efficiency_box", save)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               UTILITIES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _save_or_show(fig, name, save):
    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


# Print a formatted text summary to the console
def print_text_summary(data, algos):
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


# Print a formatted text summary for multi-run batch results
def print_multi_run_summary(data, metrics):
    batch = data["batch_metadata"]
    agg = data.get("aggregate_statistics", {})

    print("\n" + "=" * 65)
    print(f"  MULTI-RUN BATCH RESULTS — {batch['total_runs']} runs")
    print(f"  Date: {batch.get('batch_date', '?')} | "
          f"Total time: {batch.get('total_batch_time_seconds', 0):.1f}s | "
          f"Config: {batch.get('maze_config_file', '?')}")
    print("=" * 65)

    # Per-algorithm aggregate stats
    if agg:
        # Sort by avg path length
        sorted_algos = sorted(agg.items(), key=lambda x: x[1].get("avg_path_length", 1e9))
        print("\n  📊 Aggregate Statistics (sorted by avg path length):")
        print(f"  {'Algorithm':<16} {'Avg Path':>9} {'Min':>6} {'Max':>6} {'Avg Time':>10} {'Avg Eff':>8} {'Avg SR%':>8}")
        print("  " + "-" * 63)
        for name, stats in sorted_algos:
            avg_pl = stats.get('avg_path_length', 0)
            min_pl = stats.get('min_path_length', 0)
            max_pl = stats.get('max_path_length', 0)
            avg_t = stats.get('avg_time_seconds', 0)
            avg_eff = stats.get('avg_path_efficiency', 0)
            avg_sr = stats.get('avg_success_rate_percent')
            sr_str = f"{avg_sr:.1f}%" if avg_sr is not None else "  N/A"
            time_str = f"{avg_t:.4f}s" if avg_t < 1 else f"{avg_t:.2f}s"
            print(f"  {name:<16} {avg_pl:>9.1f} {min_pl:>6} {max_pl:>6} {time_str:>10} {avg_eff:>7.2f}% {sr_str:>8}")

    # Win count
    runs = data.get("runs", [])
    win_counts = {}
    for run in runs:
        algos_run = run.get("algorithms", {})
        best_len = float("inf")
        best_names = []
        for nm, ad in algos_run.items():
            cm = ad.get("common_metrics", {})
            if cm.get("path_found", False):
                pl = cm["path_length"]
                if pl < best_len:
                    best_len = pl
                    best_names = [nm]
                elif pl == best_len:
                    best_names.append(nm)
        for nm in best_names:
            win_counts[nm] = win_counts.get(nm, 0) + 1

    if win_counts:
        print("\n  🏆 Win Count (shortest path per run):")
        for name, count in sorted(win_counts.items(), key=lambda x: -x[1]):
            pct = count / len(runs) * 100
            print(f"    {name}: {count}/{len(runs)} ({pct:.0f}%)")

    print("\n" + "=" * 65 + "\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                  MAIN                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

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

    # ── Detect multi-run vs single-run format ──
    if is_multi_run(data):
        _main_multi_run(data, filepath, save_mode)
    else:
        _main_single_run(data, filepath, save_mode)


# Original flow for a single-run comparison JSON
def _main_single_run(data, filepath, save_mode):
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


# Flow for a multi-run batch JSON
def _main_multi_run(data, filepath, save_mode):
    batch = data["batch_metadata"]
    total_runs = batch["total_runs"]
    metrics = extract_multi_run_metrics(data)

    if not metrics:
        print("No algorithm data found in any run.")
        sys.exit(1)

    print(f"Multi-run batch detected: {total_runs} runs")
    print(f"Algorithms: {', '.join(sorted(metrics.keys()))}")

    # Print text summary
    print_multi_run_summary(data, metrics)

    if save_mode:
        print(f"Saving charts to: {OUTPUT_FOLDER}\n")

    # Generate multi-run charts
    print("Generating multi-run charts...")

    chart_multi_dashboard(data, metrics, save_mode)
    print("  ✓ Multi-run dashboard")

    chart_multi_path_lengths_box(data, metrics, save_mode)
    print("  ✓ Path length box plots")

    chart_multi_execution_times_box(data, metrics, save_mode)
    print("  ✓ Execution time box plots")

    chart_multi_success_rates_box(data, metrics, save_mode)
    print("  ✓ RL success rate box plots")

    chart_multi_learning_curves_avg(data, metrics, save_mode)
    print("  ✓ Averaged learning curves")

    chart_multi_efficiency_box(data, metrics, save_mode)
    print("  ✓ Path efficiency box plots")

    chart_multi_aggregate_bars(data, metrics, save_mode)
    print("  ✓ Aggregate statistics bars")

    chart_multi_win_rate(data, metrics, save_mode)
    print("  ✓ Win rate chart")

    chart_multi_consistency(data, metrics, save_mode)
    print("  ✓ Consistency chart")

    print("\nDone!")


if __name__ == "__main__":
    main()
