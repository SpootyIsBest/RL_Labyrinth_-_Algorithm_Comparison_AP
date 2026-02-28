"""
Parameter Sweep Tool for RL Labyrinth
======================================
Allows you to define multiple values for any hyperparameter and run N repetitions
per configuration.  Both Q-Learning and SARSA are trained for each config.

Usage:
    python parameter_sweep.py          # Opens the Pygame config UI
    python parameter_sweep.py --last   # Re-visualize the latest sweep results
"""

import os
import sys
import json
import time
import random
import math
import copy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import project classes (Maze, Agent, State must be on the path)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Maze import Maze
from Agent import Agent
from State import State

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       HEADLESS TRAINING ENGINE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def valid_actions(state, maze):
    x, y = state
    return maze.gridStates[y][x].actions

def masked_max(q_row, acts):
    if not acts:
        return 0.0
    return max(q_row[a] for a in acts)

def masked_argmax(q_row, acts):
    if not acts:
        return None
    best = max(q_row[a] for a in acts)
    best_as = [a for a in acts if q_row[a] == best]
    return random.choice(best_as)

def epsilon_greedy_action(state, Q, epsilon, maze):
    acts = valid_actions(state, maze)
    if not acts:
        return None
    if random.random() < epsilon:
        return random.choice(acts)
    return masked_argmax(Q[state[1], state[0]], acts)


def train_headless(algorithm, agent, maze, Q,
                   EPISODES, max_steps, gamma,
                   EPS0, EPS_MIN, EPS_DECAY,
                   ALPHA0, ALPHA_MIN, ALPHA_DECAY):
    """Run complete training WITHOUT Pygame yields.  Returns a metrics dict."""
    find_episodes = []
    first_find = None
    num_returns = 0
    final_path = []
    heat_table = [[0] * maze.maze_size_width for _ in range(maze.maze_size_height)]

    # For learning-curve tracking (sliding window)
    window_successes = []   # bool per episode

    for episode in range(EPISODES):
        epsilon = max(EPS_MIN, EPS0 * (EPS_DECAY ** episode))
        alpha   = max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** episode))

        agent.reset()
        state = agent.activeState[:]
        action = epsilon_greedy_action(state, Q, epsilon, maze)
        current_path = [state[:]]
        reached_goal = False

        for t in range(max_steps):
            if action is None:  # terminal / goal
                if first_find is None:
                    first_find = episode + 1
                num_returns += 1
                find_episodes.append(episode + 1)
                final_path = current_path[:]
                reached_goal = True
                break

            reward, next_state = agent.ProcessNextAction(action)
            x, y   = state
            nx, ny = next_state
            heat_table[y][x] += 1

            if algorithm == "Q-Learning":
                next_acts = valid_actions(next_state, maze)
                target = reward + gamma * masked_max(Q[ny, nx], next_acts)
                Q[y, x, action] += alpha * (target - Q[y, x, action])
                state  = next_state
                action = epsilon_greedy_action(state, Q, epsilon, maze)
            else:  # SARSA
                next_action = epsilon_greedy_action(next_state, Q, epsilon, maze)
                if next_action is None:
                    target = reward
                else:
                    target = reward + gamma * Q[ny, nx, next_action]
                Q[y, x, action] += alpha * (target - Q[y, x, action])
                state  = next_state
                action = next_action

            current_path.append(state[:])

        window_successes.append(reached_goal)

    # --- Build learning curve (sliding window success %) ---
    window = max(1, EPISODES // 50)
    learning_curve = []
    for i in range(0, EPISODES, window):
        chunk = window_successes[i:i + window]
        rate = sum(chunk) / len(chunk) * 100 if chunk else 0
        learning_curve.append({"episode": i + window, "success_rate": round(rate, 2)})

    success_rate = (num_returns / EPISODES * 100) if EPISODES > 0 else 0

    return {
        "find_episodes":     find_episodes,
        "first_find":        first_find,
        "num_returns":       num_returns,
        "success_rate":      round(success_rate, 2),
        "final_path_length": len(final_path),
        "learning_curve":    learning_curve,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     MAZE CREATION (headless – no screen)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Pygame must be initialised for Maze (it uses Surface internally for drawing,
# but we never actually display anything during headless training).
import pygame
pygame.init()
_dummy_screen = pygame.Surface((1, 1))


def create_maze_headless(width, height, reward_finish, reward_move):
    """Create a random maze the same way main.py does."""
    origin = [1, 1]
    maze = Maze("sweep", _dummy_screen, 800, 600, width, height, origin)
    maze.create_default()
    # Scale shuffle steps to maze size (5M for large mazes is overkill for small ones)
    shuffle_steps = max(10_000, width * height * 500)
    maze.random_sequence(shuffle_steps)
    maze.carve_walls_from_arrows()
    maze.cal_init_pos()
    maze.create_optimal_path(maze.start_pos)
    maze.create_grid_states(reward_finish, reward_move)
    return maze


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           SWEEP RUNNER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Each "parameter set" that the user defines.
# Example: {"name": "EPS0=0.3", "EPS0": 0.3, "ALPHA0": 0.72, ...}

PARAM_KEYS = [
    "EPS0", "EPS_MIN", "EPS_DECAY",
    "ALPHA0", "ALPHA_MIN", "ALPHA_DECAY",
    "gamma", "EPISODES", "max_steps",
    "MazeWidth", "MazeHeight",
]

DEFAULT_PARAMS = {
    "EPS0":       0.9,
    "EPS_MIN":    0.05,
    "EPS_DECAY":  0.9995,
    "ALPHA0":     0.72,
    "ALPHA_MIN":  0.10,
    "ALPHA_DECAY":0.997,
    "gamma":      1.0,
    "EPISODES":   5000,
    "max_steps":  100,
    "MazeWidth":  5,
    "MazeHeight": 5,
    "rewardForFinish":    50,
    "rewardForValidMove": -1,
}


def run_sweep(configs, runs_per_config, progress_callback=None):
    """
    configs : list of dicts, each is a full parameter set (merged with defaults).
    runs_per_config : int
    progress_callback(current, total, config_name): called for progress updates
    Returns a list of config result dicts.
    """
    total = len(configs) * runs_per_config * 2  # ×2 for Q-Learning + SARSA
    current = 0
    all_results = []

    for cfg_idx, cfg in enumerate(configs):
        cfg_name = cfg.get("label", f"Config {cfg_idx+1}")
        mw = int(cfg.get("MazeWidth", 5))
        mh = int(cfg.get("MazeHeight", 5))
        episodes  = int(cfg.get("EPISODES", 5000))
        msteps    = int(cfg.get("max_steps", 100))
        g         = float(cfg.get("gamma", 1.0))
        eps0      = float(cfg.get("EPS0", 0.9))
        eps_min   = float(cfg.get("EPS_MIN", 0.05))
        eps_decay = float(cfg.get("EPS_DECAY", 0.9995))
        a0        = float(cfg.get("ALPHA0", 0.72))
        a_min     = float(cfg.get("ALPHA_MIN", 0.1))
        a_decay   = float(cfg.get("ALPHA_DECAY", 0.997))
        rf        = float(cfg.get("rewardForFinish", 50))
        rm        = float(cfg.get("rewardForValidMove", -1))

        cfg_result = {
            "label": cfg_name,
            "params": {k: cfg.get(k, DEFAULT_PARAMS.get(k)) for k in PARAM_KEYS},
            "algorithms": {}
        }

        for algo in ["Q-Learning", "SARSA"]:
            run_metrics = []
            for run_idx in range(runs_per_config):
                # Fresh random maze per run
                maze = create_maze_headless(mw, mh, rf, rm)
                agent = Agent(-5, maze.gridStates, maze.start_pos)
                Q = np.zeros((mh, mw, 4), dtype=float)

                t0 = time.time()
                m = train_headless(
                    algo, agent, maze, Q,
                    episodes, msteps, g,
                    eps0, eps_min, eps_decay,
                    a0, a_min, a_decay
                )
                m["training_time"] = round(time.time() - t0, 3)
                m["optimal_path_length"] = len(maze.optimal_path)
                run_metrics.append(m)

                current += 1
                if progress_callback:
                    progress_callback(current, total, f"{cfg_name} | {algo} run {run_idx+1}")

            # Aggregate across runs
            cfg_result["algorithms"][algo] = aggregate_runs(run_metrics)

        all_results.append(cfg_result)

    return all_results


def aggregate_runs(runs):
    """Compute mean, std, min, max for each metric across runs."""
    keys = ["success_rate", "first_find", "num_returns", "final_path_length", "training_time", "optimal_path_length"]
    agg = {"runs": runs, "num_runs": len(runs)}
    for k in keys:
        vals = [r[k] for r in runs if r[k] is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            agg[k] = {
                "mean": round(float(np.mean(arr)), 4),
                "std":  round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
                "min":  round(float(np.min(arr)), 4),
                "max":  round(float(np.max(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "values": [round(float(v), 4) for v in arr],
            }
        else:
            agg[k] = {"mean": None, "std": None, "min": None, "max": None, "median": None, "values": []}

    # Aggregate learning curves (average across runs at each data point)
    all_curves = [r["learning_curve"] for r in runs if r.get("learning_curve")]
    if all_curves:
        # Find min length
        min_len = min(len(c) for c in all_curves)
        avg_curve = []
        for i in range(min_len):
            rates = [c[i]["success_rate"] for c in all_curves if i < len(c)]
            avg_curve.append({
                "episode":      all_curves[0][i]["episode"],
                "success_rate_mean": round(float(np.mean(rates)), 2),
                "success_rate_std":  round(float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0, 2),
            })
        agg["learning_curve_avg"] = avg_curve
    else:
        agg["learning_curve_avg"] = []

    return agg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         RESULTS SAVE / LOAD                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SWEEP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SweepResults")


def save_sweep_results(results, sweep_param, base_params, runs_per_config):
    os.makedirs(SWEEP_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sweep_parameter": sweep_param,
            "runs_per_config": runs_per_config,
            "num_configs": len(results),
            "total_runs": len(results) * runs_per_config * 2,
        },
        "base_params": base_params,
        "configs": results,
    }
    fname = f"Sweep_{sweep_param}_{len(results)}configs_{timestamp}.json"
    fpath = os.path.join(SWEEP_FOLDER, fname)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved sweep results to {fpath}")
    return fpath


def load_latest_sweep():
    json_files = list(Path(SWEEP_FOLDER).glob("Sweep_*.json"))
    if not json_files:
        return None, None
    path = str(max(json_files, key=os.path.getmtime))
    with open(path) as f:
        return json.load(f), path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      MATPLOTLIB VISUALIZATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ALGO_COLORS = {"Q-Learning": "#1565C0", "SARSA": "#E65100"}


def visualize_sweep(sweep_data, save_png=False):
    """Generate all comparison charts from a sweep result dict."""
    configs = sweep_data["configs"]
    sweep_param = sweep_data["metadata"]["sweep_parameter"]
    labels = [c["label"] for c in configs]
    n = len(configs)
    out_dir = os.path.join(SWEEP_FOLDER, "charts")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1) SUCCESS RATE — bar chart with error bars ──────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 6))
    x = np.arange(n)
    w = 0.35
    for i, algo in enumerate(["Q-Learning", "SARSA"]):
        means = []
        stds  = []
        for c in configs:
            a = c["algorithms"].get(algo, {})
            sr = a.get("success_rate", {})
            means.append(sr.get("mean", 0) or 0)
            stds.append(sr.get("std", 0) or 0)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, means, w, yerr=stds, label=algo,
                      color=ALGO_COLORS[algo], capsize=4, edgecolor="#333", alpha=0.85)
        # Value labels on bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Success Rate by {sweep_param}\n(mean ± std dev over {sweep_data['metadata']['runs_per_config']} runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, min(110, max(m + s + 10 for m, s in zip(means, stds)) if means else 110))
    plt.tight_layout()
    if save_png:
        fig.savefig(os.path.join(out_dir, f"sweep_success_rate_{sweep_param}.png"), dpi=150)
    plt.show(block=False)

    # ── 2) FIRST FIND EPISODE — bar chart with error bars ────────────────
    fig2, ax2 = plt.subplots(figsize=(max(8, n * 1.2), 6))
    for i, algo in enumerate(["Q-Learning", "SARSA"]):
        means = []
        stds  = []
        for c in configs:
            a = c["algorithms"].get(algo, {})
            ff = a.get("first_find", {})
            means.append(ff.get("mean", 0) or 0)
            stds.append(ff.get("std", 0) or 0)
        offset = (i - 0.5) * w
        bars = ax2.bar(x + offset, means, w, yerr=stds, label=algo,
                       color=ALGO_COLORS[algo], capsize=4, edgecolor="#333", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                         f"{m:.0f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel(sweep_param)
    ax2.set_ylabel("First Find Episode")
    ax2.set_title(f"First Find Episode by {sweep_param}\n(mean ± std dev, lower = faster learning)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.legend()
    plt.tight_layout()
    if save_png:
        fig2.savefig(os.path.join(out_dir, f"sweep_first_find_{sweep_param}.png"), dpi=150)
    plt.show(block=False)

    # ── 3) FINAL PATH LENGTH — bar chart with error bars ─────────────────
    fig3, ax3 = plt.subplots(figsize=(max(8, n * 1.2), 6))
    for i, algo in enumerate(["Q-Learning", "SARSA"]):
        means = []
        stds  = []
        for c in configs:
            a = c["algorithms"].get(algo, {})
            pl = a.get("final_path_length", {})
            means.append(pl.get("mean", 0) or 0)
            stds.append(pl.get("std", 0) or 0)
        offset = (i - 0.5) * w
        bars = ax3.bar(x + offset, means, w, yerr=stds, label=algo,
                       color=ALGO_COLORS[algo], capsize=4, edgecolor="#333", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.3,
                         f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    # Show optimal path length as a dashed line
    opt_lengths = []
    for c in configs:
        for algo in ["Q-Learning", "SARSA"]:
            a = c["algorithms"].get(algo, {})
            ol = a.get("optimal_path_length", {})
            if ol.get("mean"):
                opt_lengths.append(ol["mean"])
    if opt_lengths:
        ax3.axhline(y=np.mean(opt_lengths), color="red", linestyle="--", label="Optimal path (avg)", alpha=0.7)

    ax3.set_xlabel(sweep_param)
    ax3.set_ylabel("Final Path Length")
    ax3.set_title(f"Final Path Length by {sweep_param}\n(mean ± std dev, closer to optimal = better)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.legend()
    plt.tight_layout()
    if save_png:
        fig3.savefig(os.path.join(out_dir, f"sweep_path_length_{sweep_param}.png"), dpi=150)
    plt.show(block=False)

    # ── 4) TRAINING TIME — bar chart with error bars ─────────────────────
    fig4, ax4 = plt.subplots(figsize=(max(8, n * 1.2), 6))
    for i, algo in enumerate(["Q-Learning", "SARSA"]):
        means = []
        stds  = []
        for c in configs:
            a = c["algorithms"].get(algo, {})
            tt = a.get("training_time", {})
            means.append(tt.get("mean", 0) or 0)
            stds.append(tt.get("std", 0) or 0)
        offset = (i - 0.5) * w
        ax4.bar(x + offset, means, w, yerr=stds, label=algo,
                color=ALGO_COLORS[algo], capsize=4, edgecolor="#333", alpha=0.85)
    ax4.set_xlabel(sweep_param)
    ax4.set_ylabel("Training Time (seconds)")
    ax4.set_title(f"Training Time by {sweep_param}")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=30, ha="right")
    ax4.legend()
    plt.tight_layout()
    if save_png:
        fig4.savefig(os.path.join(out_dir, f"sweep_training_time_{sweep_param}.png"), dpi=150)
    plt.show(block=False)

    # ── 5) LEARNING CURVES — one subplot per config ──────────────────────
    if n <= 12:
        cols = min(n, 4)
        rows = math.ceil(n / cols)
        fig5, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for idx, c in enumerate(configs):
            r, cc = divmod(idx, cols)
            ax5 = axes[r][cc]
            for algo in ["Q-Learning", "SARSA"]:
                a = c["algorithms"].get(algo, {})
                lc = a.get("learning_curve_avg", [])
                if lc:
                    eps = [p["episode"] for p in lc]
                    means_lc = [p["success_rate_mean"] for p in lc]
                    stds_lc  = [p["success_rate_std"] for p in lc]
                    ax5.plot(eps, means_lc, label=algo, color=ALGO_COLORS[algo], linewidth=1.5)
                    ax5.fill_between(eps,
                                     [m - s for m, s in zip(means_lc, stds_lc)],
                                     [m + s for m, s in zip(means_lc, stds_lc)],
                                     alpha=0.2, color=ALGO_COLORS[algo])
            ax5.set_title(c["label"], fontsize=10)
            ax5.set_xlabel("Episode")
            ax5.set_ylabel("Success %")
            ax5.set_ylim(-5, 105)
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        # Hide unused subplots
        for idx in range(n, rows * cols):
            r, cc = divmod(idx, cols)
            axes[r][cc].set_visible(False)
        fig5.suptitle(f"Learning Curves by {sweep_param} (mean ± std)", fontsize=14)
        plt.tight_layout()
        if save_png:
            fig5.savefig(os.path.join(out_dir, f"sweep_learning_curves_{sweep_param}.png"), dpi=150)
        plt.show(block=False)

    # ── 6) DEFLECTION TABLE — printed to console and shown as figure ─────
    print_deflection_table(configs, sweep_param)
    draw_deflection_table(configs, sweep_param, save_png, out_dir)

    plt.show()  # block until all windows closed


def print_deflection_table(configs, sweep_param):
    """Print a table showing mean, std, min, max (deflections) for every config."""
    metrics = ["success_rate", "first_find", "final_path_length", "training_time"]
    header = f"{'Config':<20} {'Algo':<12} {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}"
    print("\n" + "=" * 100)
    print(f"  DEFLECTION TABLE  —  Sweep Parameter: {sweep_param}")
    print("=" * 100)
    print(header)
    print("-" * 100)
    for c in configs:
        for algo in ["Q-Learning", "SARSA"]:
            a = c["algorithms"].get(algo, {})
            for mk in metrics:
                d = a.get(mk, {})
                mean_v = d.get("mean")
                std_v  = d.get("std")
                min_v  = d.get("min")
                max_v  = d.get("max")
                if mean_v is not None:
                    rng = round(max_v - min_v, 4) if max_v is not None and min_v is not None else "N/A"
                    print(f"{c['label']:<20} {algo:<12} {mk:<20} {mean_v:>8.2f} {std_v:>8.2f} {min_v:>8.2f} {max_v:>8.2f} {rng:>8}")
                else:
                    print(f"{c['label']:<20} {algo:<12} {mk:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    print("=" * 100 + "\n")


def draw_deflection_table(configs, sweep_param, save_png=False, out_dir="."):
    """Draw the deflection table as a matplotlib figure."""
    metrics = ["success_rate", "first_find", "final_path_length", "training_time"]
    metric_labels = ["Success Rate (%)", "First Find Ep.", "Path Length", "Train Time (s)"]
    
    rows_data = []
    row_labels = []
    for c in configs:
        for algo in ["Q-Learning", "SARSA"]:
            a = c["algorithms"].get(algo, {})
            row = []
            for mk in metrics:
                d = a.get(mk, {})
                mean_v = d.get("mean")
                std_v  = d.get("std")
                min_v  = d.get("min")
                max_v  = d.get("max")
                if mean_v is not None:
                    row.append(f"{mean_v:.2f} ± {std_v:.2f}\n[{min_v:.1f} – {max_v:.1f}]")
                else:
                    row.append("N/A")
            rows_data.append(row)
            row_labels.append(f"{c['label']}\n{algo}")

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 3), max(4, len(rows_data) * 0.8 + 2)))
    ax.axis("off")
    col_labels = metric_labels
    table = ax.table(cellText=rows_data, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color row labels
    for i in range(len(row_labels)):
        table[i + 1, -1].set_facecolor("#E8EAF6")

    ax.set_title(f"Deflection Summary — Sweep: {sweep_param}\n(mean ± std dev  [min – max])", fontsize=13, pad=20)
    plt.tight_layout()
    if save_png:
        fig.savefig(os.path.join(out_dir, f"sweep_deflections_{sweep_param}.png"), dpi=150, bbox_inches="tight")
    plt.show(block=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         PYGAME CONFIG UI                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SWEEPABLE_PARAMS = [
    ("EPS0",       "Initial Epsilon"),
    ("ALPHA0",     "Initial Alpha"),
    ("gamma",      "Gamma (discount)"),
    ("EPS_DECAY",  "Epsilon Decay"),
    ("ALPHA_DECAY","Alpha Decay"),
    ("EPS_MIN",    "Epsilon Min"),
    ("ALPHA_MIN",  "Alpha Min"),
    ("EPISODES",   "Episodes"),
    ("max_steps",  "Max Steps / Episode"),
    ("MazeWidth",  "Maze Width"),
    ("MazeHeight", "Maze Height"),
]


def run_config_ui():
    """Open a Pygame window for the user to configure the sweep, then run it."""
    pygame.init()
    SW, SH = 1100, 750
    screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)
    pygame.display.set_caption("Parameter Sweep Configuration")
    clock = pygame.time.Clock()

    FONT      = pygame.font.Font(None, 26)
    FONT_SM   = pygame.font.Font(None, 22)
    FONT_LG   = pygame.font.Font(None, 42)
    FONT_TINY = pygame.font.Font(None, 18)
    BG_COLOR  = pygame.Color(20, 20, 35)
    FG_COLOR  = pygame.Color(255, 255, 255)
    ACCENT    = pygame.Color(30, 100, 200)
    INPUT_BG  = pygame.Color(40, 40, 60)
    INPUT_ACT = pygame.Color(60, 80, 140)
    GOOD      = pygame.Color(100, 255, 100)

    # ── State ────────────────────────────────────────────────────────────
    selected_param_idx = 0   # which parameter to sweep
    values_text = "0.1, 0.3, 0.5, 0.7, 0.9"  # comma-separated values
    runs_text = "5"
    base_params = dict(DEFAULT_PARAMS)  # deep copy of defaults

    # Build input fields for base params (key, display_name, current_text, active)
    base_fields = []
    for key, display in SWEEPABLE_PARAMS:
        base_fields.append({
            "key": key,
            "display": display,
            "text": str(DEFAULT_PARAMS.get(key, "")),
            "active": False,
        })

    # Additional non-sweepable params
    extra_fields = [
        {"key": "rewardForFinish",    "display": "Reward (Finish)",    "text": "50",  "active": False},
        {"key": "rewardForValidMove", "display": "Reward (Move)",      "text": "-1",  "active": False},
    ]

    active_field = None   # "values", "runs", or ("base", idx), or ("extra", idx)
    dropdown_open = False

    running_sweep = False
    sweep_progress = [0, 1, ""]  # [current, total, label]
    sweep_results = None
    sweep_file = None

    def get_active_text():
        if active_field == "values":
            return values_text
        elif active_field == "runs":
            return runs_text
        elif isinstance(active_field, tuple) and active_field[0] == "base":
            return base_fields[active_field[1]]["text"]
        elif isinstance(active_field, tuple) and active_field[0] == "extra":
            return extra_fields[active_field[1]]["text"]
        return ""

    def set_active_text(txt):
        nonlocal values_text, runs_text
        if active_field == "values":
            values_text = txt
        elif active_field == "runs":
            runs_text = txt
        elif isinstance(active_field, tuple) and active_field[0] == "base":
            base_fields[active_field[1]]["text"] = txt
        elif isinstance(active_field, tuple) and active_field[0] == "extra":
            extra_fields[active_field[1]]["text"] = txt

    def build_configs():
        """Parse UI state into a list of config dicts."""
        sweep_key = SWEEPABLE_PARAMS[selected_param_idx][0]
        # Parse values
        raw_vals = [v.strip() for v in values_text.split(",") if v.strip()]
        configs = []
        for v in raw_vals:
            try:
                val = float(v)
            except ValueError:
                continue
            cfg = {}
            # Base params from fields
            for bf in base_fields:
                try:
                    cfg[bf["key"]] = float(bf["text"])
                except ValueError:
                    cfg[bf["key"]] = DEFAULT_PARAMS.get(bf["key"], 0)
            for ef in extra_fields:
                try:
                    cfg[ef["key"]] = float(ef["text"])
                except ValueError:
                    cfg[ef["key"]] = DEFAULT_PARAMS.get(ef["key"], 0)
            # Override the swept parameter
            cfg[sweep_key] = val
            cfg["label"] = f"{sweep_key}={v}"
            configs.append(cfg)
        return configs, sweep_key

    def progress_cb(cur, tot, label):
        sweep_progress[0] = cur
        sweep_progress[1] = tot
        sweep_progress[2] = label

    # ── Main Loop ────────────────────────────────────────────────────────
    ui_running = True
    while ui_running:
        SW, SH = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ui_running = False
            if event.type == pygame.VIDEORESIZE:
                SW, SH = event.w, event.h
                screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)

            if running_sweep:
                continue  # ignore input during sweep

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if sweep_results:
                        # Go back to config
                        sweep_results = None
                    else:
                        ui_running = False
                elif active_field is not None:
                    txt = get_active_text()
                    if event.key == pygame.K_RETURN:
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:
                        set_active_text(txt[:-1])
                    elif event.key == pygame.K_TAB:
                        active_field = None
                    else:
                        set_active_text(txt + event.unicode)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                active_field = None

                # Check dropdown
                dropdown_rect = pygame.Rect(20, 90, 280, 30)
                if dropdown_rect.collidepoint(mx, my):
                    dropdown_open = not dropdown_open
                    continue

                if dropdown_open:
                    for i in range(len(SWEEPABLE_PARAMS)):
                        opt_rect = pygame.Rect(20, 125 + i * 28, 280, 26)
                        if opt_rect.collidepoint(mx, my):
                            selected_param_idx = i
                            dropdown_open = False
                            break
                    else:
                        dropdown_open = False
                    continue

                # Values field
                values_rect = pygame.Rect(20, 160, SW - 40, 30)
                if values_rect.collidepoint(mx, my):
                    active_field = "values"
                    continue

                # Runs field
                runs_rect = pygame.Rect(200, 200, 100, 28)
                if runs_rect.collidepoint(mx, my):
                    active_field = "runs"
                    continue

                # Base param fields
                col_x = 20
                row_y_start = 270
                row_h = 36
                items_per_col = 6
                col_w = 260
                for i, bf in enumerate(base_fields):
                    ci = i // items_per_col
                    ri = i % items_per_col
                    fx = col_x + ci * col_w + 140
                    fy = row_y_start + ri * row_h
                    r = pygame.Rect(fx, fy, 100, 26)
                    if r.collidepoint(mx, my):
                        active_field = ("base", i)
                        break

                # Extra fields
                ex_start_y = row_y_start + items_per_col * row_h + 20
                for i, ef in enumerate(extra_fields):
                    fx = col_x + 140
                    fy = ex_start_y + i * row_h
                    r = pygame.Rect(fx, fy, 100, 26)
                    if r.collidepoint(mx, my):
                        active_field = ("extra", i)
                        break

                # Start button
                btn_rect = pygame.Rect(SW // 2 - 120, SH - 70, 240, 50)
                if btn_rect.collidepoint(mx, my):
                    cfgs, skey = build_configs()
                    if cfgs:
                        try:
                            nruns = max(1, int(runs_text))
                        except ValueError:
                            nruns = 3
                        running_sweep = True
                        sweep_progress[:] = [0, len(cfgs) * nruns * 2, "Starting..."]

                        # Run sweep in-thread (blocking, but we update progress via callback)
                        import threading
                        def do_sweep():
                            nonlocal sweep_results, running_sweep, sweep_file
                            bp = {}
                            for bf in base_fields:
                                try: bp[bf["key"]] = float(bf["text"])
                                except: bp[bf["key"]] = DEFAULT_PARAMS.get(bf["key"])
                            for ef in extra_fields:
                                try: bp[ef["key"]] = float(ef["text"])
                                except: bp[ef["key"]] = DEFAULT_PARAMS.get(ef["key"])
                            res = run_sweep(cfgs, nruns, progress_callback=progress_cb)
                            sweep_file = save_sweep_results(res, skey, bp, nruns)
                            sweep_results = {
                                "metadata": {"sweep_parameter": skey, "runs_per_config": nruns, "num_configs": len(cfgs), "total_runs": len(cfgs)*nruns*2},
                                "base_params": bp,
                                "configs": res,
                            }
                            running_sweep = False

                        t = threading.Thread(target=do_sweep, daemon=True)
                        t.start()

                # View Charts button (only when results exist)
                if sweep_results:
                    chart_btn = pygame.Rect(SW // 2 - 120, SH - 130, 240, 50)
                    if chart_btn.collidepoint(mx, my):
                        visualize_sweep(sweep_results, save_png=True)

        # ── DRAW ─────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        if running_sweep:
            # Progress screen
            title = FONT_LG.render("Running Parameter Sweep...", True, FG_COLOR)
            screen.blit(title, (SW // 2 - title.get_width() // 2, SH // 3 - 40))

            cur, tot, lbl = sweep_progress
            pct = cur / tot * 100 if tot > 0 else 0
            pbar_w = SW - 100
            pbar_h = 30
            px, py = 50, SH // 2
            pygame.draw.rect(screen, pygame.Color(60, 60, 80), (px, py, pbar_w, pbar_h), border_radius=6)
            fill_w = int(pbar_w * pct / 100)
            if fill_w > 0:
                pygame.draw.rect(screen, ACCENT, (px, py, fill_w, pbar_h), border_radius=6)
            pct_surf = FONT.render(f"{pct:.1f}%  ({cur}/{tot})", True, FG_COLOR)
            screen.blit(pct_surf, (SW // 2 - pct_surf.get_width() // 2, py + 40))
            lbl_surf = FONT_SM.render(str(lbl), True, pygame.Color(180, 180, 200))
            screen.blit(lbl_surf, (SW // 2 - lbl_surf.get_width() // 2, py + 70))

        elif sweep_results:
            # Results summary screen
            title = FONT_LG.render("Sweep Complete!", True, GOOD)
            screen.blit(title, (SW // 2 - title.get_width() // 2, 20))

            meta = sweep_results["metadata"]
            lines = [
                f"Swept: {meta['sweep_parameter']}   |   {meta['num_configs']} configs   |   {meta['runs_per_config']} runs each   |   {meta['total_runs']} total trainings",
            ]
            for i, line in enumerate(lines):
                s = FONT.render(line, True, FG_COLOR)
                screen.blit(s, (30, 80 + i * 28))

            # Quick summary table
            y_off = 130
            hdr = FONT_SM.render(f"{'Config':<18} {'Algo':<12} {'Success%':>10} {'±Std':>8} {'1stFind':>10} {'±Std':>8} {'PathLen':>10} {'±Std':>8}", True, pygame.Color(200, 200, 255))
            screen.blit(hdr, (20, y_off))
            y_off += 26
            pygame.draw.line(screen, pygame.Color(100, 100, 140), (20, y_off), (SW - 20, y_off))
            y_off += 4

            for c in sweep_results["configs"]:
                for algo in ["Q-Learning", "SARSA"]:
                    a = c["algorithms"].get(algo, {})
                    sr = a.get("success_rate", {})
                    ff = a.get("first_find", {})
                    pl = a.get("final_path_length", {})

                    def fmt(d, k="mean"):
                        v = d.get(k)
                        return f"{v:.2f}" if v is not None else "N/A"

                    txt = f"{c['label']:<18} {algo:<12} {fmt(sr):>10} {fmt(sr,'std'):>8} {fmt(ff):>10} {fmt(ff,'std'):>8} {fmt(pl):>10} {fmt(pl,'std'):>8}"
                    color = pygame.Color(180, 220, 255) if algo == "Q-Learning" else pygame.Color(255, 200, 150)
                    s = FONT_TINY.render(txt, True, color)
                    screen.blit(s, (20, y_off))
                    y_off += 20
                y_off += 4

            # Buttons
            chart_btn = pygame.Rect(SW // 2 - 120, SH - 130, 240, 50)
            pygame.draw.rect(screen, ACCENT, chart_btn, border_radius=10)
            ct = FONT.render("View Charts", True, FG_COLOR)
            screen.blit(ct, (chart_btn.centerx - ct.get_width() // 2, chart_btn.centery - ct.get_height() // 2))

            back_btn = pygame.Rect(SW // 2 - 120, SH - 70, 240, 50)
            pygame.draw.rect(screen, pygame.Color(80, 40, 40), back_btn, border_radius=10)
            bt = FONT.render("New Sweep (ESC)", True, FG_COLOR)
            screen.blit(bt, (back_btn.centerx - bt.get_width() // 2, back_btn.centery - bt.get_height() // 2))

        else:
            # ── CONFIG SCREEN ────────────────────────────────────────────
            title = FONT_LG.render("Parameter Sweep Configuration", True, FG_COLOR)
            screen.blit(title, (SW // 2 - title.get_width() // 2, 15))

            # 1. Sweep parameter dropdown
            lbl = FONT.render("Parameter to Sweep:", True, FG_COLOR)
            screen.blit(lbl, (20, 65))
            dropdown_rect = pygame.Rect(20, 90, 280, 30)
            pygame.draw.rect(screen, INPUT_BG, dropdown_rect, border_radius=5)
            pygame.draw.rect(screen, ACCENT if dropdown_open else FG_COLOR, dropdown_rect, 2, border_radius=5)
            sel_name = SWEEPABLE_PARAMS[selected_param_idx][1]
            sel_surf = FONT.render(f"▼ {sel_name}", True, FG_COLOR)
            screen.blit(sel_surf, (25, 93))

            if dropdown_open:
                for i, (key, display) in enumerate(SWEEPABLE_PARAMS):
                    opt_rect = pygame.Rect(20, 125 + i * 28, 280, 26)
                    col = ACCENT if i == selected_param_idx else INPUT_BG
                    pygame.draw.rect(screen, col, opt_rect, border_radius=3)
                    pygame.draw.rect(screen, pygame.Color(80, 80, 120), opt_rect, 1, border_radius=3)
                    opt_surf = FONT_SM.render(f"  {display} ({key})", True, FG_COLOR)
                    screen.blit(opt_surf, (25, 127 + i * 28))

            # 2. Values to test
            vlbl = FONT.render("Values (comma-separated):", True, FG_COLOR)
            screen.blit(vlbl, (20, 135))
            values_rect = pygame.Rect(20, 160, SW - 40, 30)
            bc = INPUT_ACT if active_field == "values" else INPUT_BG
            pygame.draw.rect(screen, bc, values_rect, border_radius=5)
            pygame.draw.rect(screen, ACCENT if active_field == "values" else FG_COLOR, values_rect, 2, border_radius=5)
            vsurf = FONT.render(values_text + ("│" if active_field == "values" else ""), True, FG_COLOR)
            screen.blit(vsurf, (25, 163))

            # 3. Runs per config
            rlbl = FONT.render("Runs per config:", True, FG_COLOR)
            screen.blit(rlbl, (20, 202))
            runs_rect = pygame.Rect(200, 200, 100, 28)
            bc = INPUT_ACT if active_field == "runs" else INPUT_BG
            pygame.draw.rect(screen, bc, runs_rect, border_radius=5)
            pygame.draw.rect(screen, ACCENT if active_field == "runs" else FG_COLOR, runs_rect, 2, border_radius=5)
            rsurf = FONT.render(runs_text + ("│" if active_field == "runs" else ""), True, FG_COLOR)
            screen.blit(rsurf, (205, 203))

            hint = FONT_TINY.render("(each config runs N times with fresh random mazes, for both Q-Learning and SARSA)", True, pygame.Color(140, 140, 170))
            screen.blit(hint, (310, 205))

            # 4. Base parameters
            bplbl = FONT.render("Base Parameters (defaults for non-swept params):", True, FG_COLOR)
            screen.blit(bplbl, (20, 245))

            col_x = 20
            row_y_start = 270
            row_h = 36
            items_per_col = 6
            col_w = 260
            swept_key = SWEEPABLE_PARAMS[selected_param_idx][0]

            for i, bf in enumerate(base_fields):
                ci = i // items_per_col
                ri = i % items_per_col
                fx = col_x + ci * col_w
                fy = row_y_start + ri * row_h

                is_swept = (bf["key"] == swept_key)
                color = pygame.Color(255, 200, 50) if is_swept else FG_COLOR
                lbl_s = FONT_SM.render(f"{bf['display']}:", True, color)
                screen.blit(lbl_s, (fx, fy + 3))

                inp_rect = pygame.Rect(fx + 140, fy, 100, 26)
                is_active = isinstance(active_field, tuple) and active_field == ("base", i)
                bc = INPUT_ACT if is_active else INPUT_BG
                if is_swept:
                    bc = pygame.Color(60, 50, 20)
                pygame.draw.rect(screen, bc, inp_rect, border_radius=4)
                pygame.draw.rect(screen, ACCENT if is_active else pygame.Color(100, 100, 140), inp_rect, 1, border_radius=4)

                display_text = bf["text"]
                if is_swept:
                    display_text = "(swept)"
                elif is_active:
                    display_text += "│"
                ts = FONT_SM.render(display_text, True, FG_COLOR)
                screen.blit(ts, (fx + 145, fy + 4))

            # Extra fields
            ex_start_y = row_y_start + items_per_col * row_h + 20
            for i, ef in enumerate(extra_fields):
                fx = col_x
                fy = ex_start_y + i * row_h
                lbl_s = FONT_SM.render(f"{ef['display']}:", True, FG_COLOR)
                screen.blit(lbl_s, (fx, fy + 3))
                inp_rect = pygame.Rect(fx + 140, fy, 100, 26)
                is_active = isinstance(active_field, tuple) and active_field == ("extra", i)
                bc = INPUT_ACT if is_active else INPUT_BG
                pygame.draw.rect(screen, bc, inp_rect, border_radius=4)
                pygame.draw.rect(screen, ACCENT if is_active else pygame.Color(100, 100, 140), inp_rect, 1, border_radius=4)
                display_text = ef["text"] + ("│" if is_active else "")
                ts = FONT_SM.render(display_text, True, FG_COLOR)
                screen.blit(ts, (fx + 145, fy + 4))

            # Preview
            cfgs, _ = build_configs()
            preview = FONT_SM.render(f"Preview: {len(cfgs)} configs × {runs_text} runs × 2 algos = {len(cfgs) * max(1, int(runs_text) if runs_text.isdigit() else 1) * 2} total trainings", True, pygame.Color(150, 255, 150))
            screen.blit(preview, (20, SH - 100))

            # Start button
            btn_rect = pygame.Rect(SW // 2 - 120, SH - 70, 240, 50)
            pygame.draw.rect(screen, ACCENT, btn_rect, border_radius=10)
            btn_text = FONT.render("Start Sweep", True, FG_COLOR)
            screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width() // 2, btn_rect.centery - btn_text.get_height() // 2))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN ENTRY POINT                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    if "--last" in sys.argv:
        # Just re-visualize the latest sweep
        data, path = load_latest_sweep()
        if data is None:
            print(f"No sweep results found in {SWEEP_FOLDER}/")
            sys.exit(1)
        print(f"Loaded: {path}")
        visualize_sweep(data, save_png="--save" in sys.argv)
    else:
        run_config_ui()
