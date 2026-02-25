import random
import pygame
import numpy as np
import csv
import json
import shutil
import os
import time
from InputBox import InputBox
from Maze import Maze
from Monitors import Monitors
from Button import Button
from Agent import Agent
from Button_On_Off import Button_On_Off
from Drop_Down_Menu import Drop_Down_Menu
from Algorithm_Dropdown import Algorithm_Dropdown
from NonRL_Algorithms import get_algorithm
from NonRL_Visualizer import NonRL_Visualizer

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PYGAME SETUP                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

pygame.init()

# Default window size (can be modified here)
SCREEN_WIDTH = 1280   # Change this to your preferred width
SCREEN_HEIGHT = 720    # Change this to your preferred height

# Create resizable window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

# Create a clock object to manage the frame rate 
clock = pygame.time.Clock()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         FONTS, COLORS & IMAGES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Fonts
TITLE_FONT = pygame.font.Font(None,72)
HYPERPARAMETERS_FONT = pygame.font.Font(None, 24)
BUTTON_FONT = pygame.font.Font(None, 30)
BUTTON_FONT_INFLATED = pygame.font.Font(None, 32)
INPUT_BOX_FONT = pygame.font.Font(None, 28)

# Colors
# ACTIVE_COLOR = pygame.Color(193, 73, 83)
# INACTIVE_COLOR = pygame.Color(230, 170, 104)
ACTIVE_COLOR = pygame.Color(3, 29, 68)
INACTIVE_COLOR = pygame.Color(255, 255, 255)
TEXT_SAVED_COLOR = pygame.Color(178, 255, 169)

# Load and scale agent image
agent_img = pygame.image.load("Agent.png").convert_alpha()
agent_img = pygame.transform.smoothscale(agent_img, (32, 32))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIG — DEFAULT SETTINGS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FPS = 200 # FPS for Pygame display

rewardForFinish = 50 # Reward for reaching the origin
rewardForValidMove = -1 # Penalty for each valid move

EPISODES = 5000 
CURRENT_EPISODE = 0 
max_steps = 100 # Max steps per episode
gamma = 1.0 # Discount factor for future rewards

numOfReturns = 0
firstFind = False
firstEpisode = None
FindOriginEpisodes = [] # Record of episodes where agent found the origin (the episode number)
steps_per_frame = 1  # increase to speed up training visual
final_path = []  # Track the agent's path in the last successful episode

# epsilon/alpha decay (monotonic with floors)
EPS0, EPS_MIN, EPS_DECAY = 0.9, 0.05, 0.9995 # initial epsilon, min epsilon, decay rate
ALPHA0, ALPHA_MIN, ALPHA_DECAY = 0.72, 0.10, 0.997 # initial alpha, min alpha, decay rate

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MAZE INITIALIZATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

maze = None
maze_layout = None
agent = None
Q = None # Q-Table
HeatTable = None

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN PROGRAM VARIABLES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# constants to reuse
BASE_RECT_SIZE = 100 # base size of maze cells
HMARGIN = 10 # horizontal margin
VMARGIN = 10 # vertical margin

# Initialize monitors
screenArray = Monitors()

# Set initial active monitor - start with mode selection
activeMonitor = "Mode_Selection"

# Create NonRL_Results folder if it doesn't exist
if not os.path.exists("NonRL_Results"):
    os.makedirs("NonRL_Results", exist_ok=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FLAGS & STATE VARIABLES                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Main loop flag
running = True
# Flags for settings
Record_HeatMap = True 
selected_algorithm = "Q-Learning"  # Default algorithm
selected_nonrl_algorithm = "BFS"  # Default non-RL algorithm
nonrl_steps_per_second = 10  # Speed for non-RL visualization
nonrl_mode = False  # Flag to track if in non-RL mode
nonrl_solver = None  # Current non-RL algorithm solver
nonrl_visualizer = None  # NonRL visualization manager
record_mode = "ONE"  # "ONE" or "EVERY" 
comparison_running = False  # Flag for automated comparison
comparison_algorithms = []  # List of algorithms to run in comparison
comparison_current_index = 0  # Current algorithm index in comparison
comparison_results = []  # Store results from all algorithms
comparison_visualizers = {}  # Store visualizers for each algorithm {algo_name: visualizer}
comparison_trainers = {}  # Store training coroutines {algo_name: trainer}
comparison_agents = {}  # Store agents for each algorithm {algo_name: agent}
comparison_q_tables = {}  # Store Q-tables {algo_name: Q}
comparison_heat_tables = {}  # Store HeatTables for each RL algorithm {algo_name: HeatTable}
comparison_find_episodes = {}  # Store FindOriginEpisodes for each RL algorithm {algo_name: episodes}
comparison_final_paths = {}  # Store final_path for each RL algorithm {algo_name: path}
comparison_num_returns = {}  # Store numOfReturns for each RL algorithm {algo_name: count}
comparison_first_finds = {}  # Store first find info {algo_name: (bool, episode)}
comparison_expanded_view = None  # Which algorithm is expanded (None = grid view)
comparison_visualization_paused = False  # F key - pause rendering only
comparison_training_paused = False  # T key - pause computation
comparison_show_settings = False  # C key - show settings overlay
comparison_speed_multiplier = 1  # Speed multiplier for comparison mode 
comparison_start_time = 0  # Time when comparison started
comparison_show_results = False  # R key - show results overlay
comparison_show_optimal_path = False  # P key - show optimal path with arrows
comparison_json_saved = False  # Guard to prevent saving JSON twice

# Multi-run state (for batching N comparisons on random mazes into one JSON)
multi_run_total = 1          # Total number of runs requested
multi_run_current = 0        # Current run index (0-based)
multi_run_all_results = []   # List of per-run comparison data dicts
multi_run_active = False     # True while a multi-run batch is in progress
multi_run_start_time = 0     # Timestamp when the entire batch started

# Flags for RL Visualisation display options
show_path = False 
show_config = False 
show_q_values = False 
training_active = False # Indicater if training is active
screen_changed = False # Flag to track when screen mode changes

trainer = None # Coroutine for training


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS — RL TRAINING                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Get valid actions for a given state
def valid_actions(state, maze):
    x, y = state
    return maze.gridStates[y][x].actions

# Get max Q-value for valid actions
def masked_max(q_row, acts):
    if not acts:  # terminal state
        return 0.0
    return max(q_row[a] for a in acts)

# Get action with max Q-value among valid actions
def masked_argmax(q_row, acts):
    if not acts:
        return None
    best = max(q_row[a] for a in acts)
    # tie-break randomly to avoid bias
    best_as = [a for a in acts if q_row[a] == best]
    return random.choice(best_as)

# Epsilon-greedy action selection
def epsilon_greedy_action(state, Q, epsilon, maze):
    acts = valid_actions(state, maze)
    if not acts:
        return None
    if random.random() < epsilon:
        return random.choice(acts)
    return masked_argmax(Q[state[1], state[0]], acts)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS — SETUP MENU                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

new_json_file_name = "default"

# Load JSON data and create input boxes for setup menu
def load_json_for_setup():
    # Use comparison template if in EVERY mode
    if record_mode == "EVERY":
        json_file = "default_comparison_data.json"
    else:
        json_file = f"JsonData/default.json"
        if not os.path.exists(json_file):
            json_file = f"default_data.json"
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        # Create default JSON if it doesn't exist
        if not os.path.exists("JsonData"):
            os.makedirs("JsonData", exist_ok=True)
        if record_mode == "EVERY":
            # Use comparison template
            with open("default_comparison_data.json", "r") as f:
                data = json.load(f)
        else:
            shutil.copy("default_data.json", "JsonData/default.json")
            with open("JsonData/default.json", "r") as f:
                data = json.load(f)
    
    return data

# Save the edited values from setup input boxes back to JSON
def save_setup_json():
    global new_json_file_name
    
    # Load existing JSON to preserve non-editable fields
    data = load_json_for_setup()
    
    # Update fields from input boxes
    editable_keys = list(data.keys())
    
    input_box_index = 0
    for key in editable_keys:
        # Skip algorithm field in EVERY mode since it doesn't exist
        if key.lower() == "algorithm" and record_mode == "EVERY":
            continue
            
        if input_box_index >= len(setup_input_boxes):
            break
            
        input_box = setup_input_boxes[input_box_index]
        
        # Check if it's an Algorithm_Dropdown
        if isinstance(input_box, Algorithm_Dropdown):
            value = input_box.get_selected_algorithm()
        else:
            value = input_box.text
        
        # Try to convert to appropriate type
        try:
            # Try to parse as JSON to handle lists, numbers, etc.
            data[key] = json.loads(value)
        except:
            # If it fails, keep as string
            data[key] = value
        
        input_box_index += 1
    
    # Get the name from the JSON data to use as filename
    json_name = data.get("name", "default")
    # Sanitize filename (remove invalid characters)
    json_name = "".join(c for c in json_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not json_name:
        json_name = "default"
    
    # Create the filename with the name from JSON
    json_file = f"JsonData/{json_name}.json"
    # save json file name for later use
    new_json_file_name = json_name
    # Write to JSON file
    try:
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved setup data to {json_file}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

# Called when algorithm is selected from dropdown
def on_algorithm_selected(algorithm_name):
    global selected_algorithm
    selected_algorithm = algorithm_name
    print(f"Selected algorithm: {algorithm_name}")

# User wants to record ONE algorithm
def select_one_algorithm(monitor):
    global record_mode
    record_mode = "ONE"
    set_active_monitor(monitor)

# User wants to record EVERY algorithm (comparison mode)
def select_every_algorithm(monitor):
    global record_mode
    record_mode = "EVERY"
    print("Record EVERY mode selected - will run all algorithms")
    # Go to Comparison_Setup_Menu to create maze
    set_active_monitor("Comparison_Setup_Menu")

# User selected RL algorithms
def select_rl_algorithms(monitor):
    global nonrl_mode
    nonrl_mode = False
    set_active_monitor(monitor)

# User selected non-RL algorithms
def select_nonrl_algorithms(monitor):
    global nonrl_mode
    nonrl_mode = True
    set_active_monitor(monitor)

# Called when non-RL algorithm is selected from dropdown
def on_nonrl_algorithm_selected(algorithm_name):
    global selected_nonrl_algorithm
    selected_nonrl_algorithm = algorithm_name
    print(f"Selected non-RL algorithm: {algorithm_name}")

# Start non-RL algorithm visualization with current settings
def start_nonrl_visualization(monitor):
    global nonrl_visualizer, nonrl_solver
    
    if maze is None:
        print("Error: No maze loaded")
        return
    
    # Create algorithm solver
    nonrl_solver = get_algorithm(
        selected_nonrl_algorithm,
        maze,
        maze.start_pos,
        maze.origin_cor
    )
    
    # Create visualizer
    nonrl_visualizer = NonRL_Visualizer(maze, nonrl_solver, agent_img)
    nonrl_visualizer.start()
    
    print(f"Starting {selected_nonrl_algorithm} visualization")
    set_active_monitor(monitor)

# Start automated comparison of all algorithms with parallel visualization
def start_automated_comparison():
    global comparison_running, comparison_algorithms, comparison_current_index, comparison_results
    global comparison_visualizers, comparison_trainers, comparison_agents, comparison_q_tables
    global activeMonitor, comparison_start_time, comparison_json_saved
    
    if maze is None:
        print("Error: No maze loaded for comparison")
        return
    
    # Define all algorithms to test
    comparison_algorithms = [
        {"type": "RL", "name": "Q-Learning"},
        {"type": "RL", "name": "SARSA"},
        {"type": "NonRL", "name": "BFS"},
        {"type": "NonRL", "name": "Wall Follower"},
        {"type": "NonRL", "name": "Random Walk"},
        {"type": "NonRL", "name": "Greedy"}
    ]
    
    comparison_current_index = 0
    comparison_results = []
    comparison_running = True
    comparison_json_saved = False
    comparison_visualizers = {}
    comparison_trainers = {}
    comparison_agents = {}
    comparison_q_tables = {}
    comparison_heat_tables = {}
    comparison_find_episodes = {}
    comparison_final_paths = {}
    comparison_num_returns = {}
    comparison_first_finds = {}
    
    # Start timing
    comparison_start_time = time.time()
    
    print(f"Starting parallel visualization of {len(comparison_algorithms)} algorithms")
    
    # Initialize all algorithms
    for algo in comparison_algorithms:
        algo_name = algo["name"]
        
        if algo["type"] == "RL":
            # Create separate agent and Q-table for each RL algorithm
            comparison_agents[algo_name] = Agent(-5, maze.gridStates, maze.start_pos)
            comparison_q_tables[algo_name] = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
            comparison_heat_tables[algo_name] = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
            comparison_find_episodes[algo_name] = []
            comparison_final_paths[algo_name] = []
            comparison_num_returns[algo_name] = 0
            comparison_first_finds[algo_name] = (False, None)
            
            # Create training coroutine
            if algo_name == "SARSA":
                comparison_trainers[algo_name] = sarsa_coroutine(
                    comparison_agents[algo_name], maze, comparison_q_tables[algo_name],
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY,
                    algo_name  # Pass algo_name for comparison tracking
                )
            else:  # Q-Learning
                comparison_trainers[algo_name] = q_learning_coroutine(
                    comparison_agents[algo_name], maze, comparison_q_tables[algo_name],
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY,
                    algo_name  # Pass algo_name for comparison tracking
                )
        else:
            # Create Non-RL visualizer
            solver = get_algorithm(algo_name, maze, maze.start_pos, maze.origin_cor)
            comparison_visualizers[algo_name] = NonRL_Visualizer(maze, solver, agent_img)
            comparison_visualizers[algo_name].start()
    
    # Switch to comparison visualization screen
    activeMonitor = "Comparison_Visualisation"
    print("Comparison visualization started - Press C for settings, P for optimal path, F to freeze, T to pause training")

# Run the next algorithm in the comparison sequence
def run_next_comparison_algorithm():
    global comparison_current_index, selected_algorithm, selected_nonrl_algorithm
    global training_active, trainer, nonrl_visualizer, nonrl_solver
    
    if comparison_current_index >= len(comparison_algorithms):
        # All algorithms completed
        finish_comparison()
        return
    
    algo = comparison_algorithms[comparison_current_index]
    print(f"\n[{comparison_current_index + 1}/{len(comparison_algorithms)}] Running {algo['name']}...")
    
    if algo["type"] == "RL":
        # Run RL algorithm
        selected_algorithm = algo["name"]
        if selected_algorithm == "SARSA":
            trainer = sarsa_coroutine(
                agent, maze, Q,
                EPISODES, max_steps, gamma,
                EPS0, EPS_MIN, EPS_DECAY,
                ALPHA0, ALPHA_MIN, ALPHA_DECAY
            )
        else:  # Q-Learning
            trainer = q_learning_coroutine(
                agent, maze, Q,
                EPISODES, max_steps, gamma,
                EPS0, EPS_MIN, EPS_DECAY,
                ALPHA0, ALPHA_MIN, ALPHA_DECAY
            )
        training_active = True
    else:
        # Run Non-RL algorithm
        selected_nonrl_algorithm = algo["name"]
        nonrl_solver = get_algorithm(
            selected_nonrl_algorithm,
            maze,
            maze.start_pos,
            maze.origin_cor
        )
        nonrl_visualizer = NonRL_Visualizer(maze, nonrl_solver, agent_img)
        nonrl_visualizer.start()

# Save comparison results and reset
def finish_comparison():
    global comparison_running
    
    comparison_running = False
    print(f"\nComparison completed! Tested {len(comparison_results)} algorithms.")
    
    # Save comparison results to JSON
    save_comparison_json()
    
    # Return to main menu
    set_active_monitor("Main_Menu")

# Compute learning curve: success rate in sliding windows across episodes.
# Returns (data_points, convergence_episode).
def _compute_learning_curve(find_episodes, total_episodes, window_size=100):
    if total_episodes <= 0:
        return [], -1
    # Ensure at least 10 data points, at most 50
    window_size = max(1, min(total_episodes // 10, max(window_size, total_episodes // 50)))
    data_points = []
    find_set = set(find_episodes)  # O(1) lookup
    for start in range(0, total_episodes, window_size):
        end = min(start + window_size, total_episodes)
        successes = sum(1 for ep in range(start + 1, end + 1) if ep in find_set)
        actual_window = end - start
        rate = round(successes / actual_window * 100, 2) if actual_window > 0 else 0.0
        data_points.append({
            "episode_range": [start + 1, end],
            "success_rate_percent": rate,
            "successes_in_window": successes
        })
    # Convergence: first window where success rate >= threshold for 3+ consecutive windows
    convergence_episode = -1
    threshold = 70.0
    consecutive_needed = 3
    for i in range(len(data_points) - consecutive_needed + 1):
        if all(data_points[i + j]["success_rate_percent"] >= threshold for j in range(consecutive_needed)):
            convergence_episode = data_points[i]["episode_range"][0]
            break
    return data_points, convergence_episode

# Build a single run's comparison data dict from comparison_results (does NOT save to file)
def _build_single_run_data():
    from datetime import datetime
    import copy

    # Load the default template
    template_path = "default_comparison_results.json"
    try:
        with open(template_path, "r") as f:
            comparison_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {template_path} not found – using empty structure")
        comparison_data = {"metadata": {}, "shared_config": {}, "summary": {}, "rankings": {}, "algorithms": {}}

    # Deep copy so we never mutate the loaded template
    comparison_data = copy.deepcopy(comparison_data)

    # Remove template info key from output
    comparison_data.pop("_template_info", None)

    # ── Separate results by type ──
    rl_results = [r for r in comparison_results if r["type"] == "RL"]
    nonrl_results = [r for r in comparison_results if r["type"] == "NonRL"]

    # ── Fill metadata ──
    total_cells = maze.maze_size_width * maze.maze_size_height
    comparison_data["metadata"] = {
        "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "maze_name": maze.name,
        "maze_size": [maze.maze_size_width, maze.maze_size_height],
        "total_maze_cells": total_cells,
        "start_position": maze.start_pos,
        "goal_position": maze.origin_cor,
        "optimal_path_length": len(maze.optimal_path),
        "optimal_path": maze.optimal_path,
        "total_algorithms_tested": len(comparison_results),
        "rl_algorithms_count": len(rl_results),
        "nonrl_algorithms_count": len(nonrl_results)
    }

    # ── Fill shared config ──
    comparison_data["shared_config"] = {
        "training": {
            "episodes": EPISODES,
            "max_steps_per_episode": max_steps,
            "reward_for_finish": rewardForFinish,
            "reward_for_valid_move": rewardForValidMove
        },
        "rl_hyperparameters": {
            "gamma": gamma,
            "epsilon_start": EPS0,
            "epsilon_min": EPS_MIN,
            "epsilon_decay": EPS_DECAY,
            "alpha_start": ALPHA0,
            "alpha_min": ALPHA_MIN,
            "alpha_decay": ALPHA_DECAY
        }
    }

    # ── Fill each algorithm's section ──
    optimal_len = len(maze.optimal_path)

    for r in rl_results:
        algo_name = r["algorithm"]
        perf = r["performance_metrics"]
        path = r["path_analysis"]
        expl = r["exploration_data"]
        find_eps = perf.get("episodes_that_found_goal", [])
        heatmap = expl.get("heatmap", [])
        total_visits = sum(sum(row) for row in heatmap) if heatmap else 0
        unique_visited = sum(1 for row in heatmap for v in row if v > 0) if heatmap else 0
        coverage = round(unique_visited / total_cells * 100, 2) if total_cells > 0 else 0.0
        avg_per_state = round(total_visits / unique_visited, 2) if unique_visited > 0 else 0.0
        path_len = path.get("final_path_length", 0)
        path_found = path_len > 0 and perf.get("total_successful_episodes", 0) > 0
        efficiency = round(optimal_len / path_len, 4) if path_len > 0 else 0.0
        extra = path_len - optimal_len if path_len > 0 and optimal_len > 0 else 0
        train_time = perf.get("training_time_seconds", 0.0)

        # Learning curve
        curve_data, convergence_ep = _compute_learning_curve(find_eps, EPISODES)
        window_sz = curve_data[0]["episode_range"][1] - curve_data[0]["episode_range"][0] + 1 if curve_data else 0

        # Build sorted visit lists from heatmap
        all_states = []
        if heatmap:
            for y_idx in range(len(heatmap)):
                for x_idx in range(len(heatmap[0])):
                    all_states.append(([x_idx, y_idx], heatmap[y_idx][x_idx]))
        most_visited = sorted(all_states, key=lambda s: s[1], reverse=True)[:10]
        least_visited = sorted([s for s in all_states if s[1] > 0], key=lambda s: s[1])[:10]

        comparison_data["algorithms"][algo_name] = {
            "type": "RL",
            "status": "completed",
            "common_metrics": {
                "path_found": path_found,
                "path_length": path_len,
                "time_to_solution_seconds": round(train_time, 2),
                "path_efficiency": efficiency,
                "extra_steps_vs_optimal": extra
            },
            "performance_metrics": {
                "first_find_episode": perf.get("first_find_episode", -1),
                "total_successful_episodes": perf.get("total_successful_episodes", 0),
                "success_rate_percent": perf.get("success_rate_percent", 0.0),
                "find_percentage": perf.get("find_percentage", 0.0),
                "episodes_that_found_goal": find_eps[:],
                "total_episodes_trained": perf.get("total_episodes_trained", EPISODES),
                "training_time_seconds": round(train_time, 2)
            },
            "learning_curve": {
                "window_size": window_sz,
                "data_points": curve_data,
                "convergence_episode": convergence_ep,
                "convergence_threshold_percent": 70.0
            },
            "path_analysis": {
                "final_path": path.get("final_path", []),
                "final_path_length": path_len,
                "optimal_path_length": optimal_len,
                "path_efficiency": efficiency,
                "extra_steps": extra,
                "reached_goal_in_final_episode": path_found
            },
            "exploration_data": {
                "heatmap": heatmap,
                "most_visited_states_top10": most_visited,
                "least_visited_states_top10": least_visited,
                "total_state_visits": total_visits,
                "unique_states_visited": unique_visited,
                "exploration_coverage_percent": coverage,
                "average_visits_per_visited_state": avg_per_state
            },
            "q_table": r.get("q_table", [])
        }

    for r in nonrl_results:
        algo_name = r["algorithm"]
        exec_m = r["execution_metrics"]
        path_a = r["path_analysis"]
        expl_d = r["exploration_data"]
        path_taken = path_a.get("path_taken", [])
        path_len = path_a.get("path_length", 0)
        exec_time = exec_m.get("execution_time_seconds", 0.0)
        success = exec_m.get("success", False)
        efficiency = round(optimal_len / path_len, 4) if path_len > 0 else 0.0
        extra = path_len - optimal_len if path_len > 0 and optimal_len > 0 else 0
        is_optimal = (path_len == optimal_len) if success else False
        unique_cells = expl_d.get("unique_cells_visited", 0)
        coverage = round(unique_cells / total_cells * 100, 2) if total_cells > 0 else 0.0
        revisited = max(0, len(path_taken) - unique_cells) if path_taken else 0

        comparison_data["algorithms"][algo_name] = {
            "type": "NonRL",
            "status": "completed",
            "common_metrics": {
                "path_found": success,
                "path_length": path_len,
                "time_to_solution_seconds": round(exec_time, 6),
                "path_efficiency": efficiency,
                "extra_steps_vs_optimal": extra
            },
            "execution_metrics": {
                "execution_time_seconds": round(exec_time, 6),
                "steps_taken": exec_m.get("steps_taken", 0),
                "nodes_explored": exec_m.get("nodes_explored", 0),
                "success": success,
                "total_comparison_time_seconds": exec_m.get("total_comparison_time", 0.0)
            },
            "path_analysis": {
                "path_taken": path_taken,
                "path_length": path_len,
                "optimal_path_length": optimal_len,
                "efficiency_ratio": round(path_len / optimal_len, 2) if optimal_len > 0 else 0.0,
                "extra_steps": extra,
                "is_optimal_path": is_optimal
            },
            "exploration_data": {
                "unique_cells_visited": unique_cells,
                "total_cells_in_maze": total_cells,
                "exploration_coverage_percent": coverage,
                "backtracking_steps": expl_d.get("backtracking_steps", 0),
                "revisited_cells_count": revisited
            },
            "algorithm_characteristics": r.get("algorithm_characteristics", {})
        }

    # ── Compute summary & rankings across all algorithms ──
    # Collect common_metrics from all completed algorithms
    all_completed = []
    for algo_name, algo_data in comparison_data["algorithms"].items():
        if algo_data.get("status") == "completed":
            cm = algo_data["common_metrics"]
            all_completed.append({
                "algorithm": algo_name,
                "type": algo_data["type"],
                "path_found": cm["path_found"],
                "path_length": cm["path_length"],
                "time_seconds": cm["time_to_solution_seconds"],
                "path_efficiency": cm["path_efficiency"]
            })

    successful = [a for a in all_completed if a["path_found"]]

    # Best RL by success rate
    best_rl = max(rl_results, key=lambda x: x["performance_metrics"]["success_rate_percent"]) if rl_results else None
    # Fastest NonRL
    successful_nonrl = [a for a in successful if a["type"] == "NonRL"]
    fastest_nonrl = min(successful_nonrl, key=lambda x: x["time_seconds"]) if successful_nonrl else None
    # Most efficient overall (shortest path among successful)
    most_efficient = min(successful, key=lambda x: x["path_length"]) if successful else None
    # Shortest path found
    shortest_path = min(successful, key=lambda x: x["path_length"]) if successful else None
    # Fastest overall
    fastest_overall = min(successful, key=lambda x: x["time_seconds"]) if successful else None

    # Average calculations
    avg_rl_success = round(sum(r["performance_metrics"]["success_rate_percent"] for r in rl_results) / len(rl_results), 2) if rl_results else 0.0
    avg_nonrl_time = round(sum(a["time_seconds"] for a in successful_nonrl) / len(successful_nonrl), 6) if successful_nonrl else 0.0
    avg_all_paths = round(sum(a["path_length"] for a in successful) / len(successful), 2) if successful else 0.0

    comparison_data["summary"] = {
        "best_rl_by_success_rate": {
            "name": best_rl["algorithm"] if best_rl else "N/A",
            "success_rate_percent": best_rl["performance_metrics"]["success_rate_percent"] if best_rl else 0.0
        },
        "fastest_nonrl": {
            "name": fastest_nonrl["algorithm"] if fastest_nonrl else "N/A",
            "execution_time_seconds": fastest_nonrl["time_seconds"] if fastest_nonrl else 0.0
        },
        "most_efficient_path": {
            "name": most_efficient["algorithm"] if most_efficient else "N/A",
            "path_length": most_efficient["path_length"] if most_efficient else 0,
            "efficiency_ratio": most_efficient["path_efficiency"] if most_efficient else 0.0
        },
        "shortest_path_found": {
            "name": shortest_path["algorithm"] if shortest_path else "N/A",
            "path_length": shortest_path["path_length"] if shortest_path else 0
        },
        "fastest_overall": {
            "name": fastest_overall["algorithm"] if fastest_overall else "N/A",
            "time_seconds": fastest_overall["time_seconds"] if fastest_overall else 0.0
        },
        "averages": {
            "rl_success_rate_percent": avg_rl_success,
            "nonrl_execution_time_seconds": avg_nonrl_time,
            "all_path_lengths": avg_all_paths
        }
    }

    # Rankings
    comparison_data["rankings"] = {
        "by_path_length": sorted(
            [{"algorithm": a["algorithm"], "path_length": a["path_length"]} for a in successful],
            key=lambda x: x["path_length"]
        ),
        "by_execution_time": sorted(
            [{"algorithm": a["algorithm"], "time_seconds": a["time_seconds"]} for a in successful],
            key=lambda x: x["time_seconds"]
        ),
        "rl_by_success_rate": sorted(
            [{"algorithm": r["algorithm"], "success_rate_percent": r["performance_metrics"]["success_rate_percent"]} for r in rl_results],
            key=lambda x: x["success_rate_percent"],
            reverse=True
        ),
        "nonrl_by_efficiency": sorted(
            [{"algorithm": a["algorithm"], "path_efficiency": a["path_efficiency"]} for a in successful_nonrl],
            key=lambda x: x["path_efficiency"],
            reverse=True
        )
    }

    return comparison_data

# Save a single comparison run to JSON file (wrapper around _build_single_run_data)
def save_comparison_json():
    from datetime import datetime

    comparison_data = _build_single_run_data()

    # ── Save to file ──
    filename = f"NonRL_Results/Comparison_{maze.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, "w") as f:
            json.dump(comparison_data, f, indent=4)
        print(f"\nSaved comparison results to {filename}")
        rl_count = len([r for r in comparison_results if r["type"] == "RL"])
        nonrl_count = len([r for r in comparison_results if r["type"] == "NonRL"])
        print(f"  - {rl_count} RL algorithms, {nonrl_count} Non-RL algorithms")
        print(f"  - Best RL: {comparison_data['summary']['best_rl_by_success_rate']['name']} "
              f"({comparison_data['summary']['best_rl_by_success_rate']['success_rate_percent']:.1f}%)")
        print(f"  - Fastest overall: {comparison_data['summary']['fastest_overall']['name']}")
        print(f"  - Shortest path: {comparison_data['summary']['shortest_path_found']['name']} "
              f"({comparison_data['summary']['shortest_path_found']['path_length']} steps)")
        
        # Also copy to ComparisonVisualization/ folder for the visualization script
        viz_folder = "ComparisonVisualization"
        os.makedirs(viz_folder, exist_ok=True)
        viz_filename = os.path.join(viz_folder, os.path.basename(filename))
        shutil.copy(filename, viz_filename)
        print(f"  - Copied to {viz_filename} (for visualize_comparison.py)")
    except Exception as e:
        print(f"Error saving comparison JSON: {e}")

# Save all multi-run batch results into a single JSON file
def save_multi_run_json():
    from datetime import datetime

    total_batch_time = time.time() - multi_run_start_time

    # Build the batch JSON structure
    batch_data = {
        "batch_metadata": {
            "total_runs": multi_run_total,
            "batch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_batch_time_seconds": round(total_batch_time, 2),
            "average_run_time_seconds": round(total_batch_time / multi_run_total, 2) if multi_run_total > 0 else 0,
            "maze_config_file": new_json_file_name + ".json" if new_json_file_name else "unknown"
        },
        "shared_config": {
            "training": {
                "episodes": EPISODES,
                "max_steps_per_episode": max_steps,
                "reward_for_finish": rewardForFinish,
                "reward_for_valid_move": rewardForValidMove
            },
            "rl_hyperparameters": {
                "gamma": gamma,
                "epsilon_start": EPS0,
                "epsilon_min": EPS_MIN,
                "epsilon_decay": EPS_DECAY,
                "alpha_start": ALPHA0,
                "alpha_min": ALPHA_MIN,
                "alpha_decay": ALPHA_DECAY
            }
        },
        "runs": []
    }

    # Add each run's data with a run index
    for i, run_data in enumerate(multi_run_all_results):
        run_entry = {
            "run_index": i + 1,
            **run_data
        }
        batch_data["runs"].append(run_entry)

    # Compute aggregate statistics across all runs
    all_algo_names = set()
    for run_data in multi_run_all_results:
        all_algo_names.update(run_data.get("algorithms", {}).keys())

    aggregate_stats = {}
    for algo_name in sorted(all_algo_names):
        path_lengths = []
        times = []
        efficiencies = []
        success_rates = []
        for run_data in multi_run_all_results:
            algo_data = run_data.get("algorithms", {}).get(algo_name)
            if algo_data and algo_data.get("status") == "completed":
                cm = algo_data["common_metrics"]
                if cm["path_found"]:
                    path_lengths.append(cm["path_length"])
                    efficiencies.append(cm["path_efficiency"])
                times.append(cm["time_to_solution_seconds"])
                if algo_data["type"] == "RL":
                    sr = algo_data.get("performance_metrics", {}).get("success_rate_percent", 0)
                    success_rates.append(sr)

        aggregate_stats[algo_name] = {
            "runs_completed": len(times),
            "runs_with_path_found": len(path_lengths),
            "avg_path_length": round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else 0,
            "min_path_length": min(path_lengths) if path_lengths else 0,
            "max_path_length": max(path_lengths) if path_lengths else 0,
            "avg_time_seconds": round(sum(times) / len(times), 4) if times else 0,
            "avg_path_efficiency": round(sum(efficiencies) / len(efficiencies), 4) if efficiencies else 0,
            "avg_success_rate_percent": round(sum(success_rates) / len(success_rates), 2) if success_rates else None
        }

    batch_data["aggregate_statistics"] = aggregate_stats

    # ── Save to file ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    maze_name = multi_run_all_results[0]["metadata"]["maze_name"] if multi_run_all_results else "Unknown"
    filename = f"NonRL_Results/MultiRun_{maze_name}_{multi_run_total}runs_{timestamp}.json"
    try:
        with open(filename, "w") as f:
            json.dump(batch_data, f, indent=4)
        print(f"\n{'='*60}")
        print(f"MULTI-RUN BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Saved {multi_run_total} runs to {filename}")
        print(f"  Total batch time: {total_batch_time:.1f}s ({total_batch_time/60:.1f} min)")
        print(f"  Average per run: {total_batch_time/multi_run_total:.1f}s")
        for algo_name, stats in aggregate_stats.items():
            avg_pl = stats['avg_path_length']
            avg_t = stats['avg_time_seconds']
            print(f"  {algo_name}: avg path={avg_pl}, avg time={avg_t:.4f}s")

        # Also copy to ComparisonVisualization/ folder
        viz_folder = "ComparisonVisualization"
        os.makedirs(viz_folder, exist_ok=True)
        viz_filename = os.path.join(viz_folder, os.path.basename(filename))
        shutil.copy(filename, viz_filename)
        print(f"  Copied to {viz_filename}")
    except Exception as e:
        print(f"Error saving multi-run JSON: {e}")

# ── Input Box Initialization ─────────────────────────────────────────────────

# Initialize input boxes for setup menu JSON editor (only first 8 variables)
def initialize_setup_input_boxes():
    setup_json_data = load_json_for_setup()
    setup_json_keys = list(setup_json_data.keys())
    setup_json_values = [str(v) for v in setup_json_data.values()]
    
    # Only create input boxes for the first 8 variables
    max_input_fields = 8
    editable_keys = setup_json_keys[:max_input_fields]
    editable_values = setup_json_values[:max_input_fields]
    
    setup_input_boxes = []
    setup_input_box_labels = []
    setup_box_width = int(SCREEN_WIDTH * 0.15)
    setup_box_height = int(SCREEN_HEIGHT * 0.04)
    setup_box_x = SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.04)
    setup_box_y_start = int(SCREEN_HEIGHT * 0.12)
    setup_box_gap = int(SCREEN_HEIGHT * 0.06)
    
    # Create algorithm dropdown for "algorithm" field (only in ONE mode)
    algorithm_dropdown = None
    
    box_index = 0  # Track actual box position
    for i, (key, value) in enumerate(zip(editable_keys, editable_values)):
        # Skip algorithm field in EVERY mode
        if key.lower() == "algorithm" and record_mode == "EVERY":
            continue
            
        if key.lower() == "algorithm" and record_mode == "ONE":
            # Create dropdown instead of input box for algorithm field
            algorithm_dropdown = Algorithm_Dropdown(
                screen,
                algorithms=["Q-Learning", "SARSA"],
                on_select_callback=on_algorithm_selected,
                button_font=INPUT_BOX_FONT,
                width=setup_box_width,
                height=setup_box_height,
                pos=(setup_box_x, setup_box_y_start + box_index * setup_box_gap)
            )
            # Set current value if it exists
            if value in ["Q-Learning", "SARSA"]:
                algorithm_dropdown.set_selected_algorithm(value)
            setup_input_boxes.append(algorithm_dropdown)  # Add dropdown to list
        else:
            input_box = InputBox(
                setup_box_x, 
                setup_box_y_start + box_index * setup_box_gap, 
                setup_box_width, 
                setup_box_height, 
                INACTIVE_COLOR, 
                ACTIVE_COLOR, 
                TEXT_SAVED_COLOR, 
                INPUT_BOX_FONT, 
                value
            )
            setup_input_boxes.append(input_box)
        
        label_surface = INPUT_BOX_FONT.render(f"{key}:", True, pygame.Color("white"))
        label_pos = (setup_box_x - int(SCREEN_WIDTH * 0.18), setup_box_y_start + box_index * setup_box_gap + 5)
        setup_input_box_labels.append((label_surface, label_pos))
        box_index += 1
    
    return setup_input_boxes, setup_input_box_labels, setup_json_keys

# Initialize setup input boxes
setup_input_boxes, setup_input_box_labels, setup_json_keys = initialize_setup_input_boxes()

# Initialize input boxes for comparison maze setup
def initialize_comparison_input_boxes():
    with open("default_comparison_data.json", "r") as f:
        comparison_data = json.load(f)
    
    comparison_keys = list(comparison_data.keys())
    comparison_values = [str(v) for v in comparison_data.values()]
    
    comparison_input_boxes = []
    comparison_input_box_labels = []
    setup_box_width = int(SCREEN_WIDTH * 0.15)
    setup_box_height = int(SCREEN_HEIGHT * 0.04)
    setup_box_x = SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.04)
    setup_box_y_start = int(SCREEN_HEIGHT * 0.12)
    setup_box_gap = int(SCREEN_HEIGHT * 0.07)
    
    for i, (key, value) in enumerate(zip(comparison_keys, comparison_values)):
        input_box = InputBox(
            setup_box_x,
            setup_box_y_start + i * setup_box_gap,
            setup_box_width,
            setup_box_height,
            INACTIVE_COLOR,
            ACTIVE_COLOR,
            TEXT_SAVED_COLOR,
            INPUT_BOX_FONT,
            value
        )
        comparison_input_boxes.append(input_box)
        
        label_surface = INPUT_BOX_FONT.render(f"{key}:", True, pygame.Color("white"))
        label_pos = (setup_box_x - int(SCREEN_WIDTH * 0.18), setup_box_y_start + i * setup_box_gap + 5)
        comparison_input_box_labels.append((label_surface, label_pos))
    
    return comparison_input_boxes, comparison_input_box_labels, comparison_keys

comparison_input_boxes, comparison_input_box_labels, comparison_json_keys = initialize_comparison_input_boxes()

# Initialize input boxes for RL algorithm parameters
def initialize_rl_settings_input_boxes():
    global EPISODES, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, gamma
    
    rl_settings_keys = ["EPISODES", "EPS0", "EPS_MIN", "EPS_DECAY", "ALPHA0", "ALPHA_MIN", "ALPHA_DECAY", "gamma"]
    rl_settings_values = [
        str(EPISODES),
        str(EPS0),
        str(EPS_MIN),
        str(EPS_DECAY),
        str(ALPHA0),
        str(ALPHA_MIN),
        str(ALPHA_DECAY),
        str(gamma)
    ]
    
    rl_input_boxes = []
    rl_input_box_labels = []
    box_width = int(SCREEN_WIDTH * 0.15)
    box_height = int(SCREEN_HEIGHT * 0.04)
    box_x = SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.04)
    box_y_start = int(SCREEN_HEIGHT * 0.15)
    box_gap = int(SCREEN_HEIGHT * 0.065)
    
    for i, (key, value) in enumerate(zip(rl_settings_keys, rl_settings_values)):
        input_box = InputBox(
            box_x,
            box_y_start + i * box_gap,
            box_width,
            box_height,
            INACTIVE_COLOR,
            ACTIVE_COLOR,
            TEXT_SAVED_COLOR,
            INPUT_BOX_FONT,
            value
        )
        rl_input_boxes.append(input_box)
        
        label_surface = INPUT_BOX_FONT.render(f"{key}:", True, pygame.Color("white"))
        label_pos = (box_x - int(SCREEN_WIDTH * 0.18), box_y_start + i * box_gap + 5)
        rl_input_box_labels.append((label_surface, label_pos))
    
    return rl_input_boxes, rl_input_box_labels, rl_settings_keys

rl_settings_input_boxes, rl_settings_input_box_labels, rl_settings_keys = initialize_rl_settings_input_boxes()

# Update RL settings input boxes with current global variable values
def refresh_rl_settings_from_globals():
    global EPISODES, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, gamma
    
    rl_settings_values = [
        str(EPISODES),
        str(EPS0),
        str(EPS_MIN),
        str(EPS_DECAY),
        str(ALPHA0),
        str(ALPHA_MIN),
        str(ALPHA_DECAY),
        str(gamma)
    ]
    
    for i, input_box in enumerate(rl_settings_input_boxes):
        if i < len(rl_settings_values):
            input_box.text = rl_settings_values[i]

# Save comparison maze setup and go to RL settings screen
def save_comparison_and_continue_to_rl_settings(monitor):
    global new_json_file_name
    
    # Load base comparison data
    with open("default_comparison_data.json", "r") as f:
        data = json.load(f)
    
    # Update from input boxes
    for i, key in enumerate(comparison_json_keys):
        if i < len(comparison_input_boxes):
            value = comparison_input_boxes[i].text
            try:
                data[key] = json.loads(value)
            except:
                data[key] = value
    
    # Save to JsonData
    json_name = data.get("name", "ComparisonMaze")
    json_name = "".join(c for c in json_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not json_name:
        json_name = "ComparisonMaze"
    
    new_json_file_name = json_name
    json_file = f"JsonData/{json_name}.json"
    
    try:
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved comparison setup to {json_file}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return
    
    # Go to RL settings screen
    set_active_monitor("Comparison_RL_Settings")

# Apply RL settings and create maze, then start comparison
def apply_rl_settings_and_load_maze(monitor):
    global EPISODES, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, gamma
    global agent, maze, Q, HeatTable
    global multi_run_total, multi_run_current, multi_run_all_results, multi_run_active, multi_run_start_time
    
    # Update RL parameters from input boxes (skip EPISODES - index 0, use NumOfEpisodes from JSON instead)
    try:
        # EPISODES will be loaded from JSON in create_maze_from_json()
        EPS0 = float(rl_settings_input_boxes[1].text)
        EPS_MIN = float(rl_settings_input_boxes[2].text)
        EPS_DECAY = float(rl_settings_input_boxes[3].text)
        ALPHA0 = float(rl_settings_input_boxes[4].text)
        ALPHA_MIN = float(rl_settings_input_boxes[5].text)
        ALPHA_DECAY = float(rl_settings_input_boxes[6].text)
        gamma = float(rl_settings_input_boxes[7].text)
    except Exception as e:
        print(f"Error parsing RL settings: {e}")
        return
    
    # Read number of runs from multi-run input
    try:
        multi_run_total = max(1, int(multi_run_input.text))
    except (ValueError, TypeError):
        multi_run_total = 1
    
    multi_run_current = 0
    multi_run_all_results = []
    multi_run_active = (multi_run_total > 1)
    multi_run_start_time = time.time()
    
    if multi_run_active:
        print(f"\n{'='*60}")
        print(f"MULTI-RUN BATCH: {multi_run_total} runs requested")
        print(f"{'='*60}")
    
    # Create maze from saved JSON (this will set EPISODES from NumOfEpisodes in JSON)
    create_maze_from_json(new_json_file_name)
    print(f"RL settings applied: EPISODES={EPISODES}, EPS0={EPS0}, gamma={gamma}")
    # Create agent
    agent = Agent(-5, maze.gridStates, maze.start_pos)
    # Initialize Q-Table
    Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
    # Initialize HeatTable
    HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
    
    if multi_run_active:
        print(f"\n--- Run {multi_run_current + 1}/{multi_run_total} ---")
    print(f"Created maze for comparison from {new_json_file_name}.json")
    
    # Start automated comparison
    start_automated_comparison()

# Save comparison maze setup and start automated comparison
def save_comparison_and_start(monitor):
    global new_json_file_name, agent, maze, Q, HeatTable
    
    # Load base comparison data
    with open("default_comparison_data.json", "r") as f:
        data = json.load(f)
    
    # Update from input boxes
    for i, key in enumerate(comparison_json_keys):
        if i < len(comparison_input_boxes):
            value = comparison_input_boxes[i].text
            try:
                data[key] = json.loads(value)
            except:
                data[key] = value
    
    # Save to JsonData
    json_name = data.get("name", "ComparisonMaze")
    json_name = "".join(c for c in json_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not json_name:
        json_name = "ComparisonMaze"
    
    new_json_file_name = json_name
    json_file = f"JsonData/{json_name}.json"
    
    try:
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved comparison setup to {json_file}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return
    
    # Create maze from saved JSON
    create_maze_from_json(new_json_file_name)
    # Create agent
    agent = Agent(-5, maze.gridStates, maze.start_pos)
    # Initialize Q-Table
    Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
    # Initialize HeatTable
    HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
    print(f"Created maze for comparison from {new_json_file_name}.json")
    
    # Start automated comparison
    start_automated_comparison()

# Save setup JSON and go to RL settings
def save_setup_and_go_to_rl_settings(monitor):
    global new_json_file_name, agent, maze, Q, HeatTable
    # Save setup JSON
    save_setup_json()
    
    # Check if we're in Non-RL mode
    if nonrl_mode:
        # For Non-RL, create maze immediately and go to algorithm selection
        create_maze_from_json(new_json_file_name)
        agent = Agent(-5, maze.gridStates, maze.start_pos)
        Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
        HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
        print(f"Created maze from {new_json_file_name}.json")
        set_active_monitor("NonRL_Algorithm_Menu")
    else:
        # For RL, go to RL settings screen
        set_active_monitor("RL_Settings")

# Apply RL settings and create maze for single algorithm training
def apply_rl_settings_and_start_training(monitor):
    global EPISODES, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, gamma
    global agent, maze, Q, HeatTable
    
    # Update RL parameters from input boxes
    try:
        EPISODES = int(rl_settings_input_boxes[0].text)
        EPS0 = float(rl_settings_input_boxes[1].text)
        EPS_MIN = float(rl_settings_input_boxes[2].text)
        EPS_DECAY = float(rl_settings_input_boxes[3].text)
        ALPHA0 = float(rl_settings_input_boxes[4].text)
        ALPHA_MIN = float(rl_settings_input_boxes[5].text)
        ALPHA_DECAY = float(rl_settings_input_boxes[6].text)
        gamma = float(rl_settings_input_boxes[7].text)
        print(f"RL settings applied: EPISODES={EPISODES}, EPS0={EPS0}, gamma={gamma}")
    except Exception as e:
        print(f"Error parsing RL settings: {e}")
        return
    
    # Create maze from saved JSON
    create_maze_from_json(new_json_file_name)
    # Create agent with new maze
    agent = Agent(-5, maze.gridStates, maze.start_pos)
    # Initialize Q-Table with new maze size
    Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
    # Initialize HeatTable with new maze size
    HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
    print(f"Created maze and agent from {new_json_file_name}.json")
    # Go to main menu
    set_active_monitor("Main_Menu")

# Save JSON and start comparison
def save_create_maze_and_return_to_main(monitor):
    global new_json_file_name, agent, maze, Q, HeatTable
    # Save setup JSON
    save_setup_json()
    # Create maze from saved JSON
    create_maze_from_json(new_json_file_name)
    # Create agent with new maze
    agent = Agent(-5,maze.gridStates,maze.start_pos) # Creating Agent
    # Initialize Q-Table with new maze size
    Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float) # Q-Table initialization
    # Initialize HeatTable with new maze size
    HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)] # 2D list to store heatmap data
    print(f"Created maze and agent from {new_json_file_name}.json")
    # Set active monitor based on mode
    if nonrl_mode:
        set_active_monitor("NonRL_Algorithm_Menu")
    else:
        set_active_monitor(monitor)

# Create maze from JSON data
def create_maze_from_json(json_file_name):
    global maze, EPISODES, max_steps, selected_algorithm
    json_file = f"JsonData/{json_file_name}.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    maze_name = data.get("name", "MAZE_NAME_NOT_FOUND")
    maze_width = data.get("MazeSize", [3, 3])[0]
    maze_height = data.get("MazeSize", [3, 3])[1]
    origin_start_pos = [1,1] # default origin start pos
    rewardForFinish = data.get("rewardForFinish", 50)
    rewardForValidMove = data.get("rewardForValidMove", -1)
    # Load algorithm from JSON
    algorithm = data.get("algorithm", "Q-Learning")
    if algorithm in ["Q-Learning", "SARSA"]:
        selected_algorithm = algorithm
        print(f"Algorithm set to: {selected_algorithm}")
    # Load training parameters from JSON
    num_episodes = data.get("NumOfEpisodes", -1)
    # Ensure num_episodes is an integer
    try:
        num_episodes = int(num_episodes) if not isinstance(num_episodes, int) else num_episodes
    except (ValueError, TypeError):
        num_episodes = -1
    
    if num_episodes > 0:
        EPISODES = num_episodes
    else:
        EPISODES = 99
    
    steps_per_episode = data.get("stepsPerEpisode", -1)
    # Ensure steps_per_episode is an integer
    try:
        steps_per_episode = int(steps_per_episode) if not isinstance(steps_per_episode, int) else steps_per_episode
    except (ValueError, TypeError):
        steps_per_episode = -1
    
    if steps_per_episode > 0:
        max_steps = steps_per_episode
    else:
        max_steps = 99
    # max_steps can be added to JSON if needed
    # Create Maze
    maze = Maze(maze_name, screen, SCREEN_WIDTH, SCREEN_HEIGHT, maze_width, maze_height, origin_start_pos)
    # Make all arrows to point in the dirrection of the origin (necessary in order to make the random suffle work)
    maze.create_default()
    # How many random steps is origin going to take to shuffle the maze (this function shuffles the maze)
    maze.random_sequence(5000000)
    # Carve walls from arrows (necessary in order to make the random suffle work)
    maze.carve_walls_from_arrows()
    # Calculating the starting position of the agent (algorithm)
    maze.cal_init_pos()
    # Create optimal path from start to origin
    maze.create_optimal_path(maze.start_pos)
    # Create grid states for RL
    maze.create_grid_states(rewardForFinish,rewardForValidMove)

# Initialize input boxes for settings menu
def initialize_settings_input_boxes():
    input_boxes = []
    input_box_label = []
    input_box_width = int(SCREEN_WIDTH * 0.1)
    input_box_height = int(SCREEN_HEIGHT * 0.04)
    input_box_x = SCREEN_WIDTH // 4 - int(SCREEN_WIDTH * 0.1)
    input_box_y_start = SCREEN_HEIGHT // 4 - int(SCREEN_HEIGHT * 0.12)
    input_box_gap = int(SCREEN_HEIGHT * 0.09)
    
    labels = [
        "Gamma",
        "Epsilon0",
        "Epsilon Min",
        "Epsilon Decay",
        "Alpha0",
        "Alpha Min",
        "Alpha Decay",
        "Steps per Frame (Visual Speed)",
        "FPS"
    ]
    variables = [
        gamma, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, steps_per_frame, FPS
    ]
    default_values = [str(var) for var in variables]
    
    current_x = input_box_x
    current_y = input_box_y_start
    column_gap = int(SCREEN_WIDTH * 0.12)
    
    for i, label in enumerate(labels):
        if current_y + input_box_height > SCREEN_HEIGHT:
            current_x += input_box_width + column_gap
            current_y = input_box_y_start
        
        input_box = InputBox(current_x, current_y, input_box_width, input_box_height, INACTIVE_COLOR, ACTIVE_COLOR, TEXT_SAVED_COLOR, INPUT_BOX_FONT, default_values[i], variable=variables[i])
        input_boxes.append(input_box)
        label_surface = INPUT_BOX_FONT.render(label, True, pygame.Color("white"))
        label_pos = (current_x, current_y - input_box_height + 5)
        input_box_label.append((label_surface, label_pos))
        
        current_y += input_box_gap
    
    return input_boxes, input_box_label

# Initialize settings input boxes
input_boxes, input_box_label = initialize_settings_input_boxes()

# Apply values from settings input boxes
def apply_input_box_values():
    global gamma, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, steps_per_frame, FPS
    mapping = [
        ("float", "gamma"),
        ("float", "EPS0"),
        ("float", "EPS_MIN"),
        ("float", "EPS_DECAY"),
        ("float", "ALPHA0"),
        ("float", "ALPHA_MIN"),
        ("float", "ALPHA_DECAY"),
        ("int", "steps_per_frame"),
        ("int", "FPS")
    ]
    for i, box in enumerate(input_boxes):
        val = box.variable
        if val is None:
            continue
        typ, name = mapping[i]
        val = int(val) if typ == "int" else float(val)

        if name == "gamma": gamma = val
        elif name == "EPS0": EPS0 = val
        elif name == "EPS_MIN": EPS_MIN = val
        elif name == "EPS_DECAY": EPS_DECAY = val
        elif name == "ALPHA0": ALPHA0 = val
        elif name == "ALPHA_MIN": ALPHA_MIN = val
        elif name == "ALPHA_DECAY": ALPHA_DECAY = val
        elif name == "steps_per_frame": steps_per_frame = val
        elif name == "FPS": FPS = val



# TODO:
#     : Create seed for random generator of maze shuffling from JSON setup



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      COROUTINE TRACKER INITIALIZER                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Initialize tracking structures for RL coroutines.
# Returns (heat_table, find_episodes, num_returns_tracker, first_find_tracker, final_path_tracker).
# Pass algo_name=None when not in comparison mode.
def _init_coroutine_trackers(algo_name, maze):
    if algo_name is not None:
        heat_table = comparison_heat_tables.get(algo_name, [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)])
        find_episodes = comparison_find_episodes.get(algo_name, [])
        num_returns_tracker = {"count": comparison_num_returns.get(algo_name, 0)}
        first_find = comparison_first_finds.get(algo_name, (False, None))
        first_find_tracker = {"found": first_find[0], "episode": first_find[1]}
        final_path_tracker = {"path": comparison_final_paths.get(algo_name, [])}
    else:
        heat_table = HeatTable
        find_episodes = FindOriginEpisodes
        num_returns_tracker = None
        first_find_tracker = None
        final_path_tracker = None
    return heat_table, find_episodes, num_returns_tracker, first_find_tracker, final_path_tracker

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        Q-LEARNING COROUTINE                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def q_learning_coroutine(agent, maze, Q,
                         EPISODES, max_steps, gamma,
                         EPS0, EPS_MIN, EPS_DECAY,
                         ALPHA0, ALPHA_MIN, ALPHA_DECAY,
                         algo_name=None):  # Optional: for comparison mode tracking
    
    # Determine if we're in comparison mode
    is_comparison = algo_name is not None

    # Initialize tracking structures
    heat_table, find_episodes, num_returns_tracker, first_find_tracker, final_path_tracker = _init_coroutine_trackers(algo_name, maze)

    for episode in range(EPISODES):
        if not is_comparison:
            global CURRENT_EPISODE
            global firstFind
            global numOfReturns
            global final_path
            CURRENT_EPISODE = episode + 1
            
        epsilon = max(EPS_MIN, EPS0 * (EPS_DECAY ** episode))
        alpha   = max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** episode))

        agent.reset()
        state = agent.activeState[:]
        action = epsilon_greedy_action(state, Q, epsilon, maze)
        
        # Track path for this episode
        current_path = [state[:]]
        episode_reached_goal = False

        # if episode % 100 == 0 and Record_HeatMap:
        #     with open(f"HeatMap{CURRENT_EPISODE}.csv", "w", newline="", encoding="utf-8") as f:
        #         writer = csv.writer(f)
        #         writer.writerows(HeatTable)


        for t in range(max_steps):
            if action is None:  # terminal state (goal)
                if is_comparison:
                    if not first_find_tracker["found"]:
                        first_find_tracker["found"] = True
                        first_find_tracker["episode"] = episode + 1
                        comparison_first_finds[algo_name] = (True, episode + 1)
                    num_returns_tracker["count"] += 1
                    comparison_num_returns[algo_name] = num_returns_tracker["count"]
                    find_episodes.append(episode + 1)
                    comparison_find_episodes[algo_name] = find_episodes
                    final_path_tracker["path"] = current_path[:]
                    comparison_final_paths[algo_name] = current_path[:]
                else:
                    if not firstFind:
                        global firstEpisode
                        firstEpisode = CURRENT_EPISODE
                        firstFind = True
                    numOfReturns += 1
                    FindOriginEpisodes.append(CURRENT_EPISODE)
                    final_path = current_path[:]
                episode_reached_goal = True
                break

            reward, next_state = agent.ProcessNextAction(action)
            x, y   = state
            nx, ny = next_state
            if Record_HeatMap or is_comparison:
                heat_table[state[1]][state[0]] += 1
                if is_comparison:
                    comparison_heat_tables[algo_name] = heat_table
            next_acts = valid_actions(next_state, maze)
            target = reward + gamma * masked_max(Q[ny, nx], next_acts)
            Q[y, x, action] += alpha * (target - Q[y, x, action])

            state  = next_state
            action = epsilon_greedy_action(state, Q, epsilon, maze)
            
            # Track the path
            current_path.append(state[:])
            
            # here agent.activeState has been updated -> sprite will move
            yield  # let Pygame update one frame

        # optional: yield between episodes too
        # yield

    print("Training coroutine finished")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          SARSA COROUTINE                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def sarsa_coroutine(agent, maze, Q,
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY,
                    algo_name=None):  # Optional: for comparison mode tracking
    
    # Determine if we're in comparison mode
    is_comparison = algo_name is not None

    # Initialize tracking structures
    heat_table, find_episodes, num_returns_tracker, first_find_tracker, final_path_tracker = _init_coroutine_trackers(algo_name, maze)

    for episode in range(EPISODES):
        if not is_comparison:
            global CURRENT_EPISODE
            global firstFind
            global numOfReturns
            global final_path
            CURRENT_EPISODE = episode + 1
            
        epsilon = max(EPS_MIN, EPS0 * (EPS_DECAY ** episode))
        alpha   = max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** episode))

        agent.reset()
        state = agent.activeState[:]
        action = epsilon_greedy_action(state, Q, epsilon, maze)
        
        # Track path for this episode
        current_path = [state[:]]
        episode_reached_goal = False

        for t in range(max_steps):
            if action is None:  # terminal state (goal)
                if is_comparison:
                    if not first_find_tracker["found"]:
                        first_find_tracker["found"] = True
                        first_find_tracker["episode"] = episode + 1
                        comparison_first_finds[algo_name] = (True, episode + 1)
                    num_returns_tracker["count"] += 1
                    comparison_num_returns[algo_name] = num_returns_tracker["count"]
                    find_episodes.append(episode + 1)
                    comparison_find_episodes[algo_name] = find_episodes
                    final_path_tracker["path"] = current_path[:]
                    comparison_final_paths[algo_name] = current_path[:]
                else:
                    if not firstFind:
                        global firstEpisode
                        firstEpisode = CURRENT_EPISODE
                        firstFind = True
                    numOfReturns += 1
                    FindOriginEpisodes.append(CURRENT_EPISODE)
                    final_path = current_path[:]
                episode_reached_goal = True
                break

            reward, next_state = agent.ProcessNextAction(action)
            x, y   = state
            nx, ny = next_state
            if Record_HeatMap or is_comparison:
                heat_table[state[1]][state[0]] += 1
                if is_comparison:
                    comparison_heat_tables[algo_name] = heat_table
            
            # SARSA: Choose next action before update
            next_action = epsilon_greedy_action(next_state, Q, epsilon, maze)
            
            # SARSA update: Use actual next action (not max)
            if next_action is None:
                target = reward  # Terminal state
            else:
                target = reward + gamma * Q[ny, nx, next_action]
            
            Q[y, x, action] += alpha * (target - Q[y, x, action])

            state  = next_state
            action = next_action  # SARSA uses the action we already chose
            
            # Track the path
            current_path.append(state[:])
            
            # here agent.activeState has been updated -> sprite will move
            yield  # let Pygame update one frame

        # optional: yield between episodes too
        # yield

    print("Training coroutine finished")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           DISPLAY HELPERS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def showConfigValuesOnScreen():
    info_lines = [
        f"Current Episode: {CURRENT_EPISODE}/{EPISODES}",
        f"Max Steps/Episode: {max_steps}",
        f"Gamma: {gamma:.2f}",
        f"Epsilon0: {EPS0:.2f}, Epsilon Min: {EPS_MIN:.2f}, Epsilon Decay: {EPS_DECAY:.4f}",
        f"Current Epsilon: {max(EPS_MIN, EPS0 * (EPS_DECAY ** (CURRENT_EPISODE-1))):.4f}",
        f"Alpha0: {ALPHA0:.2f}, Alpha Min: {ALPHA_MIN:.2f}, Alpha Decay: {ALPHA_DECAY:.4f}",
        f"Current Alpha: {max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** (CURRENT_EPISODE-1))):.4f}",
        f"Number of Returns to Origin: {numOfReturns}",
        f"First Find Episode: {firstEpisode if firstFind else 'N/A'}",
        f"Agent Total Reward: {agent.activeReward}"
    ]

    for i, line in enumerate(info_lines):
        text_surf = HYPERPARAMETERS_FONT.render(line, True, pygame.Color("white"))
        screen.blit(text_surf, (10, 10 + i * 30))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     JSON DATA SAVING (after training)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_json_data(maze,
                description="Test description",
                max_episodes=-1,
                steps_per_episode=-1,
                first_find=-1,
                num_of_returns=-1,
                FindOriginEpisodes=[],
                find_not_find_percentage=[-1,-1],
                heatmap=[],
                final_path=[],
                optimal_path=[],
                final_path_num_of_steps=-1,
                optimal_path_num_of_steps=-1,
                Q_table=None,
                gamma_val=1.0,
                eps0=0.9,
                eps_min=0.05,
                eps_decay=0.9995,
                alpha0=0.72,
                alpha_min=0.10,
                alpha_decay=0.997
                ):
    # Load existing JSON to preserve setup data
    json_file = f"JsonData/{new_json_file_name}.json"
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        # If no existing file, create new data
        data = {
            "name": maze.name,
            "description": description
        }
    
    # Update only the training result fields
    data["firstFind"] = first_find
    data["numOfReturns"] = num_of_returns
    data["FindOriginEpisodes"] = FindOriginEpisodes
    data["FindNotFindPercentage"] = find_not_find_percentage
    data["HEATMAP"] = heatmap
    data["FinalPath"] = final_path
    data["OptimalPath"] = optimal_path
    data["FinalPath_numOfSteps"] = final_path_num_of_steps
    data["OptimalPath_numOfSteps"] = optimal_path_num_of_steps
    # Convert NumPy array to list for JSON serialization
    data["Q_table"] = Q_table.tolist() if Q_table is not None else None
    # Save hyperparameters used during training
    data["gamma"] = gamma_val
    data["EPS0"] = eps0
    data["EPS_MIN"] = eps_min
    data["EPS_DECAY"] = eps_decay
    data["ALPHA0"] = alpha0
    data["ALPHA_MIN"] = alpha_min
    data["ALPHA_DECAY"] = alpha_decay
    # Save back to the same JSON file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         MONITOR MANAGEMENT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def set_active_monitor(monitor):
    global activeMonitor
    activeMonitor = monitor
    
    # Refresh RL settings from global variables when entering RL settings screens
    if monitor in ["RL_Settings", "Comparison_RL_Settings"]:
        refresh_rl_settings_from_globals()

# Save non-RL algorithm results to JSON
def save_nonrl_json_data(maze, algorithm_name, path_taken, execution_time, success, nodes_explored):
    from datetime import datetime
    
    data = {
        "name": f"{maze.name}_{algorithm_name}",
        "algorithm": algorithm_name,
        "description": f"Non-RL algorithm: {algorithm_name}",
        "maze_name": maze.name,
        "maze_size": [maze.maze_size_width, maze.maze_size_height],
        "start_pos": maze.start_pos,
        "goal_pos": maze.origin_cor,
        "path_taken": path_taken,
        "path_length": len(path_taken),
        "optimal_path": maze.optimal_path,
        "optimal_path_length": len(maze.optimal_path),
        "execution_time": round(execution_time, 4),
        "success": success,
        "nodes_explored": nodes_explored,
        "steps_taken": len(path_taken),
        "efficiency_ratio": round(len(path_taken) / len(maze.optimal_path), 2) if len(maze.optimal_path) > 0 else -1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create filename
    filename = f"NonRL_Results/{maze.name}_{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved non-RL results to {filename}")
    except Exception as e:
        print(f"Error saving non-RL JSON: {e}")

# Stop training, reset agent, and return to specified monitor
def stop_training_and_return(monitor):
    global training_active, trainer, agent, CURRENT_EPISODE
    if training_active:
        training_active = False
        trainer = None
        print("Training stopped and reset")
    # Reset agent to initial position
    if agent is not None:
        agent.reset()
    # Reset episode counter
    CURRENT_EPISODE = 0
    # Return to specified monitor
    set_active_monitor(monitor)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       BUTTON CALLBACKS & HELPERS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# HeatMap toggle callback
def on_toggle_heatmap(is_on):
    global Record_HeatMap
    Record_HeatMap = is_on
    print(f"HeatMap recording: {is_on}")

# Save current hyperparameters and maze configuration as layout JSON
def save_current_as_layout(monitor):
    if maze is None:
        print("No maze loaded. Cannot save layout.")
        return
    
    # Create layout data
    layout_data = {
        "name": maze.name,
        "description": f"Layout saved from settings on {maze.name}",
        "MazeSize": [maze.maze_size_width, maze.maze_size_height],
        "origin": maze.origin_cor,
        "start_pos": maze.start_pos,
        "layout": maze.get_layout(),
        "NumOfEpisodes": EPISODES,
        "stepsPerEpisode": max_steps,
        "rewardForFinish": rewardForFinish,
        "rewardForValidMove": rewardForValidMove,
        "gamma": gamma,
        "EPS0": EPS0,
        "EPS_MIN": EPS_MIN,
        "EPS_DECAY": EPS_DECAY,
        "ALPHA0": ALPHA0,
        "ALPHA_MIN": ALPHA_MIN,
        "ALPHA_DECAY": ALPHA_DECAY
    }
    
    # Ensure MazeLayouts folder exists
    if not os.path.exists("MazeLayouts"):
        os.makedirs("MazeLayouts", exist_ok=True)
    
    # Save to MazeLayouts folder
    layout_file = f"MazeLayouts/{maze.name}_layout.json"
    try:
        with open(layout_file, "w") as f:
            json.dump(layout_data, f, indent=4)
        print(f"Saved layout to {layout_file}")
        # Refresh the dropdown menu
        maze_dropdown.refresh()
    except Exception as e:
        print(f"Error saving layout: {e}")

# Callback when a maze file is selected from dropdown (just stores selection)
def on_maze_selected(filename):
    print(f"Selected: {filename}")

# Load selected JSON and create maze, then continue based on mode
def load_maze_and_continue(monitor):
    global agent, maze, Q, HeatTable, new_json_file_name
    
    # Get selected file from dropdown
    selected_file = maze_dropdown.get_selected_file()
    if not selected_file:
        print("No maze selected")
        return
    
    # Extract filename without extension
    new_json_file_name = selected_file.replace('.json', '').replace('_layout', '')
    
    # Load and create maze from MazeLayouts JSON
    json_path = maze_dropdown.get_selected_path()
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded: {selected_file}")
        
        # Copy to JsonData if needed
        json_data_path = f"JsonData/{new_json_file_name}.json"
        if not os.path.exists(json_data_path):
            import shutil
            shutil.copy(json_path, json_data_path)
        
        # Create maze from JSON
        create_maze_from_json(new_json_file_name)
        # Create agent with new maze
        agent = Agent(-5, maze.gridStates, maze.start_pos)
        # Initialize Q-Table with new maze size
        Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
        # Initialize HeatTable with new maze size
        HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
        print(f"Created maze and agent from {selected_file}")
        
        # Navigate based on mode
        if nonrl_mode:
            set_active_monitor("NonRL_Algorithm_Menu")
        else:
            set_active_monitor("Main_Menu")
    except Exception as e:
        print(f"Error loading and creating maze: {e}")

# Load selected JSON from dropdown and navigate to Setup_Menu for editing
def load_and_edit_maze(monitor):
    global setup_input_boxes, setup_input_box_labels, setup_json_keys, new_json_file_name
    
    # Get selected file from dropdown
    selected_file = maze_dropdown.get_selected_file()
    if not selected_file:
        print("No maze selected")
        return
    
    # Load JSON from MazeLayouts folder
    json_path = maze_dropdown.get_selected_path()
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded: {selected_file}")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    # Store the filename for later use
    new_json_file_name = selected_file.replace('.json', '')
    
    # Update setup_input_boxes with loaded values
    setup_json_keys = list(data.keys())
    max_input_fields = 8
    editable_keys = setup_json_keys[:max_input_fields]
    
    # Update existing input boxes with new values
    for i, key in enumerate(editable_keys):
        if i < len(setup_input_boxes):
            setup_input_boxes[i].text = str(data[key])
            # Update label
            label_surface = INPUT_BOX_FONT.render(f"{key}:", True, pygame.Color("white"))
            label_pos = setup_input_box_labels[i][1]
            setup_input_box_labels[i] = (label_surface, label_pos)
    
    # Navigate to Setup_Menu for editing
    set_active_monitor(monitor)
    print(f"Loaded {selected_file} for editing")

# ── Button Definitions ──────────────────────────────────────────────────────

# Initialize/reinitialize all buttons with current screen size
def initialize_all_buttons():
    global start_button, settings_button, mainMenu_button, heatMap_Button_OnOFF
    global save_layout_button, save_setup_button, save_comparison_button, setup_menu_button, load_menu_button
    global rl_settings_continue_button, rl_settings_start_button
    global back_to_choose_button, load_edit_button, load_continue_button, rl_back_button, maze_dropdown
    # NEW: Buttons for new menu screens
    global mode_one_button, mode_every_button
    global algorithm_type_rl_button, algorithm_type_nonrl_button
    global nonrl_algorithm_dropdown, nonrl_continue_button, nonrl_speed_input, nonrl_start_button
    global nonrl_back_button
    global multi_run_input
    
    # BUTTONS for Mode Selection Menu (NEW FIRST SCREEN)
    mode_one_button = Button(screen,
                        "Record ONE Algorithm",
                        "Algorithm_Type_Menu",
                        select_one_algorithm,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.2),
                        int(SCREEN_HEIGHT * 0.12),
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.1), SCREEN_HEIGHT/2 - int(SCREEN_HEIGHT * 0.1)),
                        7,
                        20
                        )
    
    mode_every_button = Button(screen,
                        "Record EVERY Algorithm",
                        "Algorithm_Type_Menu",
                        select_every_algorithm,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.2),
                        int(SCREEN_HEIGHT * 0.12),
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.1), SCREEN_HEIGHT/2 + int(SCREEN_HEIGHT * 0.05)),
                        7,
                        20
                        )
    
    # BUTTONS for Algorithm Type Menu
    algorithm_type_rl_button = Button(screen,
                        "RL Algorithms",
                        "Choose_Menu",
                        select_rl_algorithms,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.18),
                        int(SCREEN_HEIGHT * 0.12),
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.09), SCREEN_HEIGHT/2 - int(SCREEN_HEIGHT * 0.1)),
                        7,
                        20
                        )
    
    algorithm_type_nonrl_button = Button(screen,
                        "Non-RL Algorithms",
                        "Choose_Menu",
                        select_nonrl_algorithms,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.18),
                        int(SCREEN_HEIGHT * 0.12),
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.09), SCREEN_HEIGHT/2 + int(SCREEN_HEIGHT * 0.05)),
                        7,
                        20
                        )
    
    # Non-RL Algorithm dropdown
    nonrl_algorithm_dropdown = Algorithm_Dropdown(
        screen,
        algorithms=["BFS", "Wall Follower", "Random Walk", "Greedy"],
        on_select_callback=on_nonrl_algorithm_selected,
        button_font=INPUT_BOX_FONT,
        width=int(SCREEN_WIDTH * 0.2),
        height=int(SCREEN_HEIGHT * 0.05),
        pos=(SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.1), SCREEN_HEIGHT // 2 - int(SCREEN_HEIGHT * 0.05))
    )
    
    # Continue button for non-RL algorithm menu
    nonrl_continue_button = Button(screen,
                        "Continue",
                        "NonRL_Speed_Settings",
                        set_active_monitor,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.15),
                        int(SCREEN_HEIGHT * 0.08),
                        (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.075), SCREEN_HEIGHT // 2 + int(SCREEN_HEIGHT * 0.12)),
                        7,
                        20
                        )
    
    # Non-RL speed input box
    nonrl_speed_input = InputBox(
        SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.075),
        SCREEN_HEIGHT // 2 + int(SCREEN_HEIGHT * 0.08),
        int(SCREEN_WIDTH * 0.15),
        int(SCREEN_HEIGHT * 0.05),
        INACTIVE_COLOR,
        ACTIVE_COLOR,
        TEXT_SAVED_COLOR,
        INPUT_BOX_FONT,
        str(nonrl_steps_per_second)
    )
    
    # Start button for non-RL visualization
    nonrl_start_button = Button(screen,
                        "Start Visualization",
                        "NonRL_Visualisation",
                        start_nonrl_visualization,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.18),
                        int(SCREEN_HEIGHT * 0.08),
                        (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.09), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
                        7,
                        20
                        )
    
    # Back button for non-RL visualization
    nonrl_back_button = Button(screen,
                        "Back to Menu",
                        "Mode_Selection",
                        set_active_monitor,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.12),
                        int(SCREEN_HEIGHT * 0.06),
                        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.14), 10),
                        7,
                        20
                        )
    
    # BUTTONS for Main Menu
    start_button = Button(screen,
                        "RL - Visualisation",
                        "RL - Visualisation",
                        set_active_monitor,
                        BUTTON_FONT,
                        BUTTON_FONT_INFLATED,
                        int(SCREEN_WIDTH * 0.15),
                        int(SCREEN_HEIGHT * 0.1), 
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.075), SCREEN_HEIGHT/2 - int(SCREEN_HEIGHT * 0.05)),
                        7,
                        20
                        )

    settings_button = Button(screen,
                        "Settings", 
                        "Settings",
                        set_active_monitor, 
                        BUTTON_FONT, 
                        BUTTON_FONT_INFLATED, 
                        int(SCREEN_WIDTH * 0.15),
                        int(SCREEN_HEIGHT * 0.1), 
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.075), SCREEN_HEIGHT/2 + int(SCREEN_HEIGHT * 0.1)),
                        7,
                        20
                        )

    # BUTTONS for Settings Menu
    mainMenu_button = Button(screen,
                        "Main Menu", 
                        "Main_Menu",
                        set_active_monitor,
                        BUTTON_FONT, 
                        BUTTON_FONT_INFLATED, 
                        int(SCREEN_WIDTH * 0.15),
                        int(SCREEN_HEIGHT * 0.1), 
                        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.075), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
                        7,
                        20
                        )

    # HeatMap Record Toggle Button
    heatMap_Button_OnOFF = Button_On_Off(
                        screen,
                        "Record HeatMap",
                        on_change_monitor=on_toggle_heatmap,
                        pos=(SCREEN_WIDTH - int(SCREEN_WIDTH * 0.18), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.12)),
                        button_font=BUTTON_FONT,
                        button_font_inflated=BUTTON_FONT_INFLATED
                        )

    # Save Layout Button for Settings
    save_layout_button = Button(
        screen,
        "Save as Layout",
        "Settings",
        save_current_as_layout,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.13),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.18), SCREEN_HEIGHT/2),
        7,
        20
    )

    # BUTTONS for Setup Menu
    save_setup_button = Button(
        screen,
        "Continue to RL Settings", 
        "Main_Menu",
        save_setup_and_go_to_rl_settings,
        BUTTON_FONT, 
        BUTTON_FONT_INFLATED, 
        int(SCREEN_WIDTH * 0.18),
        int(SCREEN_HEIGHT * 0.08), 
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.22), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
        7,
        20
    )
    
    # Save button for Comparison Setup Menu
    save_comparison_button = Button(
        screen,
        "Continue to RL Settings",
        "Main_Menu",
        save_comparison_and_continue_to_rl_settings,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.16),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.20), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
        7,
        20
    )
    
    # Input box for number of runs in multi-run comparison mode
    multi_run_input = InputBox(
        SCREEN_WIDTH // 2 + int(SCREEN_WIDTH * 0.12),
        int(SCREEN_HEIGHT * 0.83),
        int(SCREEN_WIDTH * 0.08),
        int(SCREEN_HEIGHT * 0.04),
        INACTIVE_COLOR,
        ACTIVE_COLOR,
        TEXT_SAVED_COLOR,
        INPUT_BOX_FONT,
        "1"
    )

    # Continue button for RL Settings screen
    rl_settings_continue_button = Button(
        screen,
        "Continue to Load Maze",
        "Main_Menu",
        apply_rl_settings_and_load_maze,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.18),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.22), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
        7,
        20
    )
    
    # Start button for single algorithm RL Settings screen
    rl_settings_start_button = Button(
        screen,
        "Start Training",
        "Main_Menu",
        apply_rl_settings_and_start_training,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.16),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.20), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
        7,
        20
    )

    # BUTTONS for Choose Menu
    setup_menu_button = Button(
        screen,
        "Create New Maze",
        "Setup_Menu",
        set_active_monitor,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.18),
        int(SCREEN_HEIGHT * 0.1),
        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.09), SCREEN_HEIGHT/2 - int(SCREEN_HEIGHT * 0.1)),
        7,
        20
    )

    load_menu_button = Button(
        screen,
        "Load Existing Maze",
        "Load_Menu",
        set_active_monitor,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.18),
        int(SCREEN_HEIGHT * 0.1),
        (SCREEN_WIDTH/2 - int(SCREEN_WIDTH * 0.09), SCREEN_HEIGHT/2 + int(SCREEN_HEIGHT * 0.05)),
        7,
        20
    )

    # BUTTONS for Load Menu
    back_to_choose_button = Button(
        screen,
        "Back",
        "Choose_Menu",
        set_active_monitor,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.12),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.18), SCREEN_HEIGHT - int(SCREEN_HEIGHT * 0.15)),
        7,
        20
    )

    # Drop-down menu for Load Menu
    maze_dropdown = Drop_Down_Menu(
        screen,
        folder_path="MazeLayouts",
        on_select_callback=on_maze_selected,
        button_font=INPUT_BOX_FONT,
        width=int(SCREEN_WIDTH * 0.3),
        height=int(SCREEN_HEIGHT * 0.06),
        pos=(SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.18)),
        max_visible_items=8
    )

    # Load & Edit button for Load Menu
    load_edit_button = Button(
        screen,
        "Load & Edit",
        "Setup_Menu",
        load_and_edit_maze,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.15),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.17), int(SCREEN_HEIGHT * 0.35)),
        7,
        20
    )
    
    # Load & Continue button for Load Menu (loads maze and continues to next step)
    load_continue_button = Button(
        screen,
        "Load",
        "Main_Menu",
        load_maze_and_continue,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.12),
        int(SCREEN_HEIGHT * 0.08),
        (SCREEN_WIDTH // 2 + int(SCREEN_WIDTH * 0.02), int(SCREEN_HEIGHT * 0.35)),
        7,
        20
    )

    # BUTTON for RL Visualization - Back to Main Menu
    rl_back_button = Button(
        screen,
        "Back to Menu",
        "Main_Menu",
        stop_training_and_return,
        BUTTON_FONT,
        BUTTON_FONT_INFLATED,
        int(SCREEN_WIDTH * 0.12),
        int(SCREEN_HEIGHT * 0.06),
        (SCREEN_WIDTH - int(SCREEN_WIDTH * 0.14), 10),
        7,
        20
    )

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         BUTTON INITIALIZATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

initialize_all_buttons()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           DRAW MENUS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Draw the mode selection screen (ONE vs EVERY algorithm)
def draw_Mode_Selection():
    title_surf = TITLE_FONT.render("Select Recording Mode", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    mode_one_button.draw()
    mode_every_button.draw()

# Draw the algorithm type selection screen (RL vs NOT RL)
def draw_Algorithm_Type_Menu():
    title_surf = TITLE_FONT.render("Select Algorithm Type", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    algorithm_type_rl_button.draw()
    algorithm_type_nonrl_button.draw()

# Draw the non-RL algorithm selection screen
def draw_NonRL_Algorithm_Menu():
    title_surf = TITLE_FONT.render("Select Non-RL Algorithm", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    # Draw label
    label_surf = INPUT_BOX_FONT.render("Choose Algorithm:", True, pygame.Color("white"))
    screen.blit(label_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), SCREEN_HEIGHT // 2 - int(SCREEN_HEIGHT * 0.12)))
    
    # Draw continue button
    nonrl_continue_button.draw()
    
    # Draw dropdown last so it appears on top
    nonrl_algorithm_dropdown.draw()

# Draw the speed settings screen for non-RL visualization
def draw_NonRL_Speed_Settings():
    title_surf = TITLE_FONT.render("Set Visualization Speed", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    # Draw label
    label_surf = INPUT_BOX_FONT.render("Steps per Second:", True, pygame.Color("white"))
    screen.blit(label_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), SCREEN_HEIGHT // 2 + int(SCREEN_HEIGHT * 0.02)))
    
    # Draw speed input box
    nonrl_speed_input.update()
    nonrl_speed_input.draw(screen)
    
    # Draw start button
    nonrl_start_button.draw()

# Draw the non-RL algorithm visualization
def draw_NonRL_Visualisation():
    global nonrl_visualizer
    
    if maze is None or nonrl_visualizer is None:
        error_surf = BUTTON_FONT.render("Error: No maze or visualizer loaded", True, pygame.Color("red"))
        screen.blit(error_surf, (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2))
        return
    
    # Draw maze
    maze.draw_maze(BASE_RECT_SIZE, HMARGIN, VMARGIN, False, True)
    
    # Draw path history
    nonrl_visualizer.draw_path_history(screen, BASE_RECT_SIZE, HMARGIN, VMARGIN)
    
    # Draw optimal path if toggled
    if show_path:
        maze.draw_optimal_path(HMARGIN, VMARGIN, BASE_RECT_SIZE, maze.start_pos)
    
    # Draw agent
    nonrl_visualizer.draw_agent(screen, BASE_RECT_SIZE, HMARGIN, VMARGIN)
    
    # Draw statistics
    stats = nonrl_visualizer.get_stats()
    info_lines = [
        f"Algorithm: {selected_nonrl_algorithm}",
        f"Steps Taken: {stats['steps_taken']}",
        f"Path Length: {stats['path_length']}",
        f"Nodes Explored: {stats['nodes_explored']}",
        f"Finished: {'Yes' if stats['is_finished'] else 'No'}",
        f"Success: {'Yes' if stats['success'] else 'No'}",
        f"Time: {stats['execution_time']:.4f}s" if stats['execution_time'] > 0 else "Time: Running..."
    ]
    
    for i, line in enumerate(info_lines):
        text_surf = HYPERPARAMETERS_FONT.render(line, True, pygame.Color("white"))
        screen.blit(text_surf, (10, 10 + i * 30))
    
    # Draw back button
    nonrl_back_button.draw()

def draw_RL_Visualisation():
    maze.draw_maze(BASE_RECT_SIZE, HMARGIN, VMARGIN, False, True)
    maze.draw_agent(agent, BASE_RECT_SIZE, HMARGIN, VMARGIN, agent_img)
    # draw optimal path if toggled
    if show_path:
        maze.draw_optimal_path(HMARGIN,VMARGIN,BASE_RECT_SIZE,maze.start_pos)
    # show config values if toggled
    if show_config:
        showConfigValuesOnScreen()
    # show Q-values if toggled
    if show_q_values:
        maze.draw_q_values(Q, BASE_RECT_SIZE, HMARGIN, VMARGIN)
    # Draw back button
    rl_back_button.draw()
    # Draw back button
    rl_back_button.draw()

def draw_Main_Menu():
    start_button.draw()
    settings_button.draw()

def draw_Settings_Menu():

    mainMenu_button.draw()
    heatMap_Button_OnOFF.draw()
    save_layout_button.draw()
    # Draw input boxes and labels
    for i, input_box in enumerate(comparison_input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = comparison_input_box_labels[i]
        screen.blit(label_surf, label_pos)

# Draw the RL algorithm settings screen for comparison mode
def draw_Comparison_RL_Settings():
    # Display title
    title_surf = TITLE_FONT.render("RL Algorithm Settings", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Configure parameters for Q-Learning and SARSA algorithms:", True, pygame.Color("white"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.25), int(SCREEN_HEIGHT * 0.08)))
    
    note_surf = INPUT_BOX_FONT.render("(Episodes are set in previous comparison settings)", True, pygame.Color("gray"))
    screen.blit(note_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.22), int(SCREEN_HEIGHT * 0.12)))
    
    # Draw input boxes and labels (skip index 0 which is EPISODES)
    for i, input_box in enumerate(rl_settings_input_boxes):
        if i == 0:  # Skip EPISODES input box in comparison mode
            continue
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = rl_settings_input_box_labels[i]
        screen.blit(label_surf, label_pos)
    
    # Draw continue button
    rl_settings_continue_button.draw()
    
    # Draw multi-run input box
    run_label = INPUT_BOX_FONT.render("Number of Runs:", True, pygame.Color("white"))
    screen.blit(run_label, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.05), int(SCREEN_HEIGHT * 0.84)))
    multi_run_input.update()
    multi_run_input.draw(screen)
    run_hint = HYPERPARAMETERS_FONT.render("(each run generates a new random maze)", True, pygame.Color("gray"))
    screen.blit(run_hint, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.05), int(SCREEN_HEIGHT * 0.89)))

# Draw the RL algorithm settings screen for single training
def draw_RL_Settings():
    # Display title
    title_surf = TITLE_FONT.render("RL Algorithm Settings", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Configure parameters for training:", True, pygame.Color("white"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.18), int(SCREEN_HEIGHT * 0.08)))
    
    # Draw input boxes and labels
    for i, input_box in enumerate(rl_settings_input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = rl_settings_input_box_labels[i]
        screen.blit(label_surf, label_pos)
    
    # Draw start button
    rl_settings_start_button.draw()

def draw_Setup_Menu():
    # Display title
    title_surf = TITLE_FONT.render("JSON Setup Editor", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.13), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw input boxes and labels (but not dropdowns yet)
    dropdown_to_draw = None
    dropdown_label_index = -1
    for i, input_box in enumerate(setup_input_boxes):
        # Check if it's a dropdown or input box
        if isinstance(input_box, Algorithm_Dropdown):
            # Save dropdown to draw last
            dropdown_to_draw = input_box
            dropdown_label_index = i
        else:
            input_box.update()
            input_box.draw(screen)
            label_surf, label_pos = setup_input_box_labels[i]
            screen.blit(label_surf, label_pos)
    
    # Draw buttons
    save_setup_button.draw()
    
    # Draw dropdown last so it appears on top
    if dropdown_to_draw is not None:
        label_surf, label_pos = setup_input_box_labels[dropdown_label_index]
        screen.blit(label_surf, label_pos)
        dropdown_to_draw.draw()

def draw_Comparison_Setup_Menu():
    # Display title
    title_surf = TITLE_FONT.render("Comparison Maze Setup", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.16), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Configure maze for algorithm comparison", True, pygame.Color("gray"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.08)))
    
    # Draw input boxes and labels
    for i, input_box in enumerate(comparison_input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = comparison_input_box_labels[i]
        screen.blit(label_surf, label_pos)
    
    # Draw save button
    save_comparison_button.draw()

# Draw parallel visualization of all 6 algorithms in grid or expanded view
def draw_Comparison_Visualisation():
    global comparison_expanded_view
    
    if maze is None:
        error_surf = BUTTON_FONT.render("Error: No maze loaded", True, pygame.Color("red"))
        screen.blit(error_surf, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
        return
    
    # Calculate grid layout (2x3 grid)
    grid_cols = 3
    grid_rows = 2
    cell_width = SCREEN_WIDTH // grid_cols
    cell_height = SCREEN_HEIGHT // grid_rows
    
    if comparison_expanded_view is None:
        # Grid view - show all 6 algorithms
        for idx, algo in enumerate(comparison_algorithms):
            algo_name = algo["name"]
            row = idx // grid_cols
            col = idx % grid_cols
            x_offset = col * cell_width
            y_offset = row * cell_height
            
            # Draw algorithm section
            draw_algorithm_in_section(algo, algo_name, x_offset, y_offset, cell_width, cell_height, idx)
    else:
        # Expanded view - show one algorithm full screen
        algo = comparison_algorithms[comparison_expanded_view]
        algo_name = algo["name"]
        draw_algorithm_in_section(algo, algo_name, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, comparison_expanded_view, expanded=True)
        
        # Draw "ESC to return" hint
        hint_surf = INPUT_BOX_FONT.render("Press ESC to return to grid view", True, pygame.Color("white"))
        screen.blit(hint_surf, (10, SCREEN_HEIGHT - 30))
    
    # Draw results overlay if active (highest priority)
    if comparison_show_results:
        draw_comparison_results_overlay()
    # Draw settings overlay if active
    elif comparison_show_settings:
        draw_comparison_settings_overlay()
    # Draw status indicators (only if not showing overlays)
    elif not comparison_show_settings and not comparison_show_results:
        draw_comparison_status()

# Draw a single algorithm visualization in its section
def draw_algorithm_in_section(algo, algo_name, x, y, width, height, idx, expanded=False):
    # Draw border
    pygame.draw.rect(screen, pygame.Color("white"), (x, y, width, height), 2)
    
    # Draw title background
    title_height = 30 if not expanded else 40
    pygame.draw.rect(screen, pygame.Color(40, 40, 60), (x, y, width, title_height))
    
    # Draw algorithm name
    font = BUTTON_FONT if not expanded else TITLE_FONT
    title_surf = font.render(algo_name, True, pygame.Color("white"))
    screen.blit(title_surf, (x + 10, y + 5))
    
    if not comparison_visualization_paused:
        # Calculate maze drawing parameters
        maze_area_height = height - title_height - 60  # Leave space for stats
        maze_area_width = width - 20
        
        # Scale maze to fit
        cells_width = maze.maze_size_width
        cells_height = maze.maze_size_height
        cell_size = min(maze_area_width // cells_width, maze_area_height // cells_height)
        
        maze_x = x + (width - cells_width * cell_size) // 2
        maze_y = y + title_height + 5
        
        # Draw maze (scaled)
        draw_scaled_maze(maze_x, maze_y, cell_size, algo, algo_name, expanded)
    else:
        # Show "VISUALIZATION PAUSED" message
        pause_surf = HYPERPARAMETERS_FONT.render("VISUALIZATION PAUSED", True, pygame.Color("yellow"))
        screen.blit(pause_surf, (x + width // 2 - 100, y + height // 2))
    
    # Draw stats at bottom
    draw_algorithm_stats(algo, algo_name, x, y + height - 55, width, expanded)

# Draw maze scaled to fit in section
def draw_scaled_maze(x, y, cell_size, algo, algo_name, expanded):
    # Draw maze cells
    for row in range(maze.maze_size_height):
        for col in range(maze.maze_size_width):
            cell_x = x + col * cell_size
            cell_y = y + row * cell_size
            
            # Draw cell background
            state = maze.gridStates[row][col]
            if (col, row) == maze.origin_cor:
                color = pygame.Color(255, 215, 0)  # Goal = gold/yellow
            elif (col, row) == maze.start_pos:
                color = pygame.Color(50, 200, 255)  # Start = bright cyan
            else:
                color = pygame.Color(200, 200, 200)  # Normal = light gray
            
            pygame.draw.rect(screen, color, (cell_x, cell_y, cell_size, cell_size))
            
            # Draw walls (if action is NOT in allowed actions list, there's a wall)
            # Actions: 0=left, 1=right, 2=up, 3=down
            if 2 not in state.actions:  # No up action = wall on top
                pygame.draw.line(screen, pygame.Color("black"), (cell_x, cell_y), (cell_x + cell_size, cell_y), 2)
            if 1 not in state.actions:  # No right action = wall on right
                pygame.draw.line(screen, pygame.Color("black"), (cell_x + cell_size, cell_y), (cell_x + cell_size, cell_y + cell_size), 2)
            if 3 not in state.actions:  # No down action = wall on bottom
                pygame.draw.line(screen, pygame.Color("black"), (cell_x, cell_y + cell_size), (cell_x + cell_size, cell_y + cell_size), 2)
            if 0 not in state.actions:  # No left action = wall on left
                pygame.draw.line(screen, pygame.Color("black"), (cell_x, cell_y), (cell_x, cell_y + cell_size), 2)
    
    # Draw optimal path arrows if enabled
    if comparison_show_optimal_path and maze.optimal_path:
        draw_scaled_optimal_path(x, y, cell_size, maze.optimal_path, maze.start_pos)
    
    # Draw agent position
    if algo["type"] == "RL" and algo_name in comparison_agents:
        agent_pos = comparison_agents[algo_name].activeState
        agent_x = x + agent_pos[0] * cell_size + cell_size // 2
        agent_y = y + agent_pos[1] * cell_size + cell_size // 2
        agent_radius = max(3, cell_size // 3)
        pygame.draw.circle(screen, pygame.Color("red"), (agent_x, agent_y), agent_radius)
    elif algo["type"] == "NonRL" and algo_name in comparison_visualizers:
        viz = comparison_visualizers[algo_name]
        current_pos = viz.get_current_position()
        if current_pos:
            agent_x = x + current_pos[0] * cell_size + cell_size // 2
            agent_y = y + current_pos[1] * cell_size + cell_size // 2
            agent_radius = max(3, cell_size // 3)
            pygame.draw.circle(screen, pygame.Color("red"), (agent_x, agent_y), agent_radius)
    
    # Draw labels for start and goal ON TOP of everything as circles (same size as agent)
    agent_radius = max(3, cell_size // 3)
    
    # Draw GOAL label/marker
    goal_x = x + maze.origin_cor[0] * cell_size + cell_size // 2
    goal_y = y + maze.origin_cor[1] * cell_size + cell_size // 2
    # Draw gold circle for goal
    pygame.draw.circle(screen, pygame.Color(255, 215, 0), (goal_x, goal_y), agent_radius)
    # Draw black border
    pygame.draw.circle(screen, pygame.Color("black"), (goal_x, goal_y), agent_radius, 2)
    
    # Draw START label/marker
    start_x = x + maze.start_pos[0] * cell_size + cell_size // 2
    start_y = y + maze.start_pos[1] * cell_size + cell_size // 2
    # Draw cyan circle for start
    pygame.draw.circle(screen, pygame.Color(50, 200, 255), (start_x, start_y), agent_radius)
    # Draw black border
    pygame.draw.circle(screen, pygame.Color("black"), (start_x, start_y), agent_radius, 2)

# Draw optimal path arrows on scaled maze
def draw_scaled_optimal_path(x, y, cell_size, optimal_path, start_pos):
    current_x = start_pos[0]
    current_y = start_pos[1]
    
    arrow_padding = cell_size * 0.12
    arrow_color = pygame.Color(0, 128, 255)  # Blue arrows
    
    # Create arrow polygon from start to end point
    def arrow_polygon(start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return []
        
        # Normalize
        dx /= length
        dy /= length
        
        # Arrow dimensions
        arrow_width = cell_size * 0.15
        arrow_head_length = cell_size * 0.25
        arrow_head_width = cell_size * 0.30
        
        # Perpendicular vector
        px = -dy
        py = dx
        
        # Shaft
        shaft_end = (end[0] - dx * arrow_head_length, end[1] - dy * arrow_head_length)
        
        # Arrow points
        points = [
            (start[0] + px * arrow_width/2, start[1] + py * arrow_width/2),
            (shaft_end[0] + px * arrow_width/2, shaft_end[1] + py * arrow_width/2),
            (shaft_end[0] + px * arrow_head_width/2, shaft_end[1] + py * arrow_head_width/2),
            end,
            (shaft_end[0] - px * arrow_head_width/2, shaft_end[1] - py * arrow_head_width/2),
            (shaft_end[0] - px * arrow_width/2, shaft_end[1] - py * arrow_width/2),
            (start[0] - px * arrow_width/2, start[1] - py * arrow_width/2),
        ]
        return points
    
    for direction in optimal_path:
        cell_x = x + current_x * cell_size
        cell_y = y + current_y * cell_size
        
        if direction == "Up":
            arrow = arrow_polygon(
                (cell_x + cell_size/2, cell_y + cell_size - arrow_padding),
                (cell_x + cell_size/2, cell_y + arrow_padding)
            )
            current_y -= 1
        elif direction == "Down":
            arrow = arrow_polygon(
                (cell_x + cell_size/2, cell_y + arrow_padding),
                (cell_x + cell_size/2, cell_y + cell_size - arrow_padding)
            )
            current_y += 1
        elif direction == "Left":
            arrow = arrow_polygon(
                (cell_x + cell_size - arrow_padding, cell_y + cell_size/2),
                (cell_x + arrow_padding, cell_y + cell_size/2)
            )
            current_x -= 1
        elif direction == "Right":
            arrow = arrow_polygon(
                (cell_x + arrow_padding, cell_y + cell_size/2),
                (cell_x + cell_size - arrow_padding, cell_y + cell_size/2)
            )
            current_x += 1
        else:
            continue
        
        if arrow:
            pygame.draw.polygon(screen, arrow_color, arrow)

# Draw statistics for an algorithm
def draw_algorithm_stats(algo, algo_name, x, y, width, expanded):
    font = HYPERPARAMETERS_FONT if not expanded else INPUT_BOX_FONT
    
    if algo["type"] == "RL" and algo_name in comparison_agents:
        # RL stats - show current position
        agent = comparison_agents[algo_name]
        pos_x, pos_y = agent.activeState
        stats_text = f"Training... Pos: ({pos_x}, {pos_y})"
    elif algo["type"] == "NonRL" and algo_name in comparison_visualizers:
        viz = comparison_visualizers[algo_name]
        if viz.is_finished:
            stats_text = f"Finished - Solution: {len(viz.algorithm_solver.path)} steps"
        else:
            stats_text = f"Exploring... {len(viz.path_history)} moves"
    else:
        stats_text = "Initializing..."
    
    stats_surf = font.render(stats_text, True, pygame.Color("white"))
    screen.blit(stats_surf, (x + 10, y))

# Draw runtime settings overlay (press C)
def draw_comparison_settings_overlay():
    # Semi-transparent background
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Settings box
    box_width = 500
    box_height = 400
    box_x = (SCREEN_WIDTH - box_width) // 2
    box_y = (SCREEN_HEIGHT - box_height) // 2
    
    pygame.draw.rect(screen, pygame.Color(40, 40, 60), (box_x, box_y, box_width, box_height))
    pygame.draw.rect(screen, pygame.Color("white"), (box_x, box_y, box_width, box_height), 3)
    
    # Title
    title_surf = BUTTON_FONT.render("Comparison Settings", True, pygame.Color("white"))
    screen.blit(title_surf, (box_x + 20, box_y + 20))
    
    # Settings text
    y_offset = box_y + 70
    settings_lines = [
        f"Speed Multiplier: {comparison_speed_multiplier}x (1-5 / 6-0)",
        "",
        "Controls:",
        "  F - Freeze/Unfreeze Visualization",
        "  T - Pause/Resume Training",
        "  C - Toggle Settings (this menu)",
        "  P - Toggle Optimal Path Display",
        "  R - Toggle Results Overlay (when finished)",
        "  ESC - Exit to grid / Exit comparison",
        "  Click - Expand algorithm to full screen",
        "",
        "Status:",
        f"  Visualization: {'PAUSED' if comparison_visualization_paused else 'RUNNING'}",
        f"  Training: {'PAUSED' if comparison_training_paused else 'RUNNING'}",
        f"  Optimal Path: {'ON' if comparison_show_optimal_path else 'OFF'}"
    ]
    
    for line in settings_lines:
        line_surf = HYPERPARAMETERS_FONT.render(line, True, pygame.Color("white"))
        screen.blit(line_surf, (box_x + 30, y_offset))
        y_offset += 30
    
    # Close hint
    close_surf = INPUT_BOX_FONT.render("Press C to close", True, pygame.Color("yellow"))
    screen.blit(close_surf, (box_x + box_width // 2 - 70, box_y + box_height - 40))

# Draw results overlay showing performance statistics
def draw_comparison_results_overlay():
    # Semi-transparent background
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Results box
    box_width = 600
    box_height = 500
    box_x = (SCREEN_WIDTH - box_width) // 2
    box_y = (SCREEN_HEIGHT - box_height) // 2
    
    pygame.draw.rect(screen, pygame.Color(40, 40, 60), (box_x, box_y, box_width, box_height))
    pygame.draw.rect(screen, pygame.Color("white"), (box_x, box_y, box_width, box_height), 3)
    
    # Title
    title_surf = BUTTON_FONT.render("Comparison Results", True, pygame.Color("gold"))
    screen.blit(title_surf, (box_x + 20, box_y + 20))
    
    # Separate NonRL and RL results
    nonrl_results = [r for r in comparison_results if r["type"] == "NonRL"]
    rl_results = [r for r in comparison_results if r["type"] == "RL"]
    
    y_offset = box_y + 70
    
    # Non-RL Results
    if nonrl_results:
        nonrl_title = BUTTON_FONT.render("Non-RL Algorithms:", True, pygame.Color("cyan"))
        screen.blit(nonrl_title, (box_x + 30, y_offset))
        y_offset += 35
        
        # Find best performers (handle both old and new data structures)
        successful_nonrl = []
        for r in nonrl_results:
            # Check if using new comprehensive structure or old simple structure
            if "execution_metrics" in r:
                if r["execution_metrics"]["success"]:
                    successful_nonrl.append(r)
            elif "success" in r:
                if r["success"]:
                    successful_nonrl.append(r)
        
        if successful_nonrl:
            # Get min steps and time (handle both structures)
            if "execution_metrics" in successful_nonrl[0]:
                min_steps = min(r["path_analysis"]["path_length"] for r in successful_nonrl)
                min_time = min(r["execution_metrics"]["execution_time_seconds"] for r in successful_nonrl)
            else:
                min_steps = min(r["steps"] for r in successful_nonrl)
                min_time = min(r["finished_time"] for r in successful_nonrl)
            
            for result in nonrl_results:
                # Handle both data structures
                if "execution_metrics" in result:
                    algo_name = result["algorithm"]
                    steps = result["path_analysis"]["path_length"]
                    time_taken = result["execution_metrics"]["execution_time_seconds"]
                    success = result["execution_metrics"]["success"]
                else:
                    algo_name = result.get("name", result.get("algorithm", "Unknown"))
                    steps = result["steps"]
                    time_taken = result["finished_time"]
                    success = result["success"]
                
                line = f"{algo_name}: {steps} steps, {time_taken:.2f}s"
                color = pygame.Color("green") if success else pygame.Color("red")
                
                # Highlight best performers
                if success and steps == min_steps:
                    line += " (FEWEST STEPS)"
                    color = pygame.Color("gold")
                elif success and time_taken == min_time and steps != min_steps:
                    line += " (FASTEST)"
                
                result_surf = HYPERPARAMETERS_FONT.render(line, True, color)
                screen.blit(result_surf, (box_x + 40, y_offset))
                y_offset += 28
        else:
            fail_surf = HYPERPARAMETERS_FONT.render("No algorithms reached the goal", True, pygame.Color("red"))
            screen.blit(fail_surf, (box_x + 40, y_offset))
            y_offset += 28
    
    y_offset += 15
    
    # RL Results
    if rl_results:
        rl_title = BUTTON_FONT.render("RL Algorithms:", True, pygame.Color("yellow"))
        screen.blit(rl_title, (box_x + 30, y_offset))
        y_offset += 35
        
        # Find best performers (handle both structures)
        if rl_results:
            if "performance_metrics" in rl_results[0]:
                # New comprehensive structure
                best_success_rate = max(r["performance_metrics"]["success_rate_percent"] for r in rl_results)
                min_train_time = min(r["performance_metrics"]["training_time_seconds"] for r in rl_results)
            else:
                # Old simple structure
                min_train_time = min(r["finished_time"] for r in rl_results)
            
            for result in rl_results:
                # Handle both data structures
                if "performance_metrics" in result:
                    algo_name = result["algorithm"]
                    train_time = result["performance_metrics"]["training_time_seconds"]
                    success_rate = result["performance_metrics"]["success_rate_percent"]
                    line = f"{algo_name}: {success_rate:.1f}% success, {train_time:.1f}s"
                    color = pygame.Color("green") if success_rate > 50 else pygame.Color("orange")
                    
                    if success_rate == best_success_rate:
                        line += " (BEST)"
                        color = pygame.Color("gold")
                    elif train_time == min_train_time:
                        line += " (FASTEST)"
                else:
                    algo_name = result.get("name", result.get("algorithm", "Unknown"))
                    train_time = result["finished_time"]
                    success = result.get("success", False)
                    line = f"{algo_name}: Completed in {train_time:.2f}s"
                    color = pygame.Color("green") if success else pygame.Color("orange")
                    
                    if train_time == min_train_time:
                        line += " (FIRST DONE)"
                        color = pygame.Color("gold")
                
                result_surf = HYPERPARAMETERS_FONT.render(line, True, color)
                screen.blit(result_surf, (box_x + 40, y_offset))
                y_offset += 28
    
    # Close hint
    y_offset = box_y + box_height - 50
    close_surf = INPUT_BOX_FONT.render("Press R to close | Press Q to quit to Main Menu", True, pygame.Color("yellow"))
    screen.blit(close_surf, (box_x + box_width // 2 - 200, y_offset))

# Draw status indicators at bottom of screen
def draw_comparison_status():
    status_y = SCREEN_HEIGHT - 25
    
    # Multi-run progress indicator
    if multi_run_active:
        run_surf = HYPERPARAMETERS_FONT.render(
            f"Run {multi_run_current + 1}/{multi_run_total}", True, pygame.Color("orange"))
        screen.blit(run_surf, (10, status_y - 18))
    
    # Visualization status
    if comparison_visualization_paused:
        viz_surf = HYPERPARAMETERS_FONT.render("VIZ: PAUSED", True, pygame.Color("yellow"))
    else:
        viz_surf = HYPERPARAMETERS_FONT.render("VIZ: RUNNING", True, pygame.Color("green"))
    screen.blit(viz_surf, (10, status_y))
    
    # Training status  
    if comparison_training_paused:
        train_surf = HYPERPARAMETERS_FONT.render("TRAIN: PAUSED", True, pygame.Color("yellow"))
    else:
        train_surf = HYPERPARAMETERS_FONT.render("TRAIN: RUNNING", True, pygame.Color("green"))
    screen.blit(train_surf, (200, status_y))
    
    # Speed indicator
    speed_surf = HYPERPARAMETERS_FONT.render(f"Speed: {comparison_speed_multiplier}x", True, pygame.Color("cyan"))
    screen.blit(speed_surf, (400, status_y))
    
    # Settings hint
    hint_surf = HYPERPARAMETERS_FONT.render("Press C for controls", True, pygame.Color("gray"))
    screen.blit(hint_surf, (SCREEN_WIDTH - 200, status_y))

def draw_Load_Menu():
    title_surf = TITLE_FONT.render("Load Maze", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.1), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Select a maze layout to load:", True, pygame.Color("white"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.13)))
    
    # Draw Load & Edit button
    load_edit_button.draw()
    
    # Draw Load button
    load_continue_button.draw()
    
    # Draw back button
    back_to_choose_button.draw()
    
    # Draw dropdown menu last so it appears on top
    maze_dropdown.draw()

def draw_Choose_Menu():
    # Display title
    title_surf = TITLE_FONT.render("Choose Option", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.1), int(SCREEN_HEIGHT * 0.12)))
    
    # Draw buttons
    setup_menu_button.draw()
    load_menu_button.draw()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          MAIN PYGAME LOOP                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Add ESC key to quit
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        # Handle window resize
        if event.type == pygame.VIDEORESIZE:
            SCREEN_WIDTH = event.w
            SCREEN_HEIGHT = event.h
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
            # Reinitialize all buttons with new screen size
            initialize_all_buttons()
            print(f"Window resized to: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        
        if activeMonitor == "Settings":
            for input_box in input_boxes:
                if input_box.handle_event(event):
                    apply_input_box_values()
        
        if activeMonitor == "Setup_Menu":
            for input_box in setup_input_boxes:
                # Check if it's a dropdown or input box
                if isinstance(input_box, Algorithm_Dropdown):
                    input_box.handle_event(event)
                else:
                    input_box.handle_event(event)
        
        if activeMonitor == "Comparison_Setup_Menu":
            for input_box in comparison_input_boxes:
                input_box.handle_event(event)
        
        if activeMonitor == "RL_Settings":
            for input_box in rl_settings_input_boxes:
                input_box.handle_event(event)
        
        if activeMonitor == "Comparison_RL_Settings":
            # Skip first input box (EPISODES) in comparison mode - it's set in comparison settings
            for i, input_box in enumerate(rl_settings_input_boxes):
                if i == 0:  # Skip EPISODES
                    continue
                input_box.handle_event(event)
            # Handle multi-run count input
            multi_run_input.handle_event(event)
        
        if activeMonitor == "Load_Menu":
            maze_dropdown.handle_event(event)
        
        # Handle non-RL algorithm menu dropdown
        if activeMonitor == "NonRL_Algorithm_Menu":
            nonrl_algorithm_dropdown.handle_event(event)
        
        # Handle non-RL speed settings input
        if activeMonitor == "NonRL_Speed_Settings":
            if nonrl_speed_input.handle_event(event):
                try:
                    nonrl_steps_per_second = int(nonrl_speed_input.text)
                    print(f"Speed set to: {nonrl_steps_per_second} steps/second")
                except ValueError:
                    print("Invalid speed value")
        
        # Handle non-RL visualization controls
        if event.type == pygame.KEYDOWN and activeMonitor == "NonRL_Visualisation":
            if event.key == pygame.K_i:  # Toggle optimal path
                show_path = not show_path
            if event.key == pygame.K_SPACE:  # Pause/Resume
                if nonrl_visualizer:
                    if nonrl_visualizer.is_running:
                        nonrl_visualizer.stop()
                        print("Paused")
                    else:
                        nonrl_visualizer.start()
                        print("Resumed")
        
        if event.type == pygame.KEYDOWN and activeMonitor == "Comparison_Visualisation":
            if event.key == pygame.K_r:  # Toggle results overlay
                if len(comparison_results) > 0:
                    comparison_show_results = not comparison_show_results
            if event.key == pygame.K_q and comparison_show_results:  # Quit to main menu from results
                # Save if not already saved (single-run case)
                if not multi_run_active and not comparison_json_saved:
                    save_comparison_json()
                    comparison_json_saved = True
                activeMonitor = "Main_Menu"
                comparison_running = False
                comparison_show_results = False
                multi_run_active = False
                multi_run_current = 0
                multi_run_all_results = []
                print("Exited comparison mode")
            if event.key == pygame.K_c:  # Toggle settings overlay
                if not comparison_show_results:
                    comparison_show_settings = not comparison_show_settings
            if event.key == pygame.K_p:  # Toggle optimal path display
                comparison_show_optimal_path = not comparison_show_optimal_path
                status = "ON" if comparison_show_optimal_path else "OFF"
                print(f"Optimal path display: {status}")
            if event.key == pygame.K_f:  # Freeze/unfreeze visualization
                comparison_visualization_paused = not comparison_visualization_paused
                status = "PAUSED" if comparison_visualization_paused else "RUNNING"
                print(f"Visualization {status}")
            if event.key == pygame.K_t:  # Pause/resume training
                comparison_training_paused = not comparison_training_paused
                status = "PAUSED" if comparison_training_paused else "RUNNING"
                print(f"Training {status}")
            if event.key == pygame.K_b:  # Back to main menu
                if comparison_expanded_view is not None:
                    # Return to grid view
                    comparison_expanded_view = None
            # Speed control keys
            if event.key == pygame.K_1:
                comparison_speed_multiplier = 1
                print("Speed: 1x")
            elif event.key == pygame.K_2:
                comparison_speed_multiplier = 2
                print("Speed: 2x")
            elif event.key == pygame.K_3:
                comparison_speed_multiplier = 5
                print("Speed: 5x")
            elif event.key == pygame.K_4:
                comparison_speed_multiplier = 10
                print("Speed: 10x")
            elif event.key == pygame.K_5:
                comparison_speed_multiplier = 20
                print("Speed: 20x")
            elif event.key == pygame.K_6:
                comparison_speed_multiplier = 50
                print("Speed: 50x")
            elif event.key == pygame.K_7:
                comparison_speed_multiplier = 100
                print("Speed: 100x")
            elif event.key == pygame.K_8:
                comparison_speed_multiplier = 200
                print("Speed: 200x")
            elif event.key == pygame.K_9:
                comparison_speed_multiplier = 500
                print("Speed: 500x")
            elif event.key == pygame.K_0:
                comparison_speed_multiplier = 1000
                print("Speed: 1000x")
        
        # Click to expand algorithm in grid view
        if event.type == pygame.MOUSEBUTTONDOWN and activeMonitor == "Comparison_Visualisation":
            if comparison_expanded_view is None and not comparison_show_settings:
                # Grid view - detect which cell was clicked
                mouse_x, mouse_y = event.pos
                cell_width = SCREEN_WIDTH // 3
                cell_height = SCREEN_HEIGHT // 2
                
                col = mouse_x // cell_width
                row = mouse_y // cell_height
                clicked_idx = row * 3 + col
                
                if 0 <= clicked_idx < len(comparison_algorithms):
                    comparison_expanded_view = clicked_idx
                    print(f"Expanded: {comparison_algorithms[clicked_idx]['name']}")
        
        if event.type == pygame.KEYDOWN and activeMonitor == "RL - Visualisation":
            if event.key == pygame.K_q: # Show Q-Values on Screen
                show_q_values = not show_q_values
            if event.key == pygame.K_i: # Show Optimal Path on Screen
                show_path = not show_path
            if event.key == pygame.K_v: # Show Config Values on Screen
                show_config = not show_config
            if event.key == pygame.K_s:  # start/stop training
                if not training_active:
                    # Choose algorithm based on selection
                    if selected_algorithm == "SARSA":
                        trainer = sarsa_coroutine(
                            agent, maze, Q,
                            EPISODES, max_steps, gamma,
                            EPS0, EPS_MIN, EPS_DECAY,
                            ALPHA0, ALPHA_MIN, ALPHA_DECAY
                        )
                    else:  # Default to Q-Learning
                        trainer = q_learning_coroutine(
                            agent, maze, Q,
                            EPISODES, max_steps, gamma,
                            EPS0, EPS_MIN, EPS_DECAY,
                            ALPHA0, ALPHA_MIN, ALPHA_DECAY
                        )
                    training_active = True
                    print(f"Training started with {selected_algorithm}")
                else:
                    training_active = False
                    trainer = None
                    print("Training stopped manually")

    screen.fill((0, 0, 0))

    # draw active monitors
    if activeMonitor == "Mode_Selection":
        draw_Mode_Selection()
    elif activeMonitor == "Algorithm_Type_Menu":
        draw_Algorithm_Type_Menu()
    elif activeMonitor == "NonRL_Algorithm_Menu":
        draw_NonRL_Algorithm_Menu()
    elif activeMonitor == "NonRL_Speed_Settings":
        draw_NonRL_Speed_Settings()
    elif activeMonitor == "NonRL_Visualisation":
        draw_NonRL_Visualisation()
    elif activeMonitor == "Choose_Menu":
        draw_Choose_Menu()
    elif activeMonitor == "Setup_Menu":
        draw_Setup_Menu()
    elif activeMonitor == "RL_Settings":
        draw_RL_Settings()
    elif activeMonitor == "Comparison_Setup_Menu":
        draw_Comparison_Setup_Menu()
    elif activeMonitor == "Comparison_RL_Settings":
        draw_Comparison_RL_Settings()
    elif activeMonitor == "Comparison_Visualisation":
        draw_Comparison_Visualisation()
    elif activeMonitor == "Load_Menu":
        draw_Load_Menu()
    elif activeMonitor == "Main_Menu":
        draw_Main_Menu()
    elif activeMonitor == "Settings":
        draw_Settings_Menu()
    # draw maze and agent
    elif activeMonitor == "RL - Visualisation":
        draw_RL_Visualisation()
        

    # Advance comparison training and visualization
    if activeMonitor == "Comparison_Visualisation" and not comparison_training_paused:
        # Advance RL trainers
        finished_trainers = []
        for algo_name, trainer in list(comparison_trainers.items()):
            if trainer is not None:
                try:
                    # Step multiple times based on speed multiplier
                    steps = max(1, int(steps_per_frame * comparison_speed_multiplier))
                    for _ in range(steps):
                        next(trainer)
                except StopIteration:
                    # Training finished for this algorithm
                    finish_time = time.time() - comparison_start_time
                    agent = comparison_agents[algo_name]
                    Q_table = comparison_q_tables[algo_name]
                    heat_table = comparison_heat_tables[algo_name]
                    
                    # Calculate heatmap
                    heatmap_list = [[heat_table[y][x] for x in range(len(heat_table[0]))] for y in range(len(heat_table))]
                    
                    # Get tracking data for this algorithm
                    find_episodes = comparison_find_episodes.get(algo_name, [])
                    final_path = comparison_final_paths.get(algo_name, [])
                    num_returns = comparison_num_returns.get(algo_name, 0)
                    first_find, first_episode = comparison_first_finds.get(algo_name, (False, None))
                    
                    # Calculate success rate
                    success_rate = (num_returns / EPISODES * 100) if EPISODES > 0 else 0
                    find_percentage = (len(find_episodes) / EPISODES * 100) if EPISODES > 0 else 0
                    
                    # Check if agent reached goal
                    final_success = agent.activeState == list(maze.origin_cor)
                    
                    comparison_results.append({
                        "algorithm": algo_name,
                        "type": "RL",
                        "hyperparameters": {
                            "gamma": gamma,
                            "epsilon_start": EPS0,
                            "epsilon_min": EPS_MIN,
                            "epsilon_decay": EPS_DECAY,
                            "alpha_start": ALPHA0,
                            "alpha_min": ALPHA_MIN,
                            "alpha_decay": ALPHA_DECAY
                        },
                        "training_config": {
                            "episodes": EPISODES,
                            "max_steps_per_episode": max_steps,
                            "reward_for_finish": rewardForFinish,
                            "reward_for_valid_move": rewardForValidMove
                        },
                        "performance_metrics": {
                            "first_find_episode": first_episode if first_find else -1,
                            "total_successful_episodes": num_returns,
                            "success_rate_percent": round(success_rate, 2),
                            "find_percentage": round(find_percentage, 2),
                            "episodes_that_found_goal": find_episodes[:],
                            "total_episodes_trained": EPISODES,
                            "training_time_seconds": round(finish_time, 2)
                        },
                        "path_analysis": {
                            "final_path": final_path[:],
                            "final_path_length": len(final_path) if final_path else 0,
                            "optimal_path": maze.optimal_path[:],
                            "optimal_path_length": len(maze.optimal_path),
                            "path_efficiency": round(len(maze.optimal_path) / len(final_path), 2) if final_path and len(final_path) > 0 else 0,
                            "extra_steps": len(final_path) - len(maze.optimal_path) if final_path and len(maze.optimal_path) > 0 else -1
                        },
                        "exploration_data": {
                            "heatmap": heatmap_list,
                            "most_visited_states": sorted(
                                [((x, y), heat_table[y][x]) for y in range(len(heat_table)) for x in range(len(heat_table[0]))],
                                key=lambda item: item[1],
                                reverse=True
                            )[:10],  # Top 10 most visited states
                            "total_state_visits": sum(sum(row) for row in heat_table),
                            "unique_states_visited": sum(1 for row in heat_table for val in row if val > 0)
                        },
                        "q_table": Q_table.tolist(),
                        "maze_info": {
                            "name": maze.name,
                            "size": [maze.maze_size_width, maze.maze_size_height],
                            "start_position": maze.start_pos,
                            "goal_position": maze.origin_cor
                        }
                    })
                    print(f"{algo_name} training completed in {finish_time:.2f}s")
                    finished_trainers.append(algo_name)
        
        # Remove finished trainers
        for algo_name in finished_trainers:
            del comparison_trainers[algo_name]
        
        # Advance Non-RL visualizers
        if not comparison_visualization_paused:
            for algo_name, viz in list(comparison_visualizers.items()):
                if viz is not None and not viz.is_finished:
                    # Step multiple times based on speed multiplier
                    steps = max(1, int(comparison_speed_multiplier))
                    for _ in range(steps):
                        result = viz.step()
                        if result is None and viz.is_finished:
                            finish_time = time.time() - comparison_start_time
                            # Use the algorithm's success flag and solution path length
                            success = viz.algorithm_solver.success
                            steps_taken = len(viz.algorithm_solver.path)
                            nodes_explored = viz.algorithm_solver.nodes_explored
                            execution_time = viz.algorithm_solver.execution_time
                            path_taken = viz.path_history[:]
                            
                            # Save comprehensive Non-RL results
                            comparison_results.append({
                                "algorithm": algo_name,
                                "type": "NonRL",
                                "execution_metrics": {
                                    "execution_time_seconds": round(execution_time, 6),
                                    "steps_taken": steps_taken,
                                    "nodes_explored": nodes_explored,
                                    "success": success,
                                    "finished": True,
                                    "total_comparison_time": round(finish_time, 2)
                                },
                                "path_analysis": {
                                    "path_taken": path_taken,
                                    "path_length": steps_taken,
                                    "optimal_path": maze.optimal_path[:],
                                    "optimal_path_length": len(maze.optimal_path),
                                    "efficiency_ratio": round(steps_taken / len(maze.optimal_path), 2) if len(maze.optimal_path) > 0 else -1,
                                    "extra_steps": steps_taken - len(maze.optimal_path) if len(maze.optimal_path) > 0 else -1
                                },
                                "exploration_data": {
                                    "unique_cells_visited": len(set(tuple(pos) for pos in path_taken)),
                                    "total_cells_in_maze": maze.maze_size_width * maze.maze_size_height,
                                    "exploration_percentage": round(len(set(tuple(pos) for pos in path_taken)) / (maze.maze_size_width * maze.maze_size_height) * 100, 2),
                                    "backtracking_steps": max(0, steps_taken - len(set(tuple(pos) for pos in path_taken)))
                                },
                                "algorithm_characteristics": {
                                    "guarantees_shortest_path": algo_name in ["BFS"],
                                    "is_deterministic": algo_name in ["BFS", "Wall Follower", "Greedy"],
                                    "requires_memory": algo_name in ["BFS", "Greedy"],
                                    "uses_heuristic": algo_name in ["Greedy"]
                                },
                                "maze_info": {
                                    "name": maze.name,
                                    "size": [maze.maze_size_width, maze.maze_size_height],
                                    "start_position": maze.start_pos,
                                    "goal_position": maze.origin_cor
                                }
                            })
                            print(f"{algo_name} completed: {steps_taken} steps in {execution_time:.4f}s, success={success}")
                            break
        
        # Check if all algorithms finished
        if len(comparison_trainers) == 0:
            all_nonrl_finished = all(
                viz.is_finished for viz in comparison_visualizers.values()
            ) if comparison_visualizers else True
            
            if all_nonrl_finished and not comparison_show_results:
                # Build this run's data and collect it
                if not comparison_json_saved:
                    run_data = _build_single_run_data()
                    multi_run_all_results.append(run_data)
                    comparison_json_saved = True
                    
                    if multi_run_active and multi_run_current < multi_run_total - 1:
                        # More runs remain – start the next one
                        multi_run_current += 1
                        print(f"\n--- Run {multi_run_current + 1}/{multi_run_total} ---")
                        
                        # Regenerate a fresh random maze from the same config
                        create_maze_from_json(new_json_file_name)
                        
                        # Re-create agent, Q-table, HeatTable for the new maze
                        agent = Agent(-5, maze.gridStates, maze.start_pos)
                        Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
                        HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
                        
                        # Reset comparison state flags so the next run works
                        comparison_show_results = False
                        comparison_json_saved = False
                        comparison_training_paused = False
                        
                        # Start a fresh comparison on the new maze
                        start_automated_comparison()
                    else:
                        # Last run (or single run) – save and show results
                        print("All algorithms completed - Press R to view results")
                        comparison_show_results = True
                        comparison_training_paused = True
                        if multi_run_active:
                            save_multi_run_json()
                        else:
                            save_comparison_json()

    # advance training coroutine a bit each frame
    if training_active and trainer is not None:
        try:
            for _ in range(steps_per_frame):
                next(trainer)
        except StopIteration:
            training_active = False
            trainer = None
            print("Training finished")
            
            # If in comparison mode, save results and continue to next algorithm
            if comparison_running:
                # Calculate heatmap
                heatmap_list = [[HeatTable[y][x] for x in range(len(HeatTable[0]))] for y in range(len(HeatTable))]
                
                # Calculate success rate
                success_rate = (numOfReturns / EPISODES * 100) if EPISODES > 0 else 0
                find_percentage = (len(FindOriginEpisodes) / EPISODES * 100) if EPISODES > 0 else 0
                
                # Save comprehensive RL results
                comparison_results.append({
                    "algorithm": selected_algorithm,
                    "type": "RL",
                    "hyperparameters": {
                        "gamma": gamma,
                        "epsilon_start": EPS0,
                        "epsilon_min": EPS_MIN,
                        "epsilon_decay": EPS_DECAY,
                        "alpha_start": ALPHA0,
                        "alpha_min": ALPHA_MIN,
                        "alpha_decay": ALPHA_DECAY
                    },
                    "training_config": {
                        "episodes": EPISODES,
                        "max_steps_per_episode": max_steps,
                        "reward_for_finish": rewardForFinish,
                        "reward_for_valid_move": rewardForValidMove
                    },
                    "performance_metrics": {
                        "first_find_episode": firstEpisode if firstFind else -1,
                        "total_successful_episodes": numOfReturns,
                        "success_rate_percent": round(success_rate, 2),
                        "find_percentage": round(find_percentage, 2),
                        "episodes_that_found_goal": FindOriginEpisodes[:],
                        "total_episodes_trained": EPISODES
                    },
                    "path_analysis": {
                        "final_path": final_path[:],
                        "final_path_length": len(final_path),
                        "optimal_path": maze.optimal_path[:],
                        "optimal_path_length": len(maze.optimal_path),
                        "path_efficiency": round(len(maze.optimal_path) / len(final_path), 2) if len(final_path) > 0 else 0
                    },
                    "exploration_data": {
                        "heatmap": heatmap_list,
                        "most_visited_states": sorted(
                            [((x, y), HeatTable[y][x]) for y in range(len(HeatTable)) for x in range(len(HeatTable[0]))],
                            key=lambda item: item[1],
                            reverse=True
                        )[:10]  # Top 10 most visited states
                    },
                    "q_table": Q.tolist(),
                    "maze_info": {
                        "name": maze.name,
                        "size": [maze.maze_size_width, maze.maze_size_height],
                        "start_position": maze.start_pos,
                        "goal_position": maze.origin_cor
                    }
                })
                
                # Reset for next algorithm
                comparison_current_index += 1
                # Reset training variables
                agent.reset()
                Q.fill(0)
                HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
                # Reset episode tracking
                firstFind = False
                firstEpisode = None
                numOfReturns = 0
                FindOriginEpisodes = []
                final_path = []
                # Run next algorithm
                run_next_comparison_algorithm()
    
    # Advance non-RL visualization (or comparison mode)
    if nonrl_visualizer is not None:
        if nonrl_visualizer.is_running and not nonrl_visualizer.is_finished:
            # Execute steps based on speed setting
            steps_this_frame = max(1, int(nonrl_steps_per_second / FPS)) if activeMonitor == "NonRL_Visualisation" else 100
            for _ in range(steps_this_frame):
                result = nonrl_visualizer.step()
                if result is None:  # Finished
                    # Save results
                    stats = nonrl_visualizer.get_stats()
                    
                    if comparison_running:
                        # Save comprehensive Non-RL results
                        comparison_results.append({
                            "algorithm": selected_nonrl_algorithm,
                            "type": "NonRL",
                            "execution_metrics": {
                                "execution_time_seconds": round(stats['execution_time'], 6),
                                "steps_taken": stats['steps_taken'],
                                "nodes_explored": stats['nodes_explored'],
                                "success": stats['success'],
                                "finished": stats['is_finished']
                            },
                            "path_analysis": {
                                "path_taken": nonrl_visualizer.path_history[:],
                                "path_length": stats['path_length'],
                                "optimal_path": maze.optimal_path[:],
                                "optimal_path_length": len(maze.optimal_path),
                                "efficiency_ratio": round(stats['path_length'] / len(maze.optimal_path), 2) if len(maze.optimal_path) > 0 else -1,
                                "extra_steps": stats['path_length'] - len(maze.optimal_path) if len(maze.optimal_path) > 0 else -1
                            },
                            "exploration_data": {
                                "unique_cells_visited": len(set(tuple(pos) for pos in nonrl_visualizer.path_history)),
                                "total_cells_in_maze": maze.maze_size_width * maze.maze_size_height,
                                "exploration_percentage": round(len(set(tuple(pos) for pos in nonrl_visualizer.path_history)) / (maze.maze_size_width * maze.maze_size_height) * 100, 2),
                                "backtracking_steps": max(0, stats['steps_taken'] - len(set(tuple(pos) for pos in nonrl_visualizer.path_history)))
                            },
                            "algorithm_characteristics": {
                                "guarantees_shortest_path": selected_nonrl_algorithm in ["BFS"],
                                "is_deterministic": selected_nonrl_algorithm in ["BFS", "Wall Follower", "Greedy"],
                                "requires_memory": selected_nonrl_algorithm in ["BFS", "Greedy"],
                                "uses_heuristic": selected_nonrl_algorithm in ["Greedy"]
                            },
                            "maze_info": {
                                "name": maze.name,
                                "size": [maze.maze_size_width, maze.maze_size_height],
                                "start_position": maze.start_pos,
                                "goal_position": maze.origin_cor
                            }
                        })
                        # Reset visualizer
                        nonrl_visualizer = None
                        # Move to next algorithm
                        comparison_current_index += 1
                        run_next_comparison_algorithm()
                    else:
                        # Normal mode - save individual JSON
                        save_nonrl_json_data(
                            maze,
                            selected_nonrl_algorithm,
                            nonrl_visualizer.path_history,
                            stats['execution_time'],
                            stats['success'],
                            stats['nodes_explored']
                        )
                        print(f"{selected_nonrl_algorithm} finished: {'Success' if stats['success'] else 'Failed'}")
                    break

    pygame.display.flip()
    clock.tick(FPS)

# End of main loop
pygame.quit()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      SAVE DATA TO JSON (post-loop)                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_heatmap_to_list(heatmap):
    return [[heatmap[y][x] for x in range(len(heatmap[0]))] for y in range(len(heatmap))]

heatmap_list = save_heatmap_to_list(HeatTable)
save_json_data(
    maze=maze,
    first_find=firstEpisode if firstFind else -1,
    num_of_returns=numOfReturns,
    FindOriginEpisodes=FindOriginEpisodes,
    find_not_find_percentage=[(len(FindOriginEpisodes)/EPISODES if EPISODES > 0 else 1)*100, ((EPISODES - len(FindOriginEpisodes))/EPISODES if EPISODES > 0 else 1)*100],
    heatmap=heatmap_list,
    optimal_path=maze.optimal_path,
    final_path=final_path,
    optimal_path_num_of_steps=len(maze.optimal_path),
    final_path_num_of_steps=len(final_path),
    Q_table=Q,
    gamma_val=gamma,
    eps0=EPS0,
    eps_min=EPS_MIN,
    eps_decay=EPS_DECAY,
    alpha0=ALPHA0,
    alpha_min=ALPHA_MIN,
    alpha_decay=ALPHA_DECAY
)

print(f"Saved training data to JsonData/{new_json_file_name}.json")
