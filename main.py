import random
import pygame
import numpy as np
import csv
import json
import shutil
import os
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

#-----------------------------
# Pygame Setup
#-----------------------------

# Initialize Pygame
pygame.init()

# Default window size (can be modified here)
DEFAULT_WINDOW_WIDTH = 1280   # Change this to your preferred width
DEFAULT_WINDOW_HEIGHT = 720    # Change this to your preferred height

SCREEN_WIDTH = DEFAULT_WINDOW_WIDTH
SCREEN_HEIGHT = DEFAULT_WINDOW_HEIGHT

# Create resizable window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

# Create a clock object to manage the frame rate 
clock = pygame.time.Clock()

# ----------------------------- 
# Fonts and Colors and Images
# -----------------------------

# Fonts
TITTLE_FONT = pygame.font.Font(None,72)
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

# -----------------------------
# Config - Default Settings
# -----------------------------

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
# -----------------------------
# Maze Initialization
# -----------------------------
maze = None
maze_layout = None
agent = None
Q = None # Q-Table
HeatTable = None
# -----------------------------
# Main Program Variables
# -----------------------------

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

# -----------------------------
# FLAGS and Variables
# -----------------------------

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
comparison_expanded_view = None  # Which algorithm is expanded (None = grid view)
comparison_visualization_paused = False  # F key - pause rendering only
comparison_training_paused = False  # T key - pause computation
comparison_show_settings = False  # C key - show settings overlay
comparison_speed_multiplier = 1  # Speed multiplier for comparison mode 
comparison_results = []  # List of {name, type, finished_time, steps, success}
comparison_start_time = 0  # Time when comparison started
comparison_show_results = False  # R key - show results overlay

# Flags for RL Visualisation display options
show_path = False 
show_config = False 
show_q_values = False 
training_active = False # Indicater if training is active
screen_changed = False # Flag to track when screen mode changes

trainer = None # Coroutine for training


# -----------------------------
# Helpers for RL training
# -----------------------------

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

# -----------------------------
# Helpers for SETUP MENU
# -----------------------------

# Setup Menu - JSON Editor variables
new_json_file_name = "default"
### Load JSON data for setup menu
def load_json_for_setup():
    """Load JSON data and create input boxes"""
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

### Save JSON data from setup menu
def save_setup_json():
    """Save the edited values from setup input boxes back to JSON"""
    
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

# Callback for algorithm dropdown
def on_algorithm_selected(algorithm_name):
    """Called when algorithm is selected from dropdown"""
    global selected_algorithm
    selected_algorithm = algorithm_name
    print(f"Selected algorithm: {algorithm_name}")

# Callbacks for mode selection
def select_one_algorithm(monitor):
    """User wants to record ONE algorithm"""
    global record_mode
    record_mode = "ONE"
    set_active_monitor(monitor)

def select_every_algorithm(monitor):
    """User wants to record EVERY algorithm (comparison mode)"""
    global record_mode
    record_mode = "EVERY"
    print("Record EVERY mode selected - will run all algorithms")
    # Go to Comparison_Setup_Menu to create maze
    set_active_monitor("Comparison_Setup_Menu")

# Callbacks for algorithm type selection
def select_rl_algorithms(monitor):
    """User selected RL algorithms"""
    global nonrl_mode
    nonrl_mode = False
    set_active_monitor(monitor)

def select_nonrl_algorithms(monitor):
    """User selected non-RL algorithms"""
    global nonrl_mode
    nonrl_mode = True
    set_active_monitor(monitor)

# Callback for non-RL algorithm selection
def on_nonrl_algorithm_selected(algorithm_name):
    """Called when non-RL algorithm is selected from dropdown"""
    global selected_nonrl_algorithm
    selected_nonrl_algorithm = algorithm_name
    print(f"Selected non-RL algorithm: {algorithm_name}")

def start_nonrl_visualization(monitor):
    """Start non-RL algorithm visualization with current settings"""
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

def start_automated_comparison():
    """Start automated comparison of all algorithms with parallel visualization"""
    global comparison_running, comparison_algorithms, comparison_current_index, comparison_results
    global comparison_visualizers, comparison_trainers, comparison_agents, comparison_q_tables
    global activeMonitor, comparison_start_time
    
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
    comparison_visualizers = {}
    comparison_trainers = {}
    comparison_agents = {}
    
    # Start timing
    import time
    comparison_start_time = time.time()
    comparison_q_tables = {}
    
    print(f"Starting parallel visualization of {len(comparison_algorithms)} algorithms")
    
    # Initialize all algorithms
    for algo in comparison_algorithms:
        algo_name = algo["name"]
        
        if algo["type"] == "RL":
            # Create separate agent and Q-table for each RL algorithm
            comparison_agents[algo_name] = Agent(-5, maze.gridStates, maze.start_pos)
            comparison_q_tables[algo_name] = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
            
            # Create training coroutine
            if algo_name == "SARSA":
                comparison_trainers[algo_name] = sarsa_coroutine(
                    comparison_agents[algo_name], maze, comparison_q_tables[algo_name],
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY
                )
            else:  # Q-Learning
                comparison_trainers[algo_name] = q_learning_coroutine(
                    comparison_agents[algo_name], maze, comparison_q_tables[algo_name],
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY
                )
        else:
            # Create Non-RL visualizer
            solver = get_algorithm(algo_name, maze, maze.start_pos, maze.origin_cor)
            comparison_visualizers[algo_name] = NonRL_Visualizer(maze, solver, agent_img)
            comparison_visualizers[algo_name].start()
    
    # Switch to comparison visualization screen
    activeMonitor = "Comparison_Visualisation"
    print("Comparison visualization started - Press C for settings, F to freeze rendering, T to pause training")

def run_next_comparison_algorithm():
    """Run the next algorithm in the comparison sequence"""
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

def finish_comparison():
    """Save comparison results and reset"""
    global comparison_running
    
    comparison_running = False
    print(f"\nComparison completed! Tested {len(comparison_results)} algorithms.")
    
    # Save comparison results to JSON
    save_comparison_json()
    
    # Return to main menu
    set_active_monitor("Main_Menu")

def save_comparison_json():
    """Save all comparison results to a single JSON file with comprehensive analysis"""
    from datetime import datetime
    
    # Calculate summary statistics
    rl_algorithms = [r for r in comparison_results if r["type"] == "RL"]
    nonrl_algorithms = [r for r in comparison_results if r["type"] == "NonRL"]
    
    # Find best performers
    best_rl_success = max(rl_algorithms, key=lambda x: x["performance_metrics"]["success_rate_percent"]) if rl_algorithms else None
    best_nonrl_time = min(nonrl_algorithms, key=lambda x: x["execution_metrics"]["execution_time_seconds"]) if nonrl_algorithms else None
    best_nonrl_efficiency = min(nonrl_algorithms, key=lambda x: x["path_analysis"]["efficiency_ratio"]) if nonrl_algorithms else None
    
    comparison_data = {
        "metadata": {
            "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "maze_name": maze.name,
            "maze_size": [maze.maze_size_width, maze.maze_size_height],
            "total_maze_cells": maze.maze_size_width * maze.maze_size_height,
            "start_position": maze.start_pos,
            "goal_position": maze.origin_cor,
            "optimal_path_length": len(maze.optimal_path),
            "optimal_path": maze.optimal_path,
            "algorithms_tested": len(comparison_results),
            "rl_algorithms_count": len(rl_algorithms),
            "nonrl_algorithms_count": len(nonrl_algorithms)
        },
        "summary_statistics": {
            "best_rl_algorithm": {
                "name": best_rl_success["algorithm"] if best_rl_success else "N/A",
                "success_rate": best_rl_success["performance_metrics"]["success_rate_percent"] if best_rl_success else 0
            },
            "fastest_nonrl_algorithm": {
                "name": best_nonrl_time["algorithm"] if best_nonrl_time else "N/A",
                "execution_time": best_nonrl_time["execution_metrics"]["execution_time_seconds"] if best_nonrl_time else 0
            },
            "most_efficient_nonrl_algorithm": {
                "name": best_nonrl_efficiency["algorithm"] if best_nonrl_efficiency else "N/A",
                "efficiency_ratio": best_nonrl_efficiency["path_analysis"]["efficiency_ratio"] if best_nonrl_efficiency else 0
            },
            "average_rl_success_rate": round(sum(r["performance_metrics"]["success_rate_percent"] for r in rl_algorithms) / len(rl_algorithms), 2) if rl_algorithms else 0,
            "average_nonrl_execution_time": round(sum(r["execution_metrics"]["execution_time_seconds"] for r in nonrl_algorithms) / len(nonrl_algorithms), 4) if nonrl_algorithms else 0
        },
        "algorithm_rankings": {
            "rl_by_success_rate": sorted(
                [{"algorithm": r["algorithm"], "success_rate": r["performance_metrics"]["success_rate_percent"]} for r in rl_algorithms],
                key=lambda x: x["success_rate"],
                reverse=True
            ),
            "nonrl_by_speed": sorted(
                [{"algorithm": r["algorithm"], "execution_time": r["execution_metrics"]["execution_time_seconds"]} for r in nonrl_algorithms],
                key=lambda x: x["execution_time"]
            ),
            "nonrl_by_efficiency": sorted(
                [{"algorithm": r["algorithm"], "efficiency_ratio": r["path_analysis"]["efficiency_ratio"]} for r in nonrl_algorithms],
                key=lambda x: x["efficiency_ratio"]
            )
        },
        "detailed_results": comparison_results
    }
    
    filename = f"NonRL_Results/Comparison_{maze.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, "w") as f:
            json.dump(comparison_data, f, indent=4)
        print(f"Saved comprehensive comparison results to {filename}")
        print(f"  - {len(rl_algorithms)} RL algorithms")
        print(f"  - {len(nonrl_algorithms)} Non-RL algorithms")
        print(f"  - Best RL: {comparison_data['summary_statistics']['best_rl_algorithm']['name']}")
        print(f"  - Fastest Non-RL: {comparison_data['summary_statistics']['fastest_nonrl_algorithm']['name']}")
    except Exception as e:
        print(f"Error saving comparison JSON: {e}")

# Function to start non-RL visualization

### Initialize setup input boxes
def initialize_setup_input_boxes():
    """Initialize input boxes for setup menu JSON editor - only first 8 variables"""
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

# Initialize comparison setup input boxes
def initialize_comparison_input_boxes():
    """Initialize input boxes for comparison maze setup"""
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

# Initialize RL settings input boxes for comparison mode
def initialize_rl_settings_input_boxes():
    """Initialize input boxes for RL algorithm parameters"""
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

# Save comparison setup and start automated comparison
def save_comparison_and_continue_to_rl_settings(monitor):
    """Save comparison maze setup and go to RL settings screen"""
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

def apply_rl_settings_and_load_maze(monitor):
    """Apply RL settings and create maze, then start comparison"""
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
    # Create agent
    agent = Agent(-5, maze.gridStates, maze.start_pos)
    # Initialize Q-Table
    Q = np.zeros((maze.maze_size_height, maze.maze_size_width, 4), dtype=float)
    # Initialize HeatTable
    HeatTable = [[0 for _ in range(maze.maze_size_width)] for _ in range(maze.maze_size_height)]
    print(f"Created maze for comparison from {new_json_file_name}.json")
    
    # Start automated comparison
    start_automated_comparison()

def save_comparison_and_start(monitor):
    """Save comparison maze setup and start automated comparison"""
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

# Settings monitor variables
def initialize_settings_input_boxes():
    """Initialize input boxes for settings menu"""
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



# -----------------------------
# Q-Learning Coroutine
# -----------------------------
def q_learning_coroutine(agent, maze, Q,
                         EPISODES, max_steps, gamma,
                         EPS0, EPS_MIN, EPS_DECAY,
                         ALPHA0, ALPHA_MIN, ALPHA_DECAY):

    for episode in range(EPISODES):
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
                if not firstFind:
                    global firstEpisode
                    firstEpisode = CURRENT_EPISODE
                    firstFind = True
                numOfReturns += 1
                FindOriginEpisodes.append(CURRENT_EPISODE)
                episode_reached_goal = True
                # Save this path as the final path
                final_path = current_path[:]
                break

            reward, next_state = agent.ProcessNextAction(action)
            x, y   = state
            nx, ny = next_state
            if Record_HeatMap:
                HeatTable[state[1]][state[0]] += 1
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

# -----------------------------
# SARSA Coroutine
# -----------------------------
def sarsa_coroutine(agent, maze, Q,
                    EPISODES, max_steps, gamma,
                    EPS0, EPS_MIN, EPS_DECAY,
                    ALPHA0, ALPHA_MIN, ALPHA_DECAY):

    for episode in range(EPISODES):
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
                if not firstFind:
                    global firstEpisode
                    firstEpisode = CURRENT_EPISODE
                    firstFind = True
                numOfReturns += 1
                FindOriginEpisodes.append(CURRENT_EPISODE)
                episode_reached_goal = True
                # Save this path as the final path
                final_path = current_path[:]
                break

            reward, next_state = agent.ProcessNextAction(action)
            x, y   = state
            nx, ny = next_state
            if Record_HeatMap:
                HeatTable[state[1]][state[0]] += 1
            
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

# -----------------------------
# Display Helpers
# -----------------------------

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

# -----------------------------
# JSON Data Saving after training
# -----------------------------

def save_json_data(maze,
                desccription="Test description",
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
            "description": desccription
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

# -----------------------------
# Monitor Management
# -----------------------------

def set_active_monitor(monitor):
    global activeMonitor
    activeMonitor = monitor

def save_nonrl_json_data(maze, algorithm_name, path_taken, execution_time, success, nodes_explored):
    """Save non-RL algorithm results to JSON"""
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

# -----------------------------
# Monitor Management
# -----------------------------

def stop_training_and_return(monitor):
    """Stop training, reset agent, and return to specified monitor"""
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

# -----------------------------
# Buttons 
# -----------------------------

# HeatMap toggle callback
def on_toggle_heatmap(is_on):
    global Record_HeatMap
    Record_HeatMap = is_on
    print(f"HeatMap recording: {is_on}")

# Function to save current configuration as layout JSON
def save_current_as_layout(monitor):
    """Save current hyperparameters and maze configuration as layout JSON"""
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

# Drop-down menu callback
def on_maze_selected(filename):
    """Callback when a maze file is selected from dropdown - just stores selection"""
    print(f"Selected: {filename}")

# Load and Edit button functionality
def load_maze_and_continue(monitor):
    """Load selected JSON and create maze, then continue based on mode"""
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

def load_and_edit_maze(monitor):
    """Load selected JSON from dropdown and navigate to Setup_Menu for editing"""
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

# Function to initialize/reinitialize all buttons with current screen size
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

# -----------------------------
# Buttons 
# -----------------------------

# Initialize all buttons with current screen size
initialize_all_buttons()

# -----------------------------
# Draw Menus
# -----------------------------

def draw_Mode_Selection():
    """Draw the mode selection screen (ONE vs EVERY algorithm)"""
    title_surf = TITTLE_FONT.render("Select Recording Mode", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    mode_one_button.draw()
    mode_every_button.draw()

def draw_Algorithm_Type_Menu():
    """Draw the algorithm type selection screen (RL vs NOT RL)"""
    title_surf = TITTLE_FONT.render("Select Algorithm Type", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    algorithm_type_rl_button.draw()
    algorithm_type_nonrl_button.draw()

def draw_NonRL_Algorithm_Menu():
    """Draw the non-RL algorithm selection screen"""
    title_surf = TITTLE_FONT.render("Select Non-RL Algorithm", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    # Draw label
    label_surf = INPUT_BOX_FONT.render("Choose Algorithm:", True, pygame.Color("white"))
    screen.blit(label_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), SCREEN_HEIGHT // 2 - int(SCREEN_HEIGHT * 0.12)))
    
    # Draw continue button
    nonrl_continue_button.draw()
    
    # Draw dropdown last so it appears on top
    nonrl_algorithm_dropdown.draw()

def draw_NonRL_Speed_Settings():
    """Draw the speed settings screen for non-RL visualization"""
    title_surf = TITTLE_FONT.render("Set Visualization Speed", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.15)))
    
    # Draw label
    label_surf = INPUT_BOX_FONT.render("Steps per Second:", True, pygame.Color("white"))
    screen.blit(label_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), SCREEN_HEIGHT // 2 + int(SCREEN_HEIGHT * 0.02)))
    
    # Draw speed input box
    nonrl_speed_input.update()
    nonrl_speed_input.draw(screen)
    
    # Draw start button
    nonrl_start_button.draw()

def draw_NonRL_Visualisation():
    """Draw the non-RL algorithm visualization"""
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

def draw_Comparison_RL_Settings():
    """Draw the RL algorithm settings screen for comparison mode"""
    # Display title
    title_surf = TITTLE_FONT.render("RL Algorithm Settings", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.03)))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Configure parameters for Q-Learning and SARSA algorithms:", True, pygame.Color("white"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.25), int(SCREEN_HEIGHT * 0.08)))
    
    # Draw input boxes and labels
    for i, input_box in enumerate(rl_settings_input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = rl_settings_input_box_labels[i]
        screen.blit(label_surf, label_pos)
    
    # Draw continue button
    rl_settings_continue_button.draw()

def draw_RL_Settings():
    """Draw the RL algorithm settings screen for single training"""
    # Display title
    title_surf = TITTLE_FONT.render("RL Algorithm Settings", True, pygame.Color("white"))
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
    title_surf = TITTLE_FONT.render("JSON Setup Editor", True, pygame.Color("white"))
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
    title_surf = TITTLE_FONT.render("Comparison Maze Setup", True, pygame.Color("white"))
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

def draw_Comparison_Visualisation():
    """Draw parallel visualization of all 6 algorithms in grid or expanded view"""
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

def draw_algorithm_in_section(algo, algo_name, x, y, width, height, idx, expanded=False):
    """Draw a single algorithm visualization in its section"""
    # Draw border
    pygame.draw.rect(screen, pygame.Color("white"), (x, y, width, height), 2)
    
    # Draw title background
    title_height = 30 if not expanded else 40
    pygame.draw.rect(screen, pygame.Color(40, 40, 60), (x, y, width, title_height))
    
    # Draw algorithm name
    font = BUTTON_FONT if not expanded else TITTLE_FONT
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

def draw_scaled_maze(x, y, cell_size, algo, algo_name, expanded):
    """Draw maze scaled to fit in section"""
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

def draw_algorithm_stats(algo, algo_name, x, y, width, expanded):
    """Draw statistics for an algorithm"""
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

def draw_comparison_settings_overlay():
    """Draw runtime settings overlay (press C)"""
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
        "  ESC - Exit to grid / Exit comparison",
        "  Click - Expand algorithm to full screen",
        "",
        "Status:",
        f"  Visualization: {'PAUSED' if comparison_visualization_paused else 'RUNNING'}",
        f"  Training: {'PAUSED' if comparison_training_paused else 'RUNNING'}"
    ]
    
    for line in settings_lines:
        line_surf = HYPERPARAMETERS_FONT.render(line, True, pygame.Color("white"))
        screen.blit(line_surf, (box_x + 30, y_offset))
        y_offset += 30
    
    # Close hint
    close_surf = INPUT_BOX_FONT.render("Press C to close", True, pygame.Color("yellow"))
    screen.blit(close_surf, (box_x + box_width // 2 - 70, box_y + box_height - 40))

def draw_comparison_results_overlay():
    """Draw results overlay showing performance statistics"""
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
        
        # Find best performers
        successful_nonrl = [r for r in nonrl_results if r["success"]]
        if successful_nonrl:
            # Find algorithm with fewest steps (minimum)
            min_steps = min(r["steps"] for r in successful_nonrl)
            # Find algorithm with fastest time (minimum)
            min_time = min(r["finished_time"] for r in successful_nonrl)
            
            # Debug output
            print(f"\nDEBUG - NonRL Results:")
            print(f"  Min steps: {min_steps}")
            print(f"  Min time: {min_time:.2f}s")
            print(f"  All algorithms:")
            for r in nonrl_results:
                status = "SUCCESS" if r["success"] else "FAILED"
                marker = ""
                if r["success"] and r["steps"] == min_steps:
                    marker = " ← FEWEST STEPS"
                print(f"    {r['name']}: {r['steps']} steps, {r['finished_time']:.2f}s [{status}]{marker}")
            
            for result in nonrl_results:
                line = f"{result['name']}: {result['steps']} steps, {result['finished_time']:.2f}s"
                color = pygame.Color("green") if result["success"] else pygame.Color("red")
                
                # Highlight best performers
                if result["success"] and result["steps"] == min_steps:
                    line += " (FEWEST STEPS)"
                    color = pygame.Color("gold")
                elif result["success"] and result["finished_time"] == min_time and result["steps"] != min_steps:
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
        
        # Find first to finish (minimum time)
        if rl_results:
            min_finish_time = min(r["finished_time"] for r in rl_results)
            
            for result in rl_results:
                line = f"{result['name']}: Completed in {result['finished_time']:.2f}s"
                color = pygame.Color("green") if result["success"] else pygame.Color("orange")
                
                if result["finished_time"] == min_finish_time:
                    line += " (FIRST DONE)"
                    color = pygame.Color("gold")
                
                result_surf = HYPERPARAMETERS_FONT.render(line, True, color)
                screen.blit(result_surf, (box_x + 40, y_offset))
                y_offset += 28
    
    # Close hint
    y_offset = box_y + box_height - 50
    close_surf = INPUT_BOX_FONT.render("Press R to close | Press Q to quit to Main Menu", True, pygame.Color("yellow"))
    screen.blit(close_surf, (box_x + box_width // 2 - 200, y_offset))

def draw_comparison_status():
    """Draw status indicators at bottom of screen"""
    status_y = SCREEN_HEIGHT - 25
    
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
    title_surf = TITTLE_FONT.render("Load Maze", True, pygame.Color("white"))
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
    title_surf = TITTLE_FONT.render("Choose Option", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.1), int(SCREEN_HEIGHT * 0.12)))
    
    # Draw buttons
    setup_menu_button.draw()
    load_menu_button.draw()

# -----------------------------
# Main Pygame Loop
# -----------------------------

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
            for input_box in rl_settings_input_boxes:
                input_box.handle_event(event)
        
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
                activeMonitor = "Main_Menu"
                comparison_running = False
                comparison_show_results = False
                print("Exited comparison mode")
            if event.key == pygame.K_c:  # Toggle settings overlay
                if not comparison_show_results:
                    comparison_show_settings = not comparison_show_settings
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
                    import time
                    finish_time = time.time() - comparison_start_time
                    agent = comparison_agents[algo_name]
                    # Check if agent reached goal
                    success = agent.activeState == list(maze.origin_cor)
                    comparison_results.append({
                        "name": algo_name,
                        "type": "RL",
                        "finished_time": finish_time,
                        "steps": -1,  # RL doesn't track single episode steps in same way
                        "success": success
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
                            import time
                            finish_time = time.time() - comparison_start_time
                            # Use the algorithm's success flag and solution path length
                            success = viz.algorithm_solver.success
                            steps_taken = len(viz.algorithm_solver.path)
                            comparison_results.append({
                                "name": algo_name,
                                "type": "NonRL",
                                "finished_time": finish_time,
                                "steps": steps_taken,
                                "success": success
                            })
                            print(f"{algo_name} completed: {steps_taken} steps in {finish_time:.2f}s, success={success}")
                            break
        
        # Check if all algorithms finished
        if len(comparison_trainers) == 0:
            all_nonrl_finished = all(
                viz.is_finished for viz in comparison_visualizers.values()
            ) if comparison_visualizers else True
            
            if all_nonrl_finished and not comparison_show_results:
                print("All algorithms completed - Press R to view results")
                comparison_show_results = True
                comparison_training_paused = True

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

# -----------------------------
# Save Data to JSON
# -----------------------------

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


# maze.print_q_values(Q)

# Save HeatTable to CSV
# if Record_HeatMap:
#     with open("HeatMap.csv", "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerows(HeatTable)

# # Save FindOriginEpisodes to CSV
# with open("FindOriginEpisodes.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerows([[value] for value in FindOriginEpisodes])