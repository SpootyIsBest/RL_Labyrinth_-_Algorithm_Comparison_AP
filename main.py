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

#-----------------------------
# Pygame Setup
#-----------------------------

# Initialize Pygame
pygame.init()

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Create the main display surface
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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

# Set initial active monitor
activeMonitor = screenArray.monitors[0]

# -----------------------------
# FLAGS and Variables
# -----------------------------

# Main loop flag
running = True
# Flags for settings
Record_HeatMap = True 

# Flags for RL Visualisation display options
show_path = False 
show_config = False 
show_q_values = False 
training_active = False # Indicater if training is active

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
    
    # Update only the first 8 editable fields from input boxes
    max_input_fields = 8
    editable_keys = list(data.keys())[:max_input_fields]
    
    for i, key in enumerate(editable_keys):
        input_box = setup_input_boxes[i]
        value = input_box.text
        
        # Try to convert to appropriate type
        try:
            # Try to parse as JSON to handle lists, numbers, etc.
            data[key] = json.loads(value)
        except:
            # If it fails, keep as string
            data[key] = value
    
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
    setup_box_width = 200
    setup_box_height = 32
    setup_box_x = SCREEN_WIDTH // 2 - 50
    setup_box_y_start = 100
    setup_box_gap = 45
    
    for i, (key, value) in enumerate(zip(editable_keys, editable_values)):
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
        setup_input_boxes.append(input_box)
        
        label_surface = INPUT_BOX_FONT.render(f"{key}:", True, pygame.Color("white"))
        label_pos = (setup_box_x - 250, setup_box_y_start + i * setup_box_gap + 5)
        setup_input_box_labels.append((label_surface, label_pos))
    
    return setup_input_boxes, setup_input_box_labels, setup_json_keys

# Initialize setup input boxes
setup_input_boxes, setup_input_box_labels, setup_json_keys = initialize_setup_input_boxes()

# Save JSON and return to main menu for setup menu
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
    # Set active monitor back to main menu
    set_active_monitor(monitor)

# Create maze from JSON data
def create_maze_from_json(json_file_name):
    global maze, EPISODES, max_steps
    json_file = f"JsonData/{json_file_name}.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    maze_name = data.get("name", "MAZE_NAME_NOT_FOUND")
    maze_width = data.get("MazeSize", [3, 3])[0]
    maze_height = data.get("MazeSize", [3, 3])[1]
    origin_start_pos = [1,1] # default origin start pos
    rewardForFinish = data.get("rewardForFinish", 50)
    rewardForValidMove = data.get("rewardForValidMove", -1)
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
    input_box_width = 140
    input_box_height = 32
    input_box_x = SCREEN_WIDTH // 4 - 150
    input_box_y_start = SCREEN_HEIGHT // 4 - 100
    input_box_gap = 70
    
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
    column_gap = 150
    
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
        CURRENT_EPISODE = episode + 1
        epsilon = max(EPS_MIN, EPS0 * (EPS_DECAY ** episode))
        alpha   = max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** episode))

        agent.reset()
        state = agent.activeState[:]
        action = epsilon_greedy_action(state, Q, epsilon, maze)

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

# -----------------------------
# Buttons 
# -----------------------------

# BUTTONS for Main Menu
start_button = Button(screen,
                    "RL - Visualisation",
                    "RL - Visualisation",
                    set_active_monitor,
                    BUTTON_FONT,
                    BUTTON_FONT_INFLATED,
                    200,
                    100, 
                    (SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 - 50),
                    7,
                    20
                    )

settings_button = Button(screen,
                    "Settings", 
                    "Settings",
                    set_active_monitor, 
                    BUTTON_FONT, 
                    BUTTON_FONT_INFLATED, 
                    200,
                    100, 
                    (SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 + 100),
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
                    200,
                    100, 
                    (SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 + 250),
                    7,
                    20
                    )

# HeatMap Record Toggle Button
def on_toggle_heatmap(is_on):
    global Record_HeatMap
    Record_HeatMap = is_on
    print(f"HeatMap recording: {is_on}")
heatMap_Button_OnOFF = Button_On_Off(
                    screen,
                    "Record HeatMap",
                    on_change_monitor=on_toggle_heatmap,
                    pos=(SCREEN_WIDTH - 250, SCREEN_HEIGHT - 100),
                    button_font=BUTTON_FONT,
                    button_font_inflated=BUTTON_FONT_INFLATED
                    )
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

# Save Layout Button for Settings
save_layout_button = Button(
    screen,
    "Save as Layout",
    "Settings",
    save_current_as_layout,
    BUTTON_FONT,
    BUTTON_FONT_INFLATED,
    180,
    80,
    (SCREEN_WIDTH - 50,100),
    7,
    20
)


# BUTTONS for Setup Menu
save_setup_button = Button(
    screen,
    "Save JSON", 
    "Main_Menu",  # Return to Main_Menu after saving
    save_create_maze_and_return_to_main,
    BUTTON_FONT, 
    BUTTON_FONT_INFLATED, 
    150,
    80, 
    (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 120),
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
    250,
    100,
    (SCREEN_WIDTH/2 - 125, SCREEN_HEIGHT/2 - 100),
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
    250,
    100,
    (SCREEN_WIDTH/2 - 125, SCREEN_HEIGHT/2 + 50),
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
    150,
    80,
    (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 120),
    7,
    20
)

# Drop-down menu for Load Menu
def on_maze_selected(filename):
    """Callback when a maze file is selected from dropdown - just stores selection"""
    print(f"Selected: {filename}")

# Load and Edit button functionality
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
    
    # Navigate to Setup_Menu
    set_active_monitor(monitor)
    print(f"Loaded {selected_file} for editing")

maze_dropdown = Drop_Down_Menu(
    screen,
    folder_path="MazeLayouts",
    on_select_callback=on_maze_selected,
    button_font=INPUT_BOX_FONT,
    width=400,
    height=50,
    pos=(SCREEN_WIDTH // 2 - 200, 150),
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
    200,
    80,
    (SCREEN_WIDTH // 2 - 100, 250),
    7,
    20
)

# -----------------------------
# Draw Menus
# -----------------------------

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

def draw_Main_Menu():
    start_button.draw()
    settings_button.draw()

def draw_Settings_Menu():

    mainMenu_button.draw()
    heatMap_Button_OnOFF.draw()
    save_layout_button.draw()
    # Draw input boxes and labels
    for i, input_box in enumerate(input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = input_box_label[i]
        screen.blit(label_surf, label_pos)

def draw_Setup_Menu():
    # Display title
    title_surf = TITTLE_FONT.render("JSON Setup Editor", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - 180, 30))
    
    # Draw input boxes and labels
    for i, input_box in enumerate(setup_input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = setup_input_box_labels[i]
        screen.blit(label_surf, label_pos)
    
    # Draw buttons
    save_setup_button.draw()

def draw_Load_Menu():
    title_surf = TITTLE_FONT.render("Load Maze", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - 150, 30))
    
    # Draw instructions
    instruction_surf = INPUT_BOX_FONT.render("Select a maze layout to load:", True, pygame.Color("white"))
    screen.blit(instruction_surf, (SCREEN_WIDTH // 2 - 200, 110))
    
    # Draw dropdown menu
    maze_dropdown.draw()
    
    # Draw Load & Edit button
    load_edit_button.draw()
    
    # Draw back button
    back_to_choose_button.draw()

def draw_Choose_Menu():
    # Display title
    title_surf = TITTLE_FONT.render("Choose Option", True, pygame.Color("white"))
    screen.blit(title_surf, (SCREEN_WIDTH // 2 - 150, 100))
    
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
        if activeMonitor == "Settings":
            for input_box in input_boxes:
                if input_box.handle_event(event):
                    apply_input_box_values()
        
        if activeMonitor == "Setup_Menu":
            for input_box in setup_input_boxes:
                input_box.handle_event(event)
        
        if activeMonitor == "Load_Menu":
            maze_dropdown.handle_event(event)
        
        if event.type == pygame.KEYDOWN and activeMonitor == "RL - Visualisation":
            if event.key == pygame.K_q: # Show Q-Values on Screen
                show_q_values = not show_q_values
            if event.key == pygame.K_i: # Show Optimal Path on Screen
                show_path = not show_path
            if event.key == pygame.K_v: # Show Config Values on Screen
                show_config = not show_config
            if event.key == pygame.K_s:  # start/stop training
                if not training_active:
                    trainer = q_learning_coroutine(
                        agent, maze, Q,
                        EPISODES, max_steps, gamma,
                        EPS0, EPS_MIN, EPS_DECAY,
                        ALPHA0, ALPHA_MIN, ALPHA_DECAY
                    )
                    training_active = True
                    print("Training started")
                else:
                    training_active = False
                    trainer = None
                    print("Training stopped manually")

    screen.fill((0, 0, 0))

    # draw active monitors
    if activeMonitor == "Choose_Menu":
        draw_Choose_Menu()
    elif activeMonitor == "Setup_Menu":
        draw_Setup_Menu()
    elif activeMonitor == "Load_Menu":
        draw_Load_Menu()
    elif activeMonitor == "Main_Menu":
        draw_Main_Menu()
    elif activeMonitor == "Settings":
        draw_Settings_Menu()
    # draw maze and agent
    elif activeMonitor == "RL - Visualisation":
        draw_RL_Visualisation()
        

    # advance training coroutine a bit each frame
    if training_active and trainer is not None:
        try:
            for _ in range(steps_per_frame):
                next(trainer)
        except StopIteration:
            training_active = False
            trainer = None
            print("Training finished")

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
    final_path=[],
    optimal_path_num_of_steps=len(maze.optimal_path),
    final_path_num_of_steps=-1,
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