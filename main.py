import random
import pygame
import numpy as np
import csv
from InputBox import InputBox
from Maze import Maze
from Monitors import Monitors
from Button import Button
from Agent import Agent

pygame.init()

# Screen width and height (on monitor) 
# TODO 
# Make the screen fullscreen for presentation purposes  
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TITTLE_FONT = pygame.font.Font(None,72)
HYPERPARAMETERS_FONT = pygame.font.Font(None, 24)
BUTTON_FONT = pygame.font.Font(None, 30)
BUTTON_FONT_INFLATED = pygame.font.Font(None, 32)
INPUT_BOX_FONT = pygame.font.Font(None, 28)

agent_img = pygame.image.load("Agent.png").convert_alpha()
agent_img = pygame.transform.smoothscale(agent_img, (32, 32))

# -----------------------------
# Helpers for masking
# -----------------------------
def valid_actions(state, maze):
    x, y = state
    return maze.gridStates[y][x].actions

def masked_max(q_row, acts):
    if not acts:  # terminal state
        return 0.0
    return max(q_row[a] for a in acts)

def masked_argmax(q_row, acts):
    if not acts:
        return None
    best = max(q_row[a] for a in acts)
    # tie-break randomly to avoid bias
    best_as = [a for a in acts if q_row[a] == best]
    return random.choice(best_as)

def epsilon_greedy_action(state, Q, epsilon, maze):
    acts = valid_actions(state, maze)
    if not acts:
        return None
    if random.random() < epsilon:
        return random.choice(acts)
    return masked_argmax(Q[state[1], state[0]], acts)

# -----------------------------
# Config - Default Settings
# -----------------------------

FPS = 200

rewardForFinish = 50
rewardForValidMove = -1
rewardForInvalidMove = -10

EPISODES = 5000
CURRENT_EPISODE = 0
max_steps = 100
gamma = 1.0

numOfReturns = 0
firstFind = False
firstEpisode = None
steps_per_frame = 1  # increase to speed up training visual

# epsilon/alpha decay (monotonic with floors)
EPS0, EPS_MIN, EPS_DECAY = 0.9, 0.05, 0.9995
ALPHA0, ALPHA_MIN, ALPHA_DECAY = 0.72, 0.10, 0.997



# Creating Maze from class maze
# (maze_Width, maze_Height, origin_start_pos)
maze1 = Maze(screen, SCREEN_WIDTH, SCREEN_HEIGHT, 200,200,[1,1])

# Make all arrows to point in the dirrection of the origin (necessary in order to make the random suffle work)
maze1.create_default()
# How many random steps is origin going to take to shuffle the maze (this function shuffles the maze)
maze1.random_sequence(5000000)
# Calculating the starting position of the agent (algorithm)
maze1.carve_walls_from_arrows()
maze1.cal_init_pos()
maze1.create_optimal_path(maze1.start_pos)
maze1.create_grid_states(rewardForFinish,rewardForValidMove)

agent = Agent(-5,maze1.gridStates,maze1.start_pos)

Q = np.zeros((maze1.maze_size_height, maze1.maze_size_width, 4), dtype=float)

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

        if episode % 100 == 0 and Record_HeatMap:
            with open(f"HeatMap{CURRENT_EPISODE}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(HeatTable)


        for t in range(max_steps):
            if action is None:  # terminal state (goal)
                if not firstFind:
                    global firstEpisode
                    firstEpisode = CURRENT_EPISODE
                    firstFind = True
                numOfReturns += 1
                FindNotFind.append(CURRENT_EPISODE)

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
def draw_Main_Menu():
    start_button.draw()
    settings_button.draw()
def draw_Settings_Menu():

    mainMenu_button.draw()
    # Draw input boxes and labels
    for i, input_box in enumerate(input_boxes):
        input_box.update()
        input_box.draw(screen)
        label_surf, label_pos = input_box_label[i]
        screen.blit(label_surf, label_pos)
def set_active_monitor(monitor):
    global activeMonitor
    activeMonitor = monitor


clock = pygame.time.Clock()

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

# Settgings monitor variables
# ACTIVE_COLOR = pygame.Color(193, 73, 83)
ACTIVE_COLOR = pygame.Color(3, 29, 68)
INACTIVE_COLOR = pygame.Color(255, 255, 255)
# INACTIVE_COLOR = pygame.Color(230, 170, 104)
TEXT_SAVED_COLOR = pygame.Color(178, 255, 169)

input_boxes = []
input_box_label = []
input_box_width = 140
input_box_height = 32
input_box_x = SCREEN_WIDTH // 4 - 150
input_box_y_start = SCREEN_HEIGHT // 4 - 100
input_box_gap = 70
labels = [
    "Episodes",
    "Max Steps/Episode",
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
    EPISODES, max_steps, gamma, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, steps_per_frame, FPS
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

def apply_input_box_values():
    global EPISODES, max_steps, gamma, EPS0, EPS_MIN, EPS_DECAY, ALPHA0, ALPHA_MIN, ALPHA_DECAY, steps_per_frame, FPS
    mapping = [
        ("int", "EPISODES"),
        ("int", "max_steps"),
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

        if name == "EPISODES": EPISODES = val
        elif name == "max_steps": max_steps = val
        elif name == "gamma": gamma = val
        elif name == "EPS0": EPS0 = val
        elif name == "EPS_MIN": EPS_MIN = val
        elif name == "EPS_DECAY": EPS_DECAY = val
        elif name == "ALPHA0": ALPHA0 = val
        elif name == "ALPHA_MIN": ALPHA_MIN = val
        elif name == "ALPHA_DECAY": ALPHA_DECAY = val
        elif name == "steps_per_frame": steps_per_frame = val
        elif name == "FPS": FPS = val





screenArray = Monitors()
activeMonitor = screenArray.monitors[0]

Record_HeatMap = True
HeatTable = [[0 for _ in range(maze1.maze_size_width)] for _ in range(maze1.maze_size_height)]

FindNotFind = []

running = True
show_path = False
show_config = False

trainer = None
training_active = False

# Main loop of the program
# constants to reuse
BASE_RECT_SIZE = 100
HMARGIN = 100
VMARGIN = 100

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if activeMonitor == "Settings":
            for input_box in input_boxes:
                if input_box.handle_event(event):
                    apply_input_box_values()
        if event.type == pygame.KEYDOWN and activeMonitor == "RL - Visualisation":
            if event.key == pygame.K_i:
                show_path = not show_path
            if event.key == pygame.K_v: # Show Config Values on Screen
                show_config = not show_config
            if event.key == pygame.K_s:  # start/stop training
                if not training_active:
                    trainer = q_learning_coroutine(
                        agent, maze1, Q,
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
    if activeMonitor == "Main_Menu":
        draw_Main_Menu()
    elif activeMonitor == "Settings":
        # draw settings menu
        draw_Settings_Menu()
    # draw maze and agent
    elif activeMonitor == "RL - Visualisation":
        maze1.draw_maze(BASE_RECT_SIZE, HMARGIN, VMARGIN, False, True)
        maze1.draw_agent(agent, BASE_RECT_SIZE, HMARGIN, VMARGIN, agent_img)
        if show_path:
            maze1.draw_optimal_path(HMARGIN,VMARGIN,BASE_RECT_SIZE,maze1.start_pos)
        if show_config:
            showConfigValuesOnScreen()

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


pygame.quit()
if Record_HeatMap:
    with open("HeatMap.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(HeatTable)

with open("FindNotFind.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows([[value] for value in FindNotFind])