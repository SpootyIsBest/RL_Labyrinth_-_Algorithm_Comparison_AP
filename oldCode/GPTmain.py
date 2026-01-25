import numpy as np
import pygame
import random
from Agent import Agent
from State import State

# --- RL Parameters ---
rewardForFinish = 5
rewardForValidMove = -1
rewardForInvalidMove = -10
gridWidth = 10
gridHeight = 5
initialPosition = [0, 4]
goal_x, goal_y = gridWidth - 1, 0

# --- Build Grid (with barriers by removing actions) ---
gridStates = []
for y in range(gridHeight):
    row = []
    for x in range(gridWidth):
        actions = [0, 1, 2, 3]  # left, right, up, down
        if x == 0:
            actions.remove(0)
        if x == gridWidth - 1:
            actions.remove(1)
        if y == 0:
            actions.remove(2)
        if y == gridHeight - 1:
            actions.remove(3)

        # Custom barriers (disabling actions)
        if x == 2 and y >= 1 and 1 in actions: actions.remove(1)
        if x == 4 and y >= 1 and 0 in actions: actions.remove(0)
        if x == 3 and y == 0 and 3 in actions: actions.remove(3)
        if x == 7 and y <= 3 and 1 in actions: actions.remove(1)
        if x == 9 and y <= 3 and 0 in actions: actions.remove(0)
        if x == 8 and y == 4 and 2 in actions: actions.remove(2)

        reward = rewardForFinish if (x == goal_x and y == goal_y) else rewardForValidMove
        row.append(State(reward, actions, [x, y]))
    gridStates.append(row)

agent = Agent(rewardForInvalidMove, gridStates)

# --- Hyperparameters ---
alpha = 0.72
gamma = 1
epsilon = 0.71
EPISODES = 300
max_steps = 65

# --- Q-Table ---
Q = np.zeros((gridHeight, gridWidth, len(agent.actionOptions)))

def epsilon_greedy_action(state):
    x, y = state
    global epsilon
    return random.choice(agent.actionOptions) if random.uniform(0, 1) < epsilon else np.argmax(Q[y, x])

# ================= PYGAME VISUALIZATION =================
pygame.init()
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_AREA_SIZE = 500  # smaller grid to leave room for info text
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RL Agent Learning Visualization")
clock = pygame.time.Clock()
FPS = 65

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Fonts
pygame.font.init()
info_font = pygame.font.SysFont('Arial', 18)
q_font = pygame.font.SysFont('Arial', 8)

# Tile size and offsets
tile_width = GRID_AREA_SIZE // gridWidth
tile_height = GRID_AREA_SIZE // gridHeight
tile_size = min(tile_width, tile_height)
grid_pixel_width = gridWidth * tile_size
grid_pixel_height = gridHeight * tile_size
offset_x = (SCREEN_WIDTH - grid_pixel_width) // 2
offset_y = (SCREEN_HEIGHT - grid_pixel_height - 100) // 2  # leave 100px bottom for text

# Rect size variable
RECT_SCALE = 0.6
rect_size = int(tile_size * RECT_SCALE)
rect_offset = (tile_size - rect_size) // 2

# Player rect
player = pygame.Rect(0, 0, rect_size, rect_size)

def draw_arrow(x, y, dx, dy, color=WHITE):
    cx = offset_x + x * tile_size + tile_size // 2
    cy = offset_y + y * tile_size + tile_size // 2
    arrow_size = tile_size // 4

    if dx == 1 and dy == 0:   # right
        points = [(cx - arrow_size, cy - arrow_size), (cx - arrow_size, cy + arrow_size), (cx + arrow_size, cy)]
    elif dx == -1 and dy == 0: # left
        points = [(cx + arrow_size, cy - arrow_size), (cx + arrow_size, cy + arrow_size), (cx - arrow_size, cy)]
    elif dx == 0 and dy == -1: # up
        points = [(cx - arrow_size, cy + arrow_size), (cx + arrow_size, cy + arrow_size), (cx, cy - arrow_size)]
    elif dx == 0 and dy == 1:  # down
        points = [(cx - arrow_size, cy - arrow_size), (cx + arrow_size, cy - arrow_size), (cx, cy + arrow_size)]
    else:
        return
    pygame.draw.polygon(SCREEN, color, points)

def draw_grid():
    SCREEN.fill(BLACK)
    # Grid lines
    for x in range(0, grid_pixel_width + 1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (offset_x + x, offset_y), (offset_x + x, offset_y + grid_pixel_height))
    for y in range(0, grid_pixel_height + 1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (offset_x, offset_y + y), (offset_x + grid_pixel_width, offset_y + y))

    # Barriers as walls + Q-values
    for y in range(gridHeight):
        for x in range(gridWidth):
            state = gridStates[y][x]
            cell_x = offset_x + x * tile_size
            cell_y = offset_y + y * tile_size
            if 0 not in state.possibleActions:
                pygame.draw.line(SCREEN, RED, (cell_x, cell_y), (cell_x, cell_y + tile_size), 4)
            if 1 not in state.possibleActions:
                pygame.draw.line(SCREEN, RED, (cell_x + tile_size, cell_y), (cell_x + tile_size, cell_y + tile_size), 4)
            if 2 not in state.possibleActions:
                pygame.draw.line(SCREEN, RED, (cell_x, cell_y), (cell_x + tile_size, cell_y), 4)
            if 3 not in state.possibleActions:
                pygame.draw.line(SCREEN, RED, (cell_x, cell_y + tile_size), (cell_x + tile_size, cell_y + tile_size), 4)

            # --- Draw Q-values with color gradient ---
            q_values = Q[y, x]
            sorted_indices = np.argsort(q_values)  # worst -> best
            color_map = [None] * 4
            color_map[sorted_indices[0]] = RED
            color_map[sorted_indices[1]] = ORANGE
            color_map[sorted_indices[2]] = ORANGE
            color_map[sorted_indices[3]] = GREEN

            up_txt = q_font.render(f"{q_values[2]:.1f}", True, color_map[2])
            down_txt = q_font.render(f"{q_values[3]:.1f}", True, color_map[3])
            left_txt = q_font.render(f"{q_values[0]:.1f}", True, color_map[0])
            right_txt = q_font.render(f"{q_values[1]:.1f}", True, color_map[1])
            SCREEN.blit(up_txt, (cell_x + tile_size/2 - up_txt.get_width()/2, cell_y + 2))
            SCREEN.blit(down_txt, (cell_x + tile_size/2 - down_txt.get_width()/2, cell_y + tile_size - down_txt.get_height() - 2))
            SCREEN.blit(left_txt, (cell_x + 2, cell_y + tile_size/2 - left_txt.get_height()/2))
            SCREEN.blit(right_txt, (cell_x + tile_size - right_txt.get_width() - 2, cell_y + tile_size/2 - right_txt.get_height()/2))

    # Start and goal
    sx, sy = initialPosition
    gx, gy = goal_x, goal_y
    start_rect = pygame.Rect(offset_x + sx * tile_size + rect_offset,
                             offset_y + sy * tile_size + rect_offset,
                             rect_size, rect_size)
    goal_rect = pygame.Rect(offset_x + gx * tile_size + rect_offset,
                            offset_y + gy * tile_size + rect_offset,
                            rect_size, rect_size)
    pygame.draw.rect(SCREEN, WHITE, start_rect)
    pygame.draw.rect(SCREEN, BLUE, goal_rect)

def draw_info(episode, step, finishes, first_finish, max_steps, alpha, epsilon, gamma):
    # First line
    info_text1 = f"Episode: {episode+1 if episode>=0 else '-'}/{EPISODES}  Step: {step}  Finishes: {finishes}  First Finish: {first_finish}"
    text_surface1 = info_font.render(info_text1, True, WHITE)
    SCREEN.blit(text_surface1, (20, SCREEN_HEIGHT - 80))

    # Second line
    info_text2 = f"Max Steps: {max_steps}   Alpha: {alpha:.3f}   Epsilon: {epsilon:.3f}   Gamma: {gamma:.3f}"
    text_surface2 = info_font.render(info_text2, True, WHITE)
    SCREEN.blit(text_surface2, (20, SCREEN_HEIGHT - 50))

def move_player(grid_pos):
    x, y = grid_pos
    player.topleft = (offset_x + x * tile_size + rect_offset,
                      offset_y + y * tile_size + rect_offset)
    pygame.draw.rect(SCREEN, GREEN, player)

# --- Training with visualization ---
first_finish_episode = None
total_finishes = 0
first_finish_path = []

for episode in range(EPISODES):
    agent.activeState = initialPosition[:]
    agent.activeReward = 0
    state = agent.activeState
    action = epsilon_greedy_action(state)
    current_path = [tuple(agent.activeState)]

    for step in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()

        reward, next_state = agent.ProcessNextAction(action)
        next_action = epsilon_greedy_action(next_state)
        x, y = state
        nx, ny = next_state
        Q[y, x, action] += alpha * (reward + gamma * Q[ny, nx, next_action] - Q[y, x, action])
        state, action = next_state, next_action
        current_path.append(tuple(agent.activeState))

        clock.tick(FPS)
        draw_grid()
        move_player(agent.activeState)
        draw_info(episode, step, total_finishes, first_finish_episode, max_steps, alpha, epsilon, gamma)
        pygame.display.flip()

        if agent.activeState == [goal_x, goal_y]:
            total_finishes += 1
            if first_finish_episode is None:
                first_finish_episode = episode
                first_finish_path = current_path[:]
            # ↓ decrease alpha and epsilon only when goal is reached
            if alpha > 0.1: 
                alpha -= 0.002
            if epsilon > 0.01: 
                epsilon -= 0.002
            break

print(f"Training finished!\nFirst goal reached at episode: {first_finish_episode}")
print(f"Total number of goal finishes: {total_finishes}")

# --- Compute optimal path ---
def compute_optimal_path(Q, start, goal):
    path = [tuple(start)]
    state = start[:]
    visited = set()
    for _ in range(gridWidth * gridHeight * 2):
        if tuple(state) == tuple(goal): break
        x, y = state
        action = np.argmax(Q[y, x])
        if action == 0: state = [x - 1, y]
        elif action == 1: state = [x + 1, y]
        elif action == 2: state = [x, y - 1]
        elif action == 3: state = [x, y + 1]
        if tuple(state) in visited: break
        visited.add(tuple(state))
        path.append(tuple(state))
    return path

optimal_path = compute_optimal_path(Q, initialPosition, [goal_x, goal_y])

def animate_path(path, fps, title):
    print(f"Animating: {title}")
    for step, pos in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
        clock.tick(fps)
        draw_grid()
        # Draw arrows for entire path
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            draw_arrow(x1, y1, x2-x1, y2-y1, WHITE)
        move_player(pos)
        draw_info(-1, step, total_finishes, first_finish_episode, max_steps, alpha, epsilon, gamma)
        pygame.display.flip()

# --- Render first finish path (if available) ---
if first_finish_path:
    FPS_FIRST_PATH = 5
    animate_path(first_finish_path, FPS_FIRST_PATH, "First Finish Path")

# --- Render optimal path ---
FPS_OPTIMAL_PATH = 10
animate_path(optimal_path, FPS_OPTIMAL_PATH, "Optimal Path")

# --- Continuous optimal path looping ---
path_index = 0
while True:
    clock.tick(FPS_OPTIMAL_PATH)
    draw_grid()
    for i in range(len(optimal_path)-1):
        x1, y1 = optimal_path[i]
        x2, y2 = optimal_path[i+1]
        draw_arrow(x1, y1, x2-x1, y2-y1, WHITE)
    move_player(optimal_path[path_index])
    draw_info(-1, path_index, total_finishes, first_finish_episode, max_steps, alpha, epsilon, gamma)
    pygame.display.flip()
    path_index = (path_index + 1) % len(optimal_path)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); quit()