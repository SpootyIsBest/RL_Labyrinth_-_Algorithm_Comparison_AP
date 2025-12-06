import numpy as np
import random
import math

# -----------------------------
# Environment + models
# -----------------------------
class State:
    def __init__(self, reward, actions, pos):
        self.reward = reward          # reward for entering this cell
        self.actions = actions[:]     # allowed actions from this cell
        self.pos = pos[:]             # [x, y]

class Agent:
    # actions: 0=left, 1=right, 2=up, 3=down
    def __init__(self, invalid_reward, grid_states, initial_pos):
        self.actionOptions = [0, 1, 2, 3]
        self.invalid_reward = invalid_reward
        self.grid = grid_states
        self.initial_pos = initial_pos[:]
        self.reset()

    def reset(self):
        self.activeState = self.initial_pos[:]
        self.activeReward = 0

    def _in_bounds(self, x, y):
        return 0 <= x < len(self.grid[0]) and 0 <= y < len(self.grid)

    def ProcessNextAction(self, action):
        """
        Returns: (reward, next_state[list[x,y]])
        - If action is illegal from current cell, gives invalid penalty and stays in place.
        - Otherwise moves into the next cell and receives that cell's reward.
        """
        x, y = self.activeState
        allowed = self.grid[y][x].actions

        if action not in allowed:
            r = self.invalid_reward
            next_state = [x, y]
            self.activeReward += r
            return r, next_state

        if action == 0:   nx, ny = x - 1, y
        elif action == 1: nx, ny = x + 1, y
        elif action == 2: nx, ny = x, y - 1
        else:             nx, ny = x, y + 1

        # Safety (should be guaranteed by allowed actions)
        if not self._in_bounds(nx, ny):
            r = self.invalid_reward
            next_state = [x, y]
            self.activeReward += r
            return r, next_state

        r = self.grid[ny][nx].reward
        self.activeState = [nx, ny]
        self.activeReward += r
        return r, [nx, ny]

# -----------------------------
# Config
# -----------------------------
rewardForFinish = 50
rewardForValidMove = -1
rewardForInvalidMove = -10

gridWidth = 20
gridHeight = 15

initialPosition = [0, gridHeight - 1]
goal_x, goal_y = gridWidth - 1, 0

EPISODES = 200
max_steps = 1000
gamma = 1.0

# epsilon/alpha decay (monotonic with floors)
EPS0, EPS_MIN, EPS_DECAY = 0.9, 0.05, 0.995
ALPHA0, ALPHA_MIN, ALPHA_DECAY = 0.72, 0.10, 0.997

# -----------------------------
# Build grid with barriers & allowed actions
# -----------------------------
gridStates = []
for y in range(gridHeight):
    row = []
    for x in range(gridWidth):
        actions = [0, 1, 2, 3]  # left right up down

        # Border walls
        if x == 0 and 0 in actions: actions.remove(0)
        if x == gridWidth - 1 and 1 in actions: actions.remove(1)
        if y == 0 and 2 in actions: actions.remove(2)
        if y == gridHeight - 1 and 3 in actions: actions.remove(3)

        # Barriers (same as your original logic)
        if x == 2 and y >= 1 and 1 in actions:
            actions.remove(1)
        if x == 4 and y >= 1 and 0 in actions:
            actions.remove(0)
        if x == 3 and y == 0 and 3 in actions:
            actions.remove(3)

        if x == 7 and y <= 3 and 1 in actions:
            actions.remove(1)
        if x == 9 and y <= 3 and 0 in actions:
            actions.remove(0)
        if x == 8 and y == 4 and 2 in actions:
            actions.remove(2)

        # Goal cell: no outgoing actions; entering it gives finish reward
        if x == goal_x and y == goal_y:
            reward = rewardForFinish
            actions = []
        else:
            reward = rewardForValidMove

        row.append(State(reward, actions, [x, y]))
    gridStates.append(row)

# -----------------------------
# Helpers for masking
# -----------------------------
def valid_actions(state):
    x, y = state
    return gridStates[y][x].actions

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

def epsilon_greedy_action(state, Q, epsilon):
    acts = valid_actions(state)
    if not acts:
        return None
    if random.random() < epsilon:
        return random.choice(acts)
    return masked_argmax(Q[state[1], state[0]], acts)

# -----------------------------
# Init agent and Q-table
# -----------------------------
agent = Agent(rewardForInvalidMove, gridStates, initialPosition)
Q = np.zeros((gridHeight, gridWidth, 4), dtype=float)

# Stats
firstFind = False
firstEpisode = -1
numOfReturns = 0

# -----------------------------
# Training
# -----------------------------
for episode in range(EPISODES):
    # Decay schedules per episode
    epsilon = max(EPS_MIN, EPS0 * (EPS_DECAY ** episode))
    alpha = max(ALPHA_MIN, ALPHA0 * (ALPHA_DECAY ** episode))

    agent.reset()
    state = agent.activeState[:]
    action = epsilon_greedy_action(state, Q, epsilon)

    for t in range(max_steps):
        if action is None:  # terminal state (should only happen at goal)
            break

        reward, next_state = agent.ProcessNextAction(action)

        x, y = state
        nx, ny = next_state

        # Q-learning target with masked max over legal next actions
        next_acts = valid_actions(next_state)
        target = reward + gamma * masked_max(Q[ny, nx], next_acts)
        Q[y, x, action] += alpha * (target - Q[y, x, action])

        # Next action for behavior (off-policy Q-learning can still behave ε-greedy)
        state = next_state
        action = epsilon_greedy_action(state, Q, epsilon)

        # Check goal
        if state == [goal_x, goal_y]:
            numOfReturns += 1
            if not firstFind:
                firstFind = True
                firstEpisode = episode
            break

    # (Optional) progress print
    # print(f"Episode {episode+1}/{EPISODES} | total reward: {agent.activeReward:.1f} | epsilon: {epsilon:.3f} | alpha: {alpha:.3f}")

print("Training finished!")
print(f"NUMBER OF FINISHES: {numOfReturns}")
print(f"FIRST GOAL EPISODE: {firstEpisode}")

# -----------------------------
# Reporting helpers
# -----------------------------
def print_q_table(Q):
    print(f"{'State':>8} | {'Left':>7} {'Right':>7} {'Up':>7} {'Down':>7}")
    print("-" * 42)
    for y in range(gridHeight):
        for x in range(gridWidth):
            values = Q[y, x]
            print(f"({x},{y})   | {values[0]:7.2f} {values[1]:7.2f} {values[2]:7.2f} {values[3]:7.2f}")
        print("-" * 42)

def print_optimal_path_visual(Q, start, goal):
    state = start[:]
    path = [tuple(state)]
    visited = set([tuple(state)])
    safety_limit = gridWidth * gridHeight * 2

    while tuple(state) != tuple(goal) and len(path) < safety_limit:
        x, y = state
        acts = valid_actions(state)
        if not acts:
            break  # terminal (e.g., goal)

        a = masked_argmax(Q[y, x], acts)
        if a is None:
            break

        if a == 0:   nxt = [x - 1, y]
        elif a == 1: nxt = [x + 1, y]
        elif a == 2: nxt = [x, y - 1]
        else:        nxt = [x, y + 1]

        # If the chosen action is somehow invalid (shouldn't happen), stop
        if a not in acts:
            print("Chosen illegal action in visualizer; stopping.")
            break

        if tuple(nxt) in visited:
            print("Loop detected, stopping path trace.")
            break

        state = nxt
        visited.add(tuple(state))
        path.append(tuple(state))

    print("Optimal path:", path)

    # Draw grid
    grid_repr = [[" . " for _ in range(gridWidth)] for _ in range(gridHeight)]
    for (x, y) in path[1:-1]:
        grid_repr[y][x] = " * "
    sx, sy = start
    gx, gy = goal
    grid_repr[sy][sx] = " S "
    grid_repr[gy][gx] = " G "

    print("\nGrid with path:")
    for y in range(gridHeight):
        print("".join(grid_repr[y]))

# -----------------------------
# Output
# -----------------------------
print_q_table(Q)
print_optimal_path_visual(Q, start=initialPosition, goal=[goal_x, goal_y])