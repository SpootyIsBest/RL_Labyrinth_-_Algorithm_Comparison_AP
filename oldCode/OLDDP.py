import numpy as np
import pygame
import random

class State:
    def __init__(self, reward, possibleActions, position):
        self.reward = reward
        self.possibleActions = possibleActions
        self.position = position

class Agent:
    def __init__(self, rewardOutOfGrid, grid):
        self.activeReward = 0
        self.activeState = [0, 0]  # x=0, y=0
        self.changePosArray = [[-1, 0], [1, 0], [0, -1], [0, 1]] #  Left, Right, Up, Down
        self.actionOptions = [0, 1, 2, 3]  # indexes to changePosArray 
        self.rewardOutOfGrid = rewardOutOfGrid
        self.grid = grid

    def random_max_with_index(self, numbers):
        max_val = float('-inf')
        indices = []
        for i, val in enumerate(numbers):
            if val > max_val:
                max_val = val
                indices = [i]
            elif val == max_val:
                indices.append(i)
        chosen_index = random.choice(indices)
        return max_val, chosen_index
    
    def MakeAction(self, action):
        x, y = self.activeState
        state = self.grid[y][x]  # grid[row][col]
        
        if action in state.possibleActions:
            dx, dy = self.changePosArray[action]
            new_x = x + dx
            new_y = y + dy
            self.activeState = [new_x, new_y]
            newState = self.grid[new_y][new_x]
            self.activeReward += newState.reward
            # print(f"Moved to {self.activeState}\nNew active reward: {self.activeReward}")
        else:
            self.activeReward += self.rewardOutOfGrid
            # print(f"Action not allowed in this state stayed on: {self.activeState}!\nActive reward: {self.activeReward}")
    
    def CheckNextAction(self, action):
        x, y = self.activeState
        state = self.grid[y][x]  # grid[row][col]

        if action in state.possibleActions:
            dx, dy = self.changePosArray[action]
            new_x = x + dx
            new_y = y + dy
            newState = self.grid[new_y][new_x]
            return newState.reward
        else:
            return self.rewardOutOfGrid

    def ProcessNextAction(self, action):
        reward = self.CheckNextAction(action)
        self.MakeAction(action)
        return reward, self.activeState

# AGENT CONFIG
rewardForFinish = 10
rewardForValidMove = -1
rewardForInvalidMove = -5
gridWidth = 3
gridHeight = 3
gridStates = []

goal_x, goal_y = gridWidth - 1, gridHeight - 1

# Create grid
for y in range(gridHeight):
    row = []
    for x in range(gridWidth):
        actions = [0, 1, 2, 3]
        if x == 0: 
            actions.remove(0)  # no left
        if x == gridWidth - 1:
            actions.remove(1)  # no right
        if y == 0:
            actions.remove(2)  # no up
        if y == gridHeight - 1:
            actions.remove(3) # no down

        if x == goal_x and y == goal_y: # goal state
            actions = []  
            reward = rewardForFinish
        else:
            reward = rewardForValidMove

        row.append(State(reward, actions, [x, y]))
    gridStates.append(row)

agent = Agent(rewardForInvalidMove, gridStates)

# --- Hyperparameters ---
alpha = 0.1      # learning rate
gamma = 0.9      # increased discount factor
EPISODES = 50000  # number of episodes
max_steps = 50    # increased step limit

# --- Initialize Q-table (y, x, action) ---
Q = np.zeros((gridHeight, gridWidth, len(agent.actionOptions)))

# Helper to get allowed actions
def get_allowed_actions(state):
    x, y = state
    state_obj = agent.grid[y][x]
    return state_obj.possibleActions

# Greedy action with allowed actions
def greedy_action(state):
    allowed_actions = get_allowed_actions(state)
    if not allowed_actions:  # No actions in terminal state
        return None
    x, y = state
    q_values = Q[y, x, allowed_actions]
    max_val = np.max(q_values)
    best_actions = [a for a in allowed_actions if Q[y, x, a] == max_val]
    return random.choice(best_actions)
numOfReturns = 0
# TRAINING LOOP
for episode in range(EPISODES):
    agent.activeState = [0, 0]
    agent.activeReward = 0
    state = [0, 0]
    action = greedy_action(state)

    for step in range(max_steps):
        reward, next_state = agent.ProcessNextAction(action)
        x_prev, y_prev = state
        nx, ny = next_state

        # Terminal state check (goal reached)
        if next_state == [goal_x, goal_y]:
            Q[y_prev, x_prev, action] += alpha * (reward - Q[y_prev, x_prev, action])
            numOfReturns += 1
            break
        
        # Choose next action (considering allowed actions)
        next_action = greedy_action(next_state)
        if next_action is None:
            break
            
        # SARSA Update (fixed indexing: [y,x] not [x,y])
        current_q = Q[y_prev, x_prev, action]
        next_q = Q[ny, nx, next_action]
        td_target = reward + gamma * next_q
        Q[y_prev, x_prev, action] += alpha * (td_target - current_q)

        # Transition to next state
        state = next_state
        action = next_action

    # Progress tracking
    if (episode + 1) % 5000 == 0:
        print(f"Episode {episode + 1}/{EPISODES}")

print("Training finished!")

# Print Q-table function (unchanged)
def print_q_table(Q):
    print(f"{'State':>8} | {'Left':>7} {'Right':>7} {'Up':>7} {'Down':>7}")
    print("-" * 42)
    for y in range(gridHeight):
        for x in range(gridWidth):
            values = Q[y, x]
            print(f"({x},{y})   | {values[0]:7.2f} {values[1]:7.2f} {values[2]:7.2f} {values[3]:7.2f}")
        print("-" * 42)

print_q_table(Q)