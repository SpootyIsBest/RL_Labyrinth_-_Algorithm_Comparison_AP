import numpy as np
import pygame
import random
from Agent import Agent
from State import State
import math




# ------ AGENT ------
# --- Rewards ---
rewardForFinish = 50
rewardForValidMove = -1
rewardForInvalidMove = -10
rewardForMarekMove = -30
# --- Grid ---
gridWidth = 20
gridHeight = 15
gridStates = []
# --- Initial position ---
initialPosition = [0,gridHeight -1]
# --- Finish position ---
goal_x, goal_y = gridWidth - 1, 0
# ------ HYPERPARAMETERS ------
alpha = 0.72      # learning rate
gamma = 1     # discount factor
epsilon = 0.4    # exploration rate
mi = 0.1
beta = 0.25
pi = math.pi
EPISODES = 100  # number of episodes
max_steps = 500  # limit steps per episode
# --- Text Information ---
firstFind = False 
firstEpisode = -1
numOfReturns = 0
# --- Grid Creation ---
for y in range(gridHeight):
    row = []
    for x in range(gridWidth):
        actions = [0, 1, 2, 3] # left right up down
        if x == 0: 
            actions.remove(0)  # no left
        if x == gridWidth - 1:
            actions.remove(1)  # no right
        if y == 0:
            actions.remove(2)  # no up
        if y == gridHeight - 1:
            actions.remove(3) # no down

        # --- Barriers ---
        if x == 2 and y >= 1:
            actions.remove(1)
        if x == 4 and y >= 1:
            actions.remove(0)
        if x == 3 and y == 0:
            actions.remove(3)
        
        if x == 7 and y <= 3:
            actions.remove(1)
        if x == 9 and y <= 3:
            actions.remove(0)
        if x == 8 and y == 4:
            actions.remove(2)
        
        # --- Set the Goal and its reward ---
        if x == goal_x and y == goal_y: # goal state
            actions = []  
            reward = rewardForFinish
        else:
            # for every other move default reward
            reward = rewardForValidMove


        row.append(State(reward, actions, [x, y]))
    gridStates.append(row)



# --- Initialize Agent ---
agent = Agent(rewardForInvalidMove, gridStates)

# --- Initialize Q-table (y, x, action) ---
Q = np.zeros((gridHeight, gridWidth, len(agent.actionOptions)))

# Q[state, action] = Q[state, action] + α * (reward + γ * max(Q[next_state]) - Q[state, action])
# Q[S, A] = Q[S, A] + α * (R + γ * Q[S', A'] - Q[S, A])


# --- Which next action to take ---
def epsilon_greedy_action(state):
    x, y = state
    global epsilon
    if random.uniform(0, 1) < epsilon:
        return random.choice(agent.actionOptions)  # Explore
    else:
        return np.argmax(Q[y, x])  # Exploit


# --- Learning loop ---
for episode in range(EPISODES):
    # Reset agent state
    agent.activeState = initialPosition
    agent.activeReward = 0

    # assign first state and first action
    state = agent.activeState
    action = epsilon_greedy_action(state)

    for i in range(max_steps):
        print(f"\n Step {i}")
        
        reward, next_state = agent.ProcessNextAction(action)

        next_action = epsilon_greedy_action(next_state)

        x, y = state
        nx, ny = next_state

        # SARSA algorithm
        Q[y,x, action] = Q[y,x, action] + alpha * (reward + gamma*Q[ny,nx, next_action]- Q[y,x, action] - (i*mi)) # (mi * ((1 + i) ** -2))

        state, action = next_state, next_action

        epsilon = abs(math.sin(episode * pi * beta))/2

        if agent.activeState == [goal_x, goal_y]:
            numOfReturns += 1
            print(f"Maze finished for {numOfReturns} number of times!")
            if alpha > 0.1:
                alpha -= 0.0045
            if firstFind == False:
                firstFind = True
                firstEpisode = episode
            break
        

    print(f"Episode {episode+1} finished with reward: {agent.activeReward}")
print(f"Training finished!\n NUMBER OF FINISHES {numOfReturns} \nEPSILON: {epsilon} \nALPHA: {alpha} \nFIRST EPISODE: {firstEpisode}")


def print_q_table(Q):
    # Header
    print(f"{'State':>8} | {'Left':>7} {'Right':>7} {'Up':>7} {'Down':>7}")
    print("-" * 42)
    for y in range(gridHeight):
        for x in range(gridWidth):
            values = Q[y, x]
            print(f"({x},{y})   | {values[0]:7.2f} {values[1]:7.2f} {values[2]:7.2f} {values[3]:7.2f}")
        print("-" * 42)
print_q_table(Q)

def print_optimal_path_visual(Q, start, goal):
    state = start[:]
    path = [tuple(state)]
    visited = set()
    max_steps = gridWidth * gridHeight * 2  # safety

    while tuple(state) != tuple(goal) and len(path) < max_steps:
        x, y = state

        # Safety check
        if not (0 <= x < gridWidth and 0 <= y < gridHeight):
            print(f"State {state} out of bounds. Stopping.")
            break

        action = np.argmax(Q[y, x])
        if action == 0:   state = [x - 1, y]
        elif action == 1: state = [x + 1, y]
        elif action == 2: state = [x, y - 1]
        elif action == 3: state = [x, y + 1]

        if tuple(state) in visited:
            print("Loop detected! Path might be invalid.")
            break
        visited.add(tuple(state))
        path.append(tuple(state))

    print("Optimal path:", path)

    # Build grid representation
    grid_repr = [[" . " for _ in range(gridWidth)] for _ in range(gridHeight)]
    for (x, y) in path[1:-1]:
        if 0 <= x < gridWidth and 0 <= y < gridHeight:
            grid_repr[y][x] = " * "
    sx, sy = start
    gx, gy = goal
    grid_repr[sy][sx] = " S "
    grid_repr[gy][gx] = " G "

    print("\nGrid with path:")
    for y in range(gridHeight):
        print("".join(grid_repr[y]))

print_optimal_path_visual(Q, start=initialPosition, goal=[goal_x, goal_y])

