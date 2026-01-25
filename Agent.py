import random

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
    
    def _greedy_action_from_Q(self, state, Q):
        """
        Pick the best action in 'state' according to Q, respecting allowed actions.
        Returns None if there are no allowed actions (terminal state).
        """
        x, y = state
        allowed = self.grid[y][x].actions  # e.g. [0,1,3] etc.

        if not allowed:
            return None

        q_row = Q[y, x]  # shape (4,)
        best_value = max(q_row[a] for a in allowed)
        best_actions = [a for a in allowed if q_row[a] == best_value]

        # tie-break randomly between equally good actions
        return random.choice(best_actions)

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