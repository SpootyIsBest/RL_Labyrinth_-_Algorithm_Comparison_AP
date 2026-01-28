import random

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
            print(f"Moved to {self.activeState}\nNew active reward: {self.activeReward}")
        else:
            self.activeReward += self.rewardOutOfGrid
            print(f"Action not allowed in this state stayed on: {self.activeState}!\nActive reward: {self.activeReward}")
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

    # def ProcessNextAction(self):
    #     x, y = self.activeState 
    #     allReward = []
    #     for i in range(len(self.actionOptions)):
    #         allReward.append(self.CheckNextAction(i))
    #     bestNewReward, action = self.random_max_with_index(allReward)
    #     print(f"Next best reward:{bestNewReward}\n On action {action}")
    #     self.MakeAction(action)
    def ProcessNextAction(self, action):
        reward = self.CheckNextAction(action)
        self.MakeAction(action)
        return reward, self.activeState