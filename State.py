class State:
    def __init__(self, reward, actions, pos):
        self.reward = reward          # reward for entering this cell
        self.actions = actions[:]     # allowed actions from this cell
        self.pos = pos[:]             # [x, y]