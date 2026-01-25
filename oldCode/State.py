class State:
    def __init__(self, reward, possibleActions, position):
        self.reward = reward
        self.possibleActions = possibleActions
        self.position = position