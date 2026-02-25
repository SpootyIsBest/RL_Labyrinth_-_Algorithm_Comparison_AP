"""
Non-RL pathfinding algorithms for maze solving.
Includes: BFS, Wall Follower, Random Walk, and Greedy.
"""

import random
from collections import deque
import time


class NonRL_Algorithm:
    """Base class for non-RL pathfinding algorithms"""
    
    def __init__(self, maze, start_pos, goal_pos):
        self.maze = maze
        self.start_pos = start_pos[:]
        self.goal_pos = goal_pos[:]
        self.path = []
        self.nodes_explored = 0
        self.execution_time = 0
        self.success = False
        
    def solve(self):
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions based on maze walls"""
        x, y = pos
        neighbors = []
        actions = self.maze.gridStates[y][x].actions
        
        # actions: 0=left, 1=right, 2=up, 3=down
        if 0 in actions:  # left
            neighbors.append((x - 1, y))
        if 1 in actions:  # right
            neighbors.append((x + 1, y))
        if 2 in actions:  # up
            neighbors.append((x, y - 1))
        if 3 in actions:  # down
            neighbors.append((x, y + 1))
        
        return neighbors
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class BFS_Algorithm(NonRL_Algorithm):
    """Breadth-First Search - guarantees shortest path"""
    
    def solve(self):
        start_time = time.time()
        
        queue = deque([(self.start_pos, [self.start_pos])])
        visited = {tuple(self.start_pos)}
        
        while queue:
            current_pos, path = queue.popleft()
            self.nodes_explored += 1
            
            # Check if goal reached
            if current_pos[0] == self.goal_pos[0] and current_pos[1] == self.goal_pos[1]:
                self.path = path
                self.success = True
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current_pos):
                if tuple(neighbor) not in visited:
                    visited.add(tuple(neighbor))
                    queue.append((neighbor, path + [neighbor]))
            
            # Yield for visualization
            yield current_pos
        
        self.execution_time = time.time() - start_time
        self.success = False
        return []


class WallFollower_Algorithm(NonRL_Algorithm):
    """Wall Follower (Right-Hand Rule) - follows right wall"""
    
    def __init__(self, maze, start_pos, goal_pos):
        super().__init__(maze, start_pos, goal_pos)
        # Directions: 0=North, 1=East, 2=South, 3=West
        self.direction = 1  # Start facing East
        self.max_steps = maze.maze_size_width * maze.maze_size_height * 4  # Prevent infinite loops
    
    def solve(self):
        start_time = time.time()
        
        current_pos = self.start_pos[:]
        path = [current_pos[:]]
        steps = 0
        
        while steps < self.max_steps:
            self.nodes_explored += 1
            steps += 1
            
            # Check if goal reached
            if current_pos[0] == self.goal_pos[0] and current_pos[1] == self.goal_pos[1]:
                self.path = path
                self.success = True
                self.execution_time = time.time() - start_time
                return path
            
            # Try to turn right and move
            right_dir = (self.direction + 1) % 4
            if self.can_move(current_pos, right_dir):
                self.direction = right_dir
                current_pos = self.move(current_pos, self.direction)
            # Try to move forward
            elif self.can_move(current_pos, self.direction):
                current_pos = self.move(current_pos, self.direction)
            # Turn left
            else:
                self.direction = (self.direction - 1) % 4
                continue
            
            path.append(current_pos[:])
            yield current_pos
        
        self.execution_time = time.time() - start_time
        self.success = False
        self.path = path
        return path
    
    def can_move(self, pos, direction):
        """Check if can move in given direction"""
        x, y = pos
        actions = self.maze.gridStates[y][x].actions
        
        # Map direction to action: 0=North(up), 1=East(right), 2=South(down), 3=West(left)
        direction_to_action = {0: 2, 1: 1, 2: 3, 3: 0}  # up, right, down, left
        return direction_to_action[direction] in actions
    
    def move(self, pos, direction):
        """Move in given direction"""
        x, y = pos
        if direction == 0:  # North (up)
            return [x, y - 1]
        elif direction == 1:  # East (right)
            return [x + 1, y]
        elif direction == 2:  # South (down)
            return [x, y + 1]
        else:  # West (left)
            return [x - 1, y]


class RandomWalk_Algorithm(NonRL_Algorithm):
    """Random Walk - explores randomly until goal is found"""
    
    def __init__(self, maze, start_pos, goal_pos):
        super().__init__(maze, start_pos, goal_pos)
        self.max_steps = maze.maze_size_width * maze.maze_size_height * 100
    
    def solve(self):
        start_time = time.time()
        
        current_pos = self.start_pos[:]
        path = [current_pos[:]]
        steps = 0
        
        while steps < self.max_steps:
            self.nodes_explored += 1
            steps += 1
            
            # Check if goal reached
            if current_pos[0] == self.goal_pos[0] and current_pos[1] == self.goal_pos[1]:
                self.path = path
                self.success = True
                self.execution_time = time.time() - start_time
                return path
            
            # Get valid neighbors and choose randomly
            neighbors = self.get_neighbors(current_pos)
            if neighbors:
                current_pos = random.choice(neighbors)
                path.append(current_pos[:])
                yield current_pos
            else:
                break
        
        self.execution_time = time.time() - start_time
        self.success = False
        self.path = path
        return path


class Greedy_Algorithm(NonRL_Algorithm):
    """Greedy Best-First Search - always moves closer to goal"""
    
    def __init__(self, maze, start_pos, goal_pos):
        super().__init__(maze, start_pos, goal_pos)
        self.max_steps = maze.maze_size_width * maze.maze_size_height * 10
    
    def solve(self):
        start_time = time.time()
        
        current_pos = self.start_pos[:]
        path = [current_pos[:]]
        visited = {tuple(current_pos)}
        steps = 0
        
        while steps < self.max_steps:
            self.nodes_explored += 1
            steps += 1
            
            # Check if goal reached
            if current_pos[0] == self.goal_pos[0] and current_pos[1] == self.goal_pos[1]:
                self.path = path
                self.success = True
                self.execution_time = time.time() - start_time
                return path
            
            # Get neighbors and choose the one closest to goal
            neighbors = self.get_neighbors(current_pos)
            unvisited_neighbors = [n for n in neighbors if tuple(n) not in visited]
            
            if not unvisited_neighbors:
                # Backtrack if stuck
                if len(path) > 1:
                    path.pop()
                    current_pos = path[-1]
                    continue
                else:
                    break
            
            # Choose neighbor with minimum distance to goal
            best_neighbor = min(unvisited_neighbors, 
                              key=lambda n: self.manhattan_distance(n, self.goal_pos))
            
            current_pos = best_neighbor
            visited.add(tuple(current_pos))
            path.append(current_pos[:])
            yield current_pos
        
        self.execution_time = time.time() - start_time
        self.success = False
        self.path = path
        return path


def get_algorithm(algorithm_name, maze, start_pos, goal_pos):
    """Factory function to create algorithm instance"""
    algorithms = {
        "BFS": BFS_Algorithm,
        "Wall Follower": WallFollower_Algorithm,
        "Random Walk": RandomWalk_Algorithm,
        "Greedy": Greedy_Algorithm
    }
    
    if algorithm_name in algorithms:
        return algorithms[algorithm_name](maze, start_pos, goal_pos)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
