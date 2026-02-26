# NonRL_Visualizer — Handles visualization of non-RL pathfinding algorithms

import pygame


# Manages the visualization of non-RL algorithm execution
class NonRL_Visualizer:
    
    def __init__(self, maze, algorithm_solver, agent_img):
        self.maze = maze
        self.algorithm_solver = algorithm_solver
        self.agent_img = agent_img
        self.current_step = 0
        self.path_history = []
        self.is_running = False
        self.is_finished = False
        self.solver_generator = None
        
    # Start the algorithm execution
    def start(self):
        if not self.is_running:
            self.solver_generator = self.algorithm_solver.solve()
            self.is_running = True
            self.is_finished = False
            self.current_step = 0
            self.path_history = [self.maze.start_pos[:]]
    
    # Stop the algorithm execution
    def stop(self):
        self.is_running = False
        self.solver_generator = None
    
    # Reset the visualization
    def reset(self):
        self.stop()
        self.current_step = 0
        self.path_history = []
        self.is_finished = False
    
    # Execute one step of the algorithm
    def step(self):
        if self.is_running and not self.is_finished and self.solver_generator:
            try:
                current_pos = next(self.solver_generator)
                self.path_history.append(current_pos[:])
                self.current_step += 1
                return current_pos
            except StopIteration:
                self.is_finished = True
                self.is_running = False
                return None
        return None
    
    # Get the current position of the agent
    def get_current_position(self):
        if self.path_history:
            return self.path_history[-1]
        return self.maze.start_pos
    
    # Draw the agent at current position
    def draw_agent(self, screen, rect_size, h_margin, v_margin):
        if not self.path_history:
            return
        
        current_pos = self.path_history[-1]
        
        # Calculate position using maze's compute_layout
        rect_size_final, gx, gy, grid_w, grid_h = self.maze.compute_layout(
            rect_size, h_margin, v_margin
        )
        
        # Calculate agent position
        agent_x = gx + current_pos[0] * rect_size_final
        agent_y = gy + current_pos[1] * rect_size_final
        
        # Scale and draw agent image
        scaled_img = pygame.transform.smoothscale(
            self.agent_img,
            (rect_size_final, rect_size_final)
        )
        screen.blit(scaled_img, (agent_x, agent_y))
    
    # Draw the path taken so far
    def draw_path_history(self, screen, rect_size, h_margin, v_margin):
        if len(self.path_history) < 2:
            return
        
        rect_size_final, gx, gy, grid_w, grid_h = self.maze.compute_layout(
            rect_size, h_margin, v_margin
        )
        
        # Draw path as colored squares
        for pos in self.path_history:
            x = gx + pos[0] * rect_size_final
            y = gy + pos[1] * rect_size_final
            
            # Semi-transparent blue for visited cells
            s = pygame.Surface((rect_size_final, rect_size_final))
            s.set_alpha(50)
            s.fill((100, 150, 255))
            screen.blit(s, (x, y))
    
    # Get current statistics
    def get_stats(self):
        return {
            "steps_taken": self.current_step,
            "path_length": len(self.path_history),
            "nodes_explored": self.algorithm_solver.nodes_explored,
            "is_finished": self.is_finished,
            "success": self.algorithm_solver.success,
            "execution_time": self.algorithm_solver.execution_time
        }
