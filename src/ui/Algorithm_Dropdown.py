import pygame

class Algorithm_Dropdown:
    def __init__(self,
                 screen,
                 algorithms=None,
                 on_select_callback=None,
                 button_font=None,
                 width=300,
                 height=40,
                 pos=(0, 0),
                 bg_color=pygame.Color(255, 255, 255, 255),
                 text_color=pygame.Color(0, 0, 0, 255),
                 selected_color=pygame.Color(100, 150, 255, 255),
                 hover_color=pygame.Color(200, 220, 255, 255),
                 outline_color=pygame.Color(0, 0, 0, 255)):
        # Drop-down menu to select RL algorithms
        self.screen = screen
        self.algorithms = algorithms if algorithms else ["Q-Learning", "SARSA"]
        self.on_select_callback = on_select_callback
        self.button_font = button_font
        self.width = width
        self.height = height
        self.pos = pos
        
        # Colors
        self.bg_color = bg_color
        self.text_color = text_color
        self.selected_color = selected_color
        self.hover_color = hover_color
        self.outline_color = outline_color
        
        # State
        self.is_open = False
        self.selected_index = 0  # Default to first algorithm
        self.hover_index = -1
        
        # Rectangles for rendering
        self.main_rect = pygame.Rect(pos[0], pos[1], width, height)
        self.dropdown_rect = pygame.Rect(pos[0], pos[1] + height, width, height * len(self.algorithms))
    
    # Returns the currently selected algorithm name
    def get_selected_algorithm(self):
        return self.algorithms[self.selected_index]
    
    # Set the selected algorithm by name
    def set_selected_algorithm(self, algorithm_name):
        if algorithm_name in self.algorithms:
            self.selected_index = self.algorithms.index(algorithm_name)
    
    # Handle pygame events for the dropdown
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            
            # Check if clicking main button
            if self.main_rect.collidepoint(mouse_pos):
                self.is_open = not self.is_open
                return True
            
            # Check if clicking dropdown items when open
            if self.is_open:
                for i in range(len(self.algorithms)):
                    item_rect = pygame.Rect(
                        self.pos[0],
                        self.pos[1] + self.height + i * self.height,
                        self.width,
                        self.height
                    )
                    if item_rect.collidepoint(mouse_pos):
                        self.selected_index = i
                        self.is_open = False
                        if self.on_select_callback:
                            self.on_select_callback(self.algorithms[i])
                        return True
                
                # Clicked outside dropdown when open - close it
                self.is_open = False
        
        elif event.type == pygame.MOUSEMOTION and self.is_open:
            mouse_pos = event.pos
            self.hover_index = -1
            for i in range(len(self.algorithms)):
                item_rect = pygame.Rect(
                    self.pos[0],
                    self.pos[1] + self.height + i * self.height,
                    self.width,
                    self.height
                )
                if item_rect.collidepoint(mouse_pos):
                    self.hover_index = i
                    break
        
        return False
    
    # Draw the dropdown menu
    def draw(self):
        # Draw main button
        pygame.draw.rect(self.screen, self.bg_color, self.main_rect)
        pygame.draw.rect(self.screen, self.outline_color, self.main_rect, 2)
        
        # Draw selected algorithm text
        if self.button_font:
            text_surf = self.button_font.render(self.algorithms[self.selected_index], True, self.text_color)
            text_rect = text_surf.get_rect(center=self.main_rect.center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw arrow indicator
        arrow_x = self.main_rect.right - 25
        arrow_y = self.main_rect.centery
        if self.is_open:
            # Up arrow
            pygame.draw.polygon(self.screen, self.text_color, [
                (arrow_x, arrow_y + 5),
                (arrow_x + 10, arrow_y + 5),
                (arrow_x + 5, arrow_y - 5)
            ])
        else:
            # Down arrow
            pygame.draw.polygon(self.screen, self.text_color, [
                (arrow_x, arrow_y - 5),
                (arrow_x + 10, arrow_y - 5),
                (arrow_x + 5, arrow_y + 5)
            ])
        
        # Draw dropdown items when open
        if self.is_open:
            for i, algorithm in enumerate(self.algorithms):
                item_rect = pygame.Rect(
                    self.pos[0],
                    self.pos[1] + self.height + i * self.height,
                    self.width,
                    self.height
                )
                
                # Determine background color
                if i == self.selected_index:
                    color = self.selected_color
                elif i == self.hover_index:
                    color = self.hover_color
                else:
                    color = self.bg_color
                
                # Draw item background
                pygame.draw.rect(self.screen, color, item_rect)
                pygame.draw.rect(self.screen, self.outline_color, item_rect, 2)
                
                # Draw item text
                if self.button_font:
                    text_surf = self.button_font.render(algorithm, True, self.text_color)
                    text_rect = text_surf.get_rect(center=item_rect.center)
                    self.screen.blit(text_surf, text_rect)
    
    # Update the position of the dropdown (useful for screen resize)
    def update_position(self, new_pos):
        self.pos = new_pos
        self.main_rect = pygame.Rect(new_pos[0], new_pos[1], self.width, self.height)
        self.dropdown_rect = pygame.Rect(new_pos[0], new_pos[1] + self.height, self.width, self.height * len(self.algorithms))
    
    # Update the size of the dropdown (useful for screen resize)
    def update_size(self, new_width, new_height):
        self.width = new_width
        self.height = new_height
        self.main_rect = pygame.Rect(self.pos[0], self.pos[1], new_width, new_height)
        self.dropdown_rect = pygame.Rect(self.pos[0], self.pos[1] + new_height, new_width, new_height * len(self.algorithms))
