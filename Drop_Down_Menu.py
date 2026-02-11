import pygame
import os

class Drop_Down_Menu:
    def __init__(self,
                 screen,
                 folder_path="MazeLayouts",
                 on_select_callback=None,
                 button_font=None,
                 width=300,
                 height=40,
                 pos=(0, 0),
                 max_visible_items=8,
                 border_radius=10,
                 bg_color=pygame.Color(255, 255, 255, 255),
                 text_color=pygame.Color(0, 0, 0, 255),
                 selected_color=pygame.Color(100, 150, 255, 255),
                 hover_color=pygame.Color(200, 220, 255, 255),
                 outline_color=pygame.Color(0, 0, 0, 255)):
        """
        Drop-down menu to display and select JSON files from a folder.
        
        Args:
            screen: Pygame screen surface
            folder_path: Path to folder containing JSON files
            on_select_callback: Function called when item is selected, receives filename
            button_font: Pygame font for text
            width: Width of the dropdown
            height: Height of each item
            pos: (x, y) position of dropdown
            max_visible_items: Maximum items visible before scrolling
            border_radius: Radius for rounded corners
            bg_color: Background color
            text_color: Text color
            selected_color: Color of selected item
            hover_color: Color when hovering over item
            outline_color: Border color
        """
        self.screen = screen
        self.folder_path = folder_path
        self.on_select_callback = on_select_callback
        self.button_font = button_font
        self.width = width
        self.height = height
        self.pos = pos
        self.max_visible_items = max_visible_items
        self.border_radius = border_radius
        
        # Colors
        self.bg_color = bg_color
        self.text_color = text_color
        self.selected_color = selected_color
        self.hover_color = hover_color
        self.outline_color = outline_color
        
        # State
        self.is_expanded = False
        self.selected_index = 0
        self.scroll_offset = 0
        self.hover_index = -1
        
        # Items
        self.items = []
        self.selected_item = None
        self.load_json_files()
        
        # Rects
        self.main_rect = pygame.Rect(pos, (width, height))
        self.arrow_rect = pygame.Rect(pos[0] + width - height, pos[1], height, height)
        
    def load_json_files(self):
        """Load all JSON files from the specified folder"""
        self.items = []
        
        # Check if folder exists
        if not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.folder_path, exist_ok=True)
                print(f"Created folder: {self.folder_path}")
            except Exception as e:
                print(f"Could not create folder {self.folder_path}: {e}")
                self.items = ["[No folder found]"]
                return
        
        # Get all JSON files
        try:
            files = [f for f in os.listdir(self.folder_path) if f.endswith('.json')]
            if files:
                self.items = sorted(files)
                self.selected_item = self.items[0] if self.items else None
            else:
                self.items = ["[No JSON files found]"]
                self.selected_item = None
        except Exception as e:
            print(f"Error reading folder {self.folder_path}: {e}")
            self.items = ["[Error reading folder]"]
            self.selected_item = None
    
    def get_selected_file(self):
        """Get the currently selected filename (without special messages)"""
        if self.selected_item and not self.selected_item.startswith('['):
            return self.selected_item
        return None
    
    def get_selected_path(self):
        """Get the full path to the selected file"""
        filename = self.get_selected_file()
        if filename:
            return os.path.join(self.folder_path, filename)
        return None
    
    def draw_arrow(self, rect, down=True):
        """Draw a small triangle arrow"""
        center_x = rect.centerx
        center_y = rect.centery
        size = 6
        
        if down:
            # Down arrow
            points = [
                (center_x, center_y + size // 2),
                (center_x - size, center_y - size // 2),
                (center_x + size, center_y - size // 2)
            ]
        else:
            # Up arrow
            points = [
                (center_x, center_y - size // 2),
                (center_x - size, center_y + size // 2),
                (center_x + size, center_y + size // 2)
            ]
        
        pygame.draw.polygon(self.screen, self.text_color, points)
    
    def draw(self):
        """Draw the dropdown menu"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw main button (collapsed view)
        pygame.draw.rect(self.screen, self.outline_color, self.main_rect.inflate(4, 4), border_radius=self.border_radius)
        pygame.draw.rect(self.screen, self.bg_color, self.main_rect, border_radius=self.border_radius)
        
        # Draw selected item text
        display_text = self.selected_item if self.selected_item else "Select a file..."
        # Truncate if too long
        max_chars = int((self.width - self.height - 10) / 8)  # Approximate character width
        if len(display_text) > max_chars:
            display_text = display_text[:max_chars-3] + "..."
        
        if self.button_font:
            text_surf = self.button_font.render(display_text, True, self.text_color)
            text_rect = text_surf.get_rect(midleft=(self.pos[0] + 10, self.main_rect.centery))
            self.screen.blit(text_surf, text_rect)
        
        # Draw arrow
        self.draw_arrow(self.arrow_rect, down=not self.is_expanded)
        
        # Draw expanded menu
        if self.is_expanded:
            self.draw_expanded_menu(mouse_pos)
    
    def draw_expanded_menu(self, mouse_pos):
        """Draw the expanded dropdown list"""
        if not self.items:
            return
        
        # Calculate visible items
        visible_count = min(len(self.items), self.max_visible_items)
        dropdown_height = visible_count * self.height
        dropdown_rect = pygame.Rect(self.pos[0], self.pos[1] + self.height, self.width, dropdown_height)
        
        # Draw background and border
        pygame.draw.rect(self.screen, self.outline_color, dropdown_rect.inflate(4, 4), border_radius=self.border_radius)
        pygame.draw.rect(self.screen, self.bg_color, dropdown_rect, border_radius=self.border_radius)
        
        # Draw scrollbar if needed
        if len(self.items) > self.max_visible_items:
            self.draw_scrollbar(dropdown_rect)
        
        # Draw items
        self.hover_index = -1
        for i in range(visible_count):
            item_index = i + self.scroll_offset
            if item_index >= len(self.items):
                break
            
            item_rect = pygame.Rect(
                self.pos[0] + 2,
                self.pos[1] + self.height + i * self.height + 2,
                self.width - 4,
                self.height - 2
            )
            
            # Check hover
            is_hover = item_rect.collidepoint(mouse_pos)
            if is_hover:
                self.hover_index = item_index
            
            # Draw item background
            if item_index == self.selected_index:
                pygame.draw.rect(self.screen, self.selected_color, item_rect, border_radius=self.border_radius - 2)
            elif is_hover:
                pygame.draw.rect(self.screen, self.hover_color, item_rect, border_radius=self.border_radius - 2)
            
            # Draw item text
            item_text = self.items[item_index]
            # Truncate if too long
            max_chars = int((self.width - 20) / 8)
            if len(item_text) > max_chars:
                item_text = item_text[:max_chars-3] + "..."
            
            if self.button_font:
                text_surf = self.button_font.render(item_text, True, self.text_color)
                text_rect = text_surf.get_rect(midleft=(item_rect.left + 10, item_rect.centery))
                self.screen.blit(text_surf, text_rect)
    
    def draw_scrollbar(self, dropdown_rect):
        """Draw scrollbar for long lists"""
        scrollbar_width = 8
        scrollbar_x = dropdown_rect.right - scrollbar_width - 4
        
        # Scrollbar background
        scrollbar_bg_rect = pygame.Rect(scrollbar_x, dropdown_rect.top + 4, scrollbar_width, dropdown_rect.height - 8)
        pygame.draw.rect(self.screen, pygame.Color(200, 200, 200), scrollbar_bg_rect, border_radius=4)
        
        # Scrollbar handle
        total_items = len(self.items)
        handle_height = max(20, (self.max_visible_items / total_items) * scrollbar_bg_rect.height)
        handle_y = scrollbar_bg_rect.top + (self.scroll_offset / total_items) * scrollbar_bg_rect.height
        
        handle_rect = pygame.Rect(scrollbar_x, handle_y, scrollbar_width, handle_height)
        pygame.draw.rect(self.screen, pygame.Color(100, 100, 100), handle_rect, border_radius=4)
    
    def handle_event(self, event):
        """
        Handle pygame events for the dropdown.
        Returns True if an item was selected, False otherwise.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                
                # Check if clicking main button
                if self.main_rect.collidepoint(mouse_pos):
                    self.is_expanded = not self.is_expanded
                    return False
                
                # Check if clicking an item in expanded menu
                if self.is_expanded and self.hover_index >= 0:
                    # Only allow selection of valid items (not error messages)
                    if not self.items[self.hover_index].startswith('['):
                        self.selected_index = self.hover_index
                        self.selected_item = self.items[self.hover_index]
                        self.is_expanded = False
                        
                        # Call callback
                        if self.on_select_callback:
                            self.on_select_callback(self.selected_item)
                        
                        return True
                    self.is_expanded = False
                    return False
                
                # Click outside - collapse
                if self.is_expanded:
                    self.is_expanded = False
                    return False
        
        elif event.type == pygame.MOUSEWHEEL and self.is_expanded:
            # Scroll through items
            if len(self.items) > self.max_visible_items:
                self.scroll_offset = max(0, min(
                    len(self.items) - self.max_visible_items,
                    self.scroll_offset - event.y
                ))
            return False
        
        return False
    
    def refresh(self):
        """Reload the list of JSON files"""
        old_selected = self.selected_item
        self.load_json_files()
        
        # Try to keep the same selection if it still exists
        if old_selected and old_selected in self.items:
            self.selected_index = self.items.index(old_selected)
            self.selected_item = old_selected
        else:
            self.selected_index = 0
            self.selected_item = self.items[0] if self.items else None
