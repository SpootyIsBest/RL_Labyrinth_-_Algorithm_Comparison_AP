import pygame

class Button:
    def __init__(self,
                  screen,
                  text,
                  nextMonitor,
                  on_change_monitor, 
                  button_font, 
                  button_font_inflated, 
                  width, 
                  height, 
                  pos, 
                  inflation_on_press, 
                  border_radius, 
                  outline_inflation=7,
                  click_press_time=200,
                  button_background_color=pygame.Color(0,0,0,255), 
                  button_pressed_color=pygame.Color(0,0,0,255), 
                  button_color=pygame.Color(255,255,255,255),
                  text_pressed_color=pygame.Color(0,0,0,255), 
                  text_color=pygame.Color(255,255,255,255)
                  ):
        # Core attributes
        self.pressed = False
        self.click_until = 0
        self.click_press_time = click_press_time
        self.nextMonitor = nextMonitor
        self.screen = screen
        self.on_change_monitor = on_change_monitor
        # Colors
        self.button_color = button_color
        self.button_pressed_color = button_pressed_color
        self.button_background_color = button_background_color
        self.text_color = text_color
        self.text_pressed_color = text_pressed_color

        # Inflation settings
        self.inflated = False
        self.inflation = inflation_on_press
        self.border_radius = border_radius

        # Top Rect
        self.top_rect = pygame.Rect(pos, (width, height))
        self.top_color = self.button_background_color

        # Middle Rect
        self.middle_rect = pygame.Rect(pos, (width, height))
        self.middle_color = self.button_background_color

        # Bottom Rect (outline)
        self.bottom_rect = pygame.Rect(pos, (width, height)).inflate(outline_inflation,outline_inflation)
        self.bottom_color = button_color

        # Text
        self.text = text
        self.button_font = button_font
        self.button_font_inflated = button_font_inflated

    def _draw_alpha_rect(self, rect, color, radius=20):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, color, surf.get_rect(), border_radius=radius)
        self.screen.blit(surf, rect.topleft)
    def _on_click(self):
        # store the timestamp when the color should reset
        self.click_until = pygame.time.get_ticks() + self.click_press_time  # 300 ms = 0.3 s
        if self.on_change_monitor:
            self.on_change_monitor(self.nextMonitor)
 

    def draw(self):
        
        pygame.draw.rect(self.screen,self.bottom_color,self.bottom_rect, border_radius=self.border_radius)
        pygame.draw.rect(self.screen,self.middle_color,self.middle_rect, border_radius=self.border_radius)

        now = pygame.time.get_ticks()

        if self.inflated:
            font = self.button_font_inflated
            color = self.text_color
            if now < self.click_until:
                color = self.button_pressed_color  # clicked color
                self._draw_alpha_rect(self.top_rect, pygame.Color(255,255,255,30), radius=self.border_radius)
            else:
                self._draw_alpha_rect(self.top_rect, pygame.Color(255,255,255,130), radius=self.border_radius)
        else:
            pygame.draw.rect(self.screen,self.top_color,self.top_rect, border_radius=self.border_radius)
            font = self.button_font
            color = self.text_color

        text_surf = font.render(self.text, True, color)
        text_rect = text_surf.get_rect(center=self.top_rect.center)
        self.screen.blit(text_surf,text_rect)

        self.click_check()

    def click_check(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            if not self.inflated:
                self.top_rect = self.top_rect.inflate(self.inflation,self.inflation)
                self.middle_rect = self.middle_rect.inflate(self.inflation,self.inflation)
                self.bottom_rect = self.bottom_rect.inflate(self.inflation,self.inflation)
                
                self.inflated = True
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed:
                    self.pressed = False
                    self._on_click()
                    
        else:
            if self.inflated:
                
                self.top_rect = self.top_rect.inflate(-self.inflation,-self.inflation)
                self.middle_rect = self.middle_rect.inflate(-self.inflation,-self.inflation)
                self.bottom_rect = self.bottom_rect.inflate(-self.inflation,-self.inflation)
                self.inflated = False