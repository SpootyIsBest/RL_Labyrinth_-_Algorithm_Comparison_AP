
import pygame


class InputBox:

    def __init__(self, x, y, w, h,COLOR_INACTIVE, COLOR_ACTIVE, COLOR_SAVED, FONT, text='', variable=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = COLOR_INACTIVE
        self.color_active = COLOR_ACTIVE
        self.color = self.color_inactive
        self.color_saved = COLOR_SAVED
        self.font = FONT
        self.text = text
        
        self.active = False

        self.variable = variable

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.variable = self.updateVariable()
                    self.color = self.color_saved
                    return True
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        self.txt_surface = self.font.render(self.text, True, self.color)
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width
    def updateVariable(self):
        try:
            value = int(self.text)
            return value
        except ValueError:
            try:
                value = float(self.text)
                return value
            except ValueError:
                return None

    def draw(self, screen):
        # Blit the text.
        self.txt_surface = self.font.render(self.text, True, self.color)
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)