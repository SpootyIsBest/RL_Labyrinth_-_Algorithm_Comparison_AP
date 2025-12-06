import pygame

pygame.init()

# --- Grid size ---
gridHeight = 5
gridWidth = 9

# --- Window setup ---
SCREEN_SIZE = 600
SCREEN_WIDTH = SCREEN_SIZE
SCREEN_HEIGHT = SCREEN_SIZE
CAP_NAME = 'RL - visualization'
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(CAP_NAME)

clock = pygame.time.Clock()
FPS = 3

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# --- Tile size calculation ---
tile_width = SCREEN_WIDTH // gridWidth
tile_height = SCREEN_HEIGHT // gridHeight
tile_size = min(tile_width, tile_height)

# --- Center offsets (to center grid) ---
grid_pixel_width = gridWidth * tile_size
grid_pixel_height = gridHeight * tile_size
offset_x = (SCREEN_WIDTH - grid_pixel_width) // 2
offset_y = (SCREEN_HEIGHT - grid_pixel_height) // 2

# --- Player size scale ---
PLAYER_SCALE = 0.6  # 0.9 -> 10% smaller than cell
player_size = int(tile_size * PLAYER_SCALE)
offset_inside_cell = (tile_size - player_size) // 2
player = pygame.Rect(0, 0, player_size, player_size)

# --- Initial position in grid ---
player_grid_pos = [0, 0]  # (x, y)

def DrawGrid():
    SCREEN.fill(BLACK)
    # Vertical lines (including borders)
    for x in range(0, grid_pixel_width + 1, tile_size):
        pygame.draw.line(SCREEN, WHITE,
                         (offset_x + x, offset_y),
                         (offset_x + x, offset_y + grid_pixel_height))
    # Horizontal lines (including borders)
    for y in range(0, grid_pixel_height + 1, tile_size):
        pygame.draw.line(SCREEN, WHITE,
                         (offset_x, offset_y + y),
                         (offset_x + grid_pixel_width, offset_y + y))

def MovePlayerToGridPosition(grid_pos):
    grid_x, grid_y = grid_pos
    # Center inside cell
    top_left_x = offset_x + grid_x * tile_size + offset_inside_cell
    top_left_y = offset_y + grid_y * tile_size + offset_inside_cell
    player.topleft = (top_left_x, top_left_y)
    pygame.draw.rect(SCREEN, GREEN, player)

running = True
while running:
    clock.tick(FPS)

    DrawGrid()
    MovePlayerToGridPosition(player_grid_pos)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("Space pressed!")

pygame.quit()