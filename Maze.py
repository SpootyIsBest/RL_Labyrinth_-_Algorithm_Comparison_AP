import pygame
import random
import math
from State import State


class Maze:
    def __init__(self, screen, SCREEN_WIDTH, SCREEN_HEIGHT, maze_size_width, maze_size_height, origin=[0,0]):
        self.screen = screen
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.maze_size_width = maze_size_width
        self.maze_size_height = maze_size_height
        self.origin_cor = origin
        self.cell_options = ["Up", "Right", "Down", "Left", "Center"] # ["Up", "Right", "Down", "Left", "Center"] ["↑","→","↓","←","•"]
        self.maze_grid = [["" for _ in range(maze_size_height)] for _ in range(maze_size_width)]
        self.maze_walls = [[["Wall" for _ in range(4)] for _ in range(maze_size_height)] for _ in range(maze_size_width)]
        self.start_pos = [0,0]
        self.optimal_path = []
        self.gridStates = []

    def create_default(self):
        for y in range(self.maze_size_height):
            for x in range(self.maze_size_width):
                if x == self.origin_cor[0] and y == self.origin_cor[1]: self.maze_grid[x][y] = self.cell_options[-1] # Origin cell
                elif x > self.origin_cor[0]:    self.maze_grid[x][y] = self.cell_options[3] # Arrow left
                elif x < self.origin_cor[0]:    self.maze_grid[x][y] = self.cell_options[1] # Arrow Right
                elif y > self.origin_cor[1]:    self.maze_grid[x][y] = self.cell_options[0] # Arrow Down
                elif y < self.origin_cor[1]:    self.maze_grid[x][y] = self.cell_options[2] # Arrow Up
    
    def print_maze(self):
        symbols = {
            "Up": "↑",
            "Down": "↓",
            "Left": "←",
            "Right": "→",
            "Center": "●"
            }
        for y in range(self.maze_size_height):
            row = ""
            for x in range(self.maze_size_width):
                cell = self.maze_grid[x][y]
                row += f"{symbols[cell]:^4}"  # center-align symbol
            print(row)
        print("\n")

    def move_origin(self, direction):
        if direction == "Up":
            self.maze_grid[self.origin_cor[0]][self.origin_cor[1]] = self.cell_options[0]
            self.origin_cor[1] -= 1
        elif direction == "Down":
            self.maze_grid[self.origin_cor[0]][self.origin_cor[1]] = self.cell_options[2]
            self.origin_cor[1] += 1
        elif direction == "Left":
            self.maze_grid[self.origin_cor[0]][self.origin_cor[1]] = self.cell_options[3]
            self.origin_cor[0] -= 1
        elif direction == "Right":
            self.maze_grid[self.origin_cor[0]][self.origin_cor[1]] = self.cell_options[1]
            self.origin_cor[0] += 1
        
        self.maze_grid[self.origin_cor[0]][self.origin_cor[1]] = self.cell_options[-1]

    def create_grid_states(self, rewardForFinish, rewardForValidMove):
        # Clear previous states if this is called multiple times
        self.gridStates = []

        for y in range(self.maze_size_height):
            row = []
            for x in range(self.maze_size_width):
                mw = self.maze_walls[x][y]  # [top, bottom, left, right]

                actions = []

                # left (action 0) -> must not have left wall
                if mw[2] != "Wall":
                    actions.append(0)

                # right (action 1) -> must not have right wall
                if mw[3] != "Wall":
                    actions.append(1)

                # up (action 2) -> must not have top wall
                if mw[0] != "Wall":
                    actions.append(2)

                # down (action 3) -> must not have bottom wall
                if mw[1] != "Wall":
                    actions.append(3)

                # Goal cell (origin) – terminal state
                if x == self.origin_cor[0] and y == self.origin_cor[1]:
                    reward = rewardForFinish
                    actions = []  # no actions from goal
                else:
                    reward = rewardForValidMove

                row.append(State(reward, actions, [x, y]))

            self.gridStates.append(row)

    def random_sequence(self, moves_num):
        movements = ["Up", "Down", "Left", "Right"]
        for _ in range(moves_num):
            while True:
                random_move = random.choice(movements)

                if random_move == "Up"    and self.origin_cor[1] == 0:                      continue
                if random_move == "Down"  and self.origin_cor[1] == self.maze_size_height-1: continue
                if random_move == "Left"  and self.origin_cor[0] == 0:                      continue
                if random_move == "Right" and self.origin_cor[0] == self.maze_size_width-1:  continue

                break  # legal move picked
            self.move_origin(random_move)

    def draw_wall_top(self, cor_x, cor_y, rect_size, grid_wall_width=2, color=pygame.Color(0,255,0)):
        top_rect = pygame.Rect((cor_x, cor_y - grid_wall_width/2), (rect_size, grid_wall_width)) # UP
        pygame.draw.rect(self.screen, color, top_rect)

    def draw_wall_bottom(self, cor_x, cor_y, rect_size, grid_wall_width=2, color=pygame.Color(0,255,0)):
        bottom_rect = pygame.Rect((cor_x, cor_y+rect_size - grid_wall_width/2), (rect_size, grid_wall_width)) # DOWN
        pygame.draw.rect(self.screen, color, bottom_rect)
    def draw_wall_left(self, cor_x, cor_y, rect_size, grid_wall_width=2, color=pygame.Color(0,255,0)):
        left_rect = pygame.Rect((cor_x - grid_wall_width/2, cor_y), (grid_wall_width, rect_size)) # LEFT
        pygame.draw.rect(self.screen, color, left_rect)

    def draw_wall_right(self, cor_x,cor_y,rect_size,grid_wall_width=2, color=pygame.Color(0,255,0)):
        right_rect = pygame.Rect((cor_x+rect_size - grid_wall_width/2, cor_y), (grid_wall_width, rect_size))  # RIGHT
        pygame.draw.rect(self.screen, color, right_rect)

    def compute_layout(self, rect_size, horizontal_margin, vertical_margin):
        avail_w = self.SCREEN_WIDTH  - (2 * horizontal_margin)
        avail_h = self.SCREEN_HEIGHT - (2 * vertical_margin)

        cell_by_avali_w = avail_w / self.maze_size_width
        cell_by_avali_h = avail_h / self.maze_size_height
        rect_size_final  = int(min(rect_size, cell_by_avali_h, cell_by_avali_w))

        grid_w = int(round(self.maze_size_width  * rect_size_final))
        grid_h = int(round(self.maze_size_height * rect_size_final))

        gx = int(round(horizontal_margin + (avail_w - grid_w) / 2))
        gy = int(round(vertical_margin   + (avail_h - grid_h) / 2))

        return rect_size_final, gx, gy, grid_w, grid_h
    
    def draw_maze(self, rect_size, horizontal_margin, vertical_margin,
                  draw_arrows=True, draw_walls=True):

        rect_size, gx, gy, grid_w, grid_h = self.compute_layout(
            rect_size, horizontal_margin, vertical_margin
        )

        if not draw_walls:
            self.draw_rect_with_outline((gx, gy), grid_w, grid_h, 4)

        arrow_padding   = rect_size * 0.12
        grid_wall_width = 2

        for y in range(self.maze_size_height):
            for x in range(self.maze_size_width):
                cor_x = x * rect_size + gx
                cor_y = y * rect_size + gy
                arrow_direction = self.maze_grid[x][y]

                if draw_walls:
                    for i, wall in enumerate(self.maze_walls[x][y]):
                        if wall == "Wall":
                            match i:
                                case 0: self.draw_wall_top(cor_x, cor_y, rect_size)
                                case 1: self.draw_wall_bottom(cor_x, cor_y, rect_size)
                                case 2: self.draw_wall_left(cor_x, cor_y, rect_size)
                                case 3: self.draw_wall_right(cor_x, cor_y, rect_size)

                if draw_arrows:
                    match arrow_direction:
                        case "Up":
                            arrow = self.arrow_polygon(
                                (cor_x + (rect_size / 2),
                                 cor_y + rect_size - arrow_padding),
                                (cor_x + (rect_size / 2),
                                 cor_y + arrow_padding)
                            )
                        case "Down":
                            arrow = self.arrow_polygon(
                                (cor_x + (rect_size / 2),
                                 cor_y + arrow_padding),
                                (cor_x + (rect_size / 2),
                                 cor_y + rect_size - arrow_padding)
                            )
                        case "Left":
                            arrow = self.arrow_polygon(
                                (cor_x + rect_size - arrow_padding,
                                 cor_y + (rect_size / 2)),
                                (cor_x + arrow_padding,
                                 cor_y + (rect_size / 2))
                            )
                        case "Right":
                            arrow = self.arrow_polygon(
                                (cor_x + arrow_padding,
                                 cor_y + (rect_size / 2)),
                                (cor_x + rect_size - arrow_padding,
                                 cor_y + (rect_size / 2))
                            )
                        case "Center":
                            continue
                    pygame.draw.polygon(self.screen, pygame.Color("white"), arrow)

        origin_x = self.origin_cor[0] * rect_size + gx + rect_size / 4
        origin_y = self.origin_cor[1] * rect_size + gy + rect_size / 4
        origin_rect = pygame.Rect((origin_x, origin_y), (rect_size / 2, rect_size / 2))
        pygame.draw.rect(self.screen, pygame.Color(255, 0, 0), origin_rect)
    def draw_agent(self, agent, rect_size, horizontal_margin, vertical_margin, img):
        rect_size, gx, gy, grid_w, grid_h = self.compute_layout(
            rect_size, horizontal_margin, vertical_margin
        )

        ax, ay = agent.activeState  # [x, y] in grid coordinates

        cell_x = gx + ax * rect_size
        cell_y = gy + ay * rect_size

        # Center the 64x64 sprite inside the cell
        img_w, img_h = img.get_width(), img.get_height()
        draw_x = cell_x + (rect_size - img_w) // 2
        draw_y = cell_y + (rect_size - img_h) // 2

        self.screen.blit(img, (draw_x, draw_y))

    def carve_walls_from_arrows(self):
        for y in range(self.maze_size_height):
            for x in range(self.maze_size_width):
                arrow_direction = self.maze_grid[x][y]


                
                match arrow_direction:
                    case "Up":
                        self.maze_walls[x][y][0] = ""
                        if(y > 0):
                            self.maze_walls[x][y-1][1] = ""
                    case "Down":
                        self.maze_walls[x][y][1] = ""
                        if(y < self.maze_size_height - 1):
                            self.maze_walls[x][y+1][0] = ""
                    case "Left":  
                        self.maze_walls[x][y][2] = ""
                        if(x > 0):
                            self.maze_walls[x-1][y][3] = ""
                    case "Right":
                        self.maze_walls[x][y][3] = ""
                        if(x < self.maze_size_width - 1):
                            self.maze_walls[x+1][y][2] = ""
                    case "Center":
                        continue

    def arrow_polygon(self,start, end, shaft_width=4, head_len=14, head_width=12):
        x1, y1 = start
        x2, y2 = end

        dx, dy = (x2 - x1), (y2 - y1)
        L = math.hypot(dx, dy)
        if L == 0:
            return []  # nothing to draw

        # Unit direction along the arrow
        ux, uy = dx / L, dy / L
        # Perpendicular unit (left normal)
        px, py = -uy, ux

        # Where the head starts: back from the tip by head_len
        hx, hy = x2 - ux * head_len, y2 - uy * head_len

        # Half widths
        sw2 = shaft_width / 2
        hw2 = head_width  / 2

        # Shaft rectangle (4 points)
        s1 = (x1 + px * sw2, y1 + py * sw2)      # shaft tail left
        s2 = (hx + px * sw2, hy + py * sw2)      # shaft head left
        s3 = (hx - px * sw2, hy - py * sw2)      # shaft head right
        s4 = (x1 - px * sw2, y1 - py * sw2)      # shaft tail right

        # Head triangle (2 base points + tip)
        h1 = (hx + px * hw2, hy + py * hw2)      # head base left
        tip = (x2, y2)                            # tip
        h2 = (hx - px * hw2, hy - py * hw2)      # head base right

        # One continuous polygon (shaft + head)
        pts = [s1, s2, h1, tip, h2, s3, s4]

        # Round for crisp pixels
        return [(int(round(x)), int(round(y))) for (x, y) in pts]
        
    def draw_rect_with_outline(self,
                               pos,
                               w,
                               h,
                               outline_inflation,
                               top_rect_color=pygame.Color(0,0,0,255),
                               bottom_outline_color=pygame.Color(255,255,255,255)):
        top_rect = pygame.Rect(pos, (w, h))
        bottom_rect = top_rect.inflate(int(outline_inflation),int(outline_inflation))

        pygame.draw.rect(self.screen, bottom_outline_color, bottom_rect)
        pygame.draw.rect(self.screen, top_rect_color, top_rect)

    def cal_init_pos(self):
        o_cor_x = self.origin_cor[0]
        o_cor_y = self.origin_cor[1]

        if(o_cor_x < self.maze_size_width/2): # Left
            init_pos_x = self.maze_size_width -1
        else: # Right
            init_pos_x = 0
        if(o_cor_y < self.maze_size_height/2): # Top
            init_pos_y = self.maze_size_height -1
        else: # bottom
            init_pos_y = 0
        
        self.start_pos[0], self.start_pos[1] = init_pos_x, init_pos_y
    
    def create_optimal_path(self,init_pos):
        starting_pos = init_pos
        notInOrigin = True
        pos_x, pos_y = init_pos[0], init_pos[1]
        self.optimal_path = []
        # this will go with the 
        while notInOrigin:
            curr_arw = self.maze_grid[pos_x][pos_y]
            self.optimal_path.append(curr_arw)
            match curr_arw:
                case "Up":
                    pos_y -= 1
                case "Down":
                    pos_y += 1
                case "Left":
                    pos_x -= 1
                case "Right":
                    pos_x += 1
                case "Center":
                    notInOrigin = False
        # print("Most optimal way was found")
        
    def draw_optimal_path(self,horizontal_margin,vertical_margin, rect_size, starting_pos):

        opt_path = self.optimal_path
        # Calculate the available width and height
        avail_w = self.SCREEN_WIDTH - (2*horizontal_margin)
        avail_h = self.SCREEN_HEIGHT - (2*vertical_margin)
        # Calculate how many cells takes the space from the available space and use the one that takes more space (use the one that has more cells)
        cell_by_avali_w = avail_w/self.maze_size_width
        cell_by_avali_h = avail_h/self.maze_size_height
        rect_size = min(rect_size, cell_by_avali_h, cell_by_avali_w)
        rect_size = int(rect_size)
        # Calculate the space that the grid needs
        grid_w = int(round(self.maze_size_width  * rect_size))
        grid_h = int(round(self.maze_size_height * rect_size))
        # Calculate the left top corner of the grid that is centered
        gx = int(round(horizontal_margin + (avail_w - grid_w)/2))
        gy = int(round(vertical_margin   + (avail_h - grid_h)/2))

        top_l_x = gx
        top_l_y = gy

        gx = top_l_x + starting_pos[0] * rect_size
        gy = top_l_y + starting_pos[1] * rect_size

        # Precompute once
        max_gx = top_l_x + grid_w - rect_size  # top-left x of last column
        max_gy = top_l_y + grid_h - rect_size  # top-left y of last row

        arrow_padding = rect_size * 0.12

        for cur_arw in opt_path:
            match cur_arw:
                case "Up":
                    if gy - rect_size < top_l_y:
                        print("ERROR: arrow wants to get out of bounds")
                        continue
                    arrow = self.arrow_polygon(
                        (gx + rect_size/2, gy + rect_size - arrow_padding),
                        (gx + rect_size/2, gy + arrow_padding)
                    )
                    gy -= rect_size

                case "Down":
                    if gy + rect_size > max_gy:
                        print("ERROR: arrow wants to get out of bounds")
                        continue
                    arrow = self.arrow_polygon(
                        (gx + rect_size/2, gy + arrow_padding),
                        (gx + rect_size/2, gy + rect_size - arrow_padding)
                    )
                    gy += rect_size

                case "Left":
                    if gx - rect_size < top_l_x:
                        print("ERROR: arrow wants to get out of bounds")
                        continue
                    arrow = self.arrow_polygon(
                        (gx + rect_size - arrow_padding,   gy + rect_size/2),
                        (gx + arrow_padding,               gy + rect_size/2)
                    )
                    gx -= rect_size

                case "Right":
                    if gx + rect_size > max_gx:
                        print("ERROR: arrow wants to get out of bounds")
                        continue
                    arrow = self.arrow_polygon(
                        (gx + arrow_padding,               gy + rect_size/2),
                        (gx + rect_size - arrow_padding,   gy + rect_size/2)
                    )
                    gx += rect_size
                case "Center":
                    continue
            if arrow is not None:
                pygame.draw.polygon(self.screen, pygame.Color("blue"), arrow)