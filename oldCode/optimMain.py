# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
import math
import json
import os

pygame.init()

# =========================
#   CONFIG / GLOBAL STATE
# =========================
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 720
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("RL - Maze: Editor, Options & Visual Training")

clock = pygame.time.Clock()
FPS = 60  # configurable in Options (1..500)

# Files
GRIDS_FILE = os.path.join(os.getcwd(), "grids.json")
SETTINGS_FILE = os.path.join(os.getcwd(), "settings.json")

# Colors
BLACK=(0,0,0); WHITE=(255,255,255); GREY=(40,40,40); LIGHT_GREY=(80,80,80); DARK_GREY=(25,25,25)
GREEN=(0,200,0); RED=(200,60,60); RED_DARK=(160,40,40); BLUE=(60,120,240); YELLOW=(230,200,40); TEAL=(40,180,170)
PATH_YELLOW=(240,210,70); PATH_PURPLE=(160,120,220)

def mkfont(sz): return pygame.font.SysFont(None, sz)
TITLE_FONT = mkfont(56); BTN_FONT = mkfont(30); LBL_FONT = mkfont(24); INPUT_FONT = mkfont(26); SMALL_FONT = mkfont(20)

# ---- Cell types ----
class Cell:
    NONE = 0
    START = 1
    FINISH = 2
    OBSTACLE = 3
    TRAP = 4

# ---- Hyperparameters & rewards (editable in Options) ----
alpha   = 0.72
gamma   = 1.0
epsilon = 0.4
mi      = 0.1
beta    = 0.25
EPISODES  = 100
max_steps = 500

# Rewards
reward_start_visual   = 0
reward_finish         = 50
reward_obstacle_block = -10
reward_trap           = -30
reward_valid_move     = -1
reward_invalid_move   = -10

# ---- Grid & map data ----
gridWidth, gridHeight = 20, 15
def blank_map(w, h): return [[Cell.NONE for _ in range(w)] for _ in range(h)]
map_data = blank_map(gridWidth, gridHeight)
start_pos = None
finish_pos = None

# =========================
#         AGENT
# =========================
class Agent:
    """Movement validity uses NEXT state (bounds + obstacle)."""
    def __init__(self, grid_ref_get, rewards_ref_get):
        self.grid_ref_get = grid_ref_get
        self.rewards_ref_get = rewards_ref_get
        self.activeState = [0, 0]
        self.activeReward = 0
        self.changePosArray = [[-1,0],[1,0],[0,-1],[0,1]]  # L,R,U,D
        self.actionOptions = [0,1,2,3]

    def _next_from(self, state, action):
        x,y = state
        dx,dy = self.changePosArray[action]
        return [x+dx, y+dy]

    def CheckNextAction(self, action):
        md, w, h, _, _ = self.grid_ref_get()
        r = self.rewards_ref_get()
        nx, ny = self._next_from(self.activeState, action)

        if not (0 <= nx < w and 0 <= ny < h):
            return r["invalid_move"]

        c = md[ny][nx]
        if c == Cell.OBSTACLE:
            return r["obstacle_block"]
        elif c == Cell.TRAP:
            return r["trap"]
        elif c == Cell.FINISH:
            return r["finish"]
        elif c == Cell.START:
            return r["start_visual"]
        else:
            return r["valid_move"]

    def MakeAction(self, action):
        md, w, h, _, _ = self.grid_ref_get()
        r = self.rewards_ref_get()
        nx, ny = self._next_from(self.activeState, action)

        if not (0 <= nx < w and 0 <= ny < h):
            self.activeReward += r["invalid_move"];  return

        c = md[ny][nx]
        if c == Cell.OBSTACLE:
            self.activeReward += r["obstacle_block"];  return

        self.activeState = [nx, ny]
        if c == Cell.TRAP:      self.activeReward += r["trap"]
        elif c == Cell.FINISH:  self.activeReward += r["finish"]
        elif c == Cell.START:   self.activeReward += r["start_visual"]
        else:                   self.activeReward += r["valid_move"]

    def ProcessNextAction(self, action):
        reward = self.CheckNextAction(action)
        self.MakeAction(action)
        return reward, self.activeState

def grid_ref_get():
    return map_data, gridWidth, gridHeight, start_pos, finish_pos

def rewards_ref_get():
    return {
        "start_visual": reward_start_visual,
        "finish": reward_finish,
        "obstacle_block": reward_obstacle_block,
        "trap": reward_trap,
        "valid_move": reward_valid_move,
        "invalid_move": reward_invalid_move
    }

agent = Agent(grid_ref_get, rewards_ref_get)

def alloc_q():
    return np.zeros((gridHeight, gridWidth, len(agent.actionOptions)), dtype=float)

# =========================
#        UI HELPERS
# =========================
def draw_button(rect, text, mouse_pos, *, base=GREY, hover=LIGHT_GREY, text_color=WHITE, radius=12, active=False):
    hovered = rect.collidepoint(mouse_pos)
    color = hover if (hovered or active) else base
    pygame.draw.rect(SCREEN, color, rect, border_radius=radius)
    txt = BTN_FONT.render(text, True, text_color)
    SCREEN.blit(txt, txt.get_rect(center=rect.center))
    return hovered

class InputBox:
    def __init__(self, rect, text="", numeric=True, float_ok=True, minv=None, maxv=None):
        self.rect = pygame.Rect(rect)
        self.text = str(text)
        self.focus = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_interval = 500
        self.numeric = numeric
        self.float_ok = float_ok
        self.minv = minv
        self.maxv = maxv

    def set_rect(self, rect): self.rect = pygame.Rect(rect)
    def has_focus(self): return self.focus

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.focus = self.rect.collidepoint(e.pos)
        if e.type == pygame.KEYDOWN and self.focus:
            if e.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif e.key == pygame.K_RETURN:
                self.focus = False
            else:
                ch = e.unicode
                if not self.numeric:
                    self.text += ch
                else:
                    allowed = "0123456789"
                    if self.float_ok: allowed += "."
                    if ch in allowed:
                        if ch == "." and "." in self.text: return
                        self.text += ch

    def update(self, dt):
        self.cursor_timer += dt
        if self.cursor_timer >= self.cursor_interval:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def draw(self):
        pygame.draw.rect(SCREEN, DARK_GREY, self.rect, border_radius=10)
        pygame.draw.rect(SCREEN, WHITE if self.focus else LIGHT_GREY, self.rect, width=2, border_radius=10)
        txt = INPUT_FONT.render(self.text, True, WHITE)
        SCREEN.blit(txt, (self.rect.x+10, self.rect.y+(self.rect.h - txt.get_height())//2))
        if self.focus and self.cursor_visible:
            cursor_x = self.rect.x + 10 + txt.get_width() + 2
            pygame.draw.rect(SCREEN, WHITE, (cursor_x, self.rect.y+8, 2, self.rect.h-16))

    def get_int(self, default):
        try:
            v = int(float(self.text)) if self.text else default
            if self.minv is not None: v = max(self.minv, v)
            if self.maxv is not None: v = min(self.maxv, v)
            return v
        except:
            return default

    def get_float(self, default):
        try:
            v = float(self.text) if self.text else default
            if self.minv is not None: v = max(self.minv, v)
            if self.maxv is not None: v = min(self.maxv, v)
            return v
        except:
            return default

# Simple Dropdown (for GridMap load)
class Dropdown:
    def __init__(self, rect, items=None, placeholder="(no maps)", max_visible=4):
        self.rect = pygame.Rect(rect)
        self.items = items or []
        self.open = False
        self.selected = -1
        self.placeholder = placeholder
        self.max_visible = max_visible
        self.item_height = self.rect.height
        self.changed = False  # NEW: becomes True exactly when a new selection is made
        self.scroll_offset = 0  # NEW: top index when open list is scrolled

    def set_rect(self, rect):
        self.rect = pygame.Rect(rect)
        self.item_height = self.rect.height

    def set_items(self, items):
        self.items = list(items)
        if self.selected >= len(self.items):
            self.selected = -1
        self.scroll_offset = 0  # NEW: reset scroll on items change

    def draw(self, mouse_pos):
        # draw field
        txt = self.items[self.selected] if 0 <= self.selected < len(self.items) else self.placeholder
        pygame.draw.rect(SCREEN, DARK_GREY, self.rect, border_radius=10)
        pygame.draw.rect(SCREEN, WHITE, self.rect, 1, border_radius=10)
        SCREEN.blit(INPUT_FONT.render(txt, True, WHITE), (self.rect.x+10, self.rect.y+(self.rect.h-24)//2))
        arrow = "▲" if self.open else "▼"
        SCREEN.blit(INPUT_FONT.render(arrow, True, WHITE), (self.rect.right-28, self.rect.y+(self.rect.h-24)//2))

        # top-layer dropdown list (limited to max_visible, scrolled by scroll_offset)
        if self.open and self.items:
            visible = min(self.max_visible, len(self.items))
            list_rect = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.w, visible*self.item_height)
            pygame.draw.rect(SCREEN, DARK_GREY, list_rect, border_radius=10)
            pygame.draw.rect(SCREEN, WHITE, list_rect, 1, border_radius=10)
            # render visible window
            for i in range(visible):
                idx = self.scroll_offset + i
                if idx >= len(self.items): break
                r = pygame.Rect(list_rect.x, list_rect.y+i*self.item_height, list_rect.w, self.item_height)
                if r.collidepoint(mouse_pos): pygame.draw.rect(SCREEN, LIGHT_GREY, r)
                SCREEN.blit(INPUT_FONT.render(self.items[idx], True, WHITE), (r.x+10, r.y+(r.h-24)//2))

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.rect.collidepoint(e.pos):
                self.open = not self.open
            elif self.open:
                visible = min(self.max_visible, len(self.items))
                list_rect = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.w, visible*self.item_height)
                if list_rect.collidepoint(e.pos):
                    rel_y = e.pos[1] - list_rect.y
                    i = rel_y // self.item_height
                    idx = self.scroll_offset + int(i)
                    if 0 <= idx < len(self.items):
                        if idx != self.selected:
                            self.selected = idx
                            self.changed = True
                self.open = False
        # wheel scroll (both legacy buttons 4/5 and pygame.MOUSEWHEEL)
        if self.open:
            visible = min(self.max_visible, len(self.items))
            max_off = max(0, len(self.items) - visible)
            if e.type == pygame.MOUSEWHEEL:
                # e.y > 0 means scroll up
                self.scroll_offset = max(0, min(max_off, self.scroll_offset - e.y))
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (4,5):
                dy = 1 if e.button == 4 else -1  # 4: up, 5: down
                self.scroll_offset = max(0, min(max_off, self.scroll_offset - dy))

# Simple Slider class for Q-overlay controls
class Slider:
    def __init__(self, rect, min_val, max_val, initial_val, integer=False):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.integer = integer
        self.dragging = False
        
    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.rect.collidepoint(e.pos):
                self.dragging = True
                self._update_value(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(e.pos[0])
    
    def _update_value(self, x):
        rel_x = max(0, min(self.rect.width, x - self.rect.x))
        ratio = rel_x / self.rect.width
        self.val = self.min_val + ratio * (self.max_val - self.min_val)
        if self.integer:
            self.val = int(self.val)
    
    def draw(self, label="", fmt=None):
        # Background
        pygame.draw.rect(SCREEN, DARK_GREY, self.rect, border_radius=5)
        pygame.draw.rect(SCREEN, WHITE, self.rect, 2, border_radius=5)
        
        # Handle position
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.width
        handle_rect = pygame.Rect(handle_x - 5, self.rect.y - 2, 10, self.rect.height + 4)
        pygame.draw.rect(SCREEN, WHITE, handle_rect, border_radius=3)
        
        # Label
        if label:
            text = f"{label}: {fmt(self.val) if fmt else self.val}"
            txt_surf = SMALL_FONT.render(text, True, WHITE)
            SCREEN.blit(txt_surf, (self.rect.x, self.rect.y - 20))

# =========================
#     DRAWING HELPERS
# =========================
def draw_cell_visual(screen, cell_rect, cell_type):
    """Draw visual representation of different cell types"""
    if cell_type == Cell.START:
        pygame.draw.rect(screen, GREEN, cell_rect.inflate(-6, -6), border_radius=3)
    elif cell_type == Cell.FINISH:
        pygame.draw.rect(screen, BLUE, cell_rect.inflate(-6, -6), border_radius=3)
    elif cell_type == Cell.OBSTACLE:
        pygame.draw.rect(screen, RED, cell_rect.inflate(-2, -2))
    elif cell_type == Cell.TRAP:
        pygame.draw.rect(screen, YELLOW, cell_rect.inflate(-4, -4), border_radius=2)

def draw_grid_lines(screen, origin, cell_size, cols, rows):
    """Draw grid lines for the maze"""
    # Vertical lines
    for x in range(cols + 1):
        start_pos = (origin.x + x * cell_size, origin.y)
        end_pos = (origin.x + x * cell_size, origin.y + rows * cell_size)
        pygame.draw.line(screen, WHITE, start_pos, end_pos)
    
    # Horizontal lines
    for y in range(rows + 1):
        start_pos = (origin.x, origin.y + y * cell_size)
        end_pos = (origin.x + cols * cell_size, origin.y + y * cell_size)
        pygame.draw.line(screen, WHITE, start_pos, end_pos)

# =========================
#       SAVE / LOAD MAPS
# =========================
def read_grids():
    if not os.path.exists(GRIDS_FILE):
        return {}
    try:
        with open(GRIDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def write_grids(data: dict):
    try:
        with open(GRIDS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Failed to write grids.json:", e)

def list_grid_names():
    data = read_grids()
    return list(data.keys())

def save_current_grid(name: str):
    if not name: return
    data = read_grids()
    data[name] = {
        "width": gridWidth,
        "height": gridHeight,
        "cells": map_data
    }
    write_grids(data)

def load_grid(name: str):
    global gridWidth, gridHeight, map_data, start_pos, finish_pos
    data = read_grids()
    if name not in data: return
    entry = data[name]
    w = int(entry.get("width", 1))
    h = int(entry.get("height", 1))
    cells = entry.get("cells", [])
    if not cells or len(cells) != h or any(len(row) != w for row in cells):
        print("Saved grid shape mismatch; ignoring.")
        return
    gridWidth, gridHeight = w, h
    map_data = [[int(c) for c in row] for row in cells]
    start = finish = None
    for y in range(gridHeight):
        for x in range(gridWidth):
            if map_data[y][x] == Cell.START:  start = (x,y)
            if map_data[y][x] == Cell.FINISH: finish = (x,y)
    start_pos, finish_pos = start, finish

load_dropdown = Dropdown((0,0,10,10), items=list_grid_names(), placeholder="Load Map ▼")
# NEW: Algorithm selector (Options scene)
ALGORITHMS = ["SARSA", "Q-Learning"]
algorithm = "SARSA"
algorithm_dropdown = Dropdown((0,0,10,10), items=ALGORITHMS, placeholder="Algorithm", max_visible=4)

def _sync_algorithm_dropdown():
    try:
        algorithm_dropdown.selected = algorithm_dropdown.items.index(algorithm)
    except ValueError:
        algorithm_dropdown.selected = -1

# =========================
#          SCENES
# =========================
class Scene:
    MENU="menu"; TRAINING="training"; GRIDMAP="gridmap"; OPTIONS="options"
state = Scene.MENU
prev_state = None

# GRIDMAP editing buffers
edit_w, edit_h = gridWidth, gridHeight
edit_map = [row[:] for row in map_data]
edit_start = None
edit_finish = None

def init_edit_from_runtime():
    global edit_w, edit_h, edit_map, edit_start, edit_finish
    edit_w, edit_h = gridWidth, gridHeight
    edit_map = [row[:] for row in map_data]
    edit_start = edit_finish = None
    for y in range(edit_h):
        for x in range(edit_w):
            if edit_map[y][x] == Cell.START:  edit_start  = (x,y)
            if edit_map[y][x] == Cell.FINISH: edit_finish = (x,y)

def apply_edit_to_runtime():
    global gridWidth, gridHeight, map_data, start_pos, finish_pos
    gridWidth, gridHeight = edit_w, edit_h
    map_data = [row[:] for row in edit_map]
    start_pos = finish_pos = None
    for y in range(gridHeight):
        for x in range(gridWidth):
            c = map_data[y][x]
            if c == Cell.START:  start_pos  = (x,y)
            if c == Cell.FINISH: finish_pos = (x,y)

# --- LAYOUT HELPERS ---
def layout_menu():
    bw = int(SCREEN_WIDTH*0.36); bh=int(SCREEN_HEIGHT*0.09)
    cx = SCREEN_WIDTH//2
    y1 = SCREEN_HEIGHT//2 - bh - int(SCREEN_HEIGHT*0.015)
    y2 = SCREEN_HEIGHT//2 + int(SCREEN_HEIGHT*0.015)
    y3 = y2 + bh + int(SCREEN_HEIGHT*0.02)
    start_btn   = pygame.Rect(cx-bw//2, y1, bw, bh)
    gridmap_btn = pygame.Rect(cx-bw//2, y2, bw, bh)
    options_btn = pygame.Rect(cx-bw//2, y3, bw, bh)
    return start_btn, gridmap_btn, options_btn

def layout_training():
    # two rows so everything is always visible
    bw = int(SCREEN_WIDTH*0.16); bh=int(SCREEN_HEIGHT*0.065); gap=int(SCREEN_WIDTH*0.012)
    left = int(SCREEN_WIDTH*0.04)
    row1_y = SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.16)
    row2_y = SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.08)

    play_btn   = pygame.Rect(left, row1_y, bw, bh)
    reset_btn  = pygame.Rect(play_btn.right + gap, row1_y, bw, bh)
    stop_btn   = pygame.Rect(reset_btn.right + gap, row1_y, bw, bh)
    qviz_btn   = pygame.Rect(stop_btn.right + gap, row1_y, bw, bh)

    show_first_btn = pygame.Rect(left, row2_y, bw, bh)
    show_best_btn  = pygame.Rect(show_first_btn.right + gap, row2_y, bw, bh)
    clear_path_btn = pygame.Rect(show_best_btn.right + gap, row2_y, bw, bh)

    # sliders panel above row1 (for Q overlay)
    panel_h = int(SCREEN_HEIGHT*0.10)
    sliders_panel = pygame.Rect(left, row1_y - panel_h - int(SCREEN_HEIGHT*0.015), int(SCREEN_WIDTH*0.92), panel_h)
    s_gap = int(sliders_panel.w*0.02)
    s_w = (sliders_panel.w - s_gap*3)//2
    font_slider_rect = pygame.Rect(sliders_panel.x + s_gap, sliders_panel.y + sliders_panel.h//2 - 10, s_w, 20)
    alpha_slider_rect= pygame.Rect(font_slider_rect.right + s_gap, font_slider_rect.y, s_w, 20)

    return (play_btn, reset_btn, stop_btn, qviz_btn,
            show_first_btn, show_best_btn, clear_path_btn,
            sliders_panel, font_slider_rect, alpha_slider_rect)

def layout_gridmap():
    pw = int(SCREEN_WIDTH*0.92); ph=int(SCREEN_HEIGHT*0.86)
    panel = pygame.Rect((SCREEN_WIDTH-pw)//2, (SCREEN_HEIGHT-ph)//2, pw, ph)
    pad = int(min(pw, ph)*0.03)
    preview_area = pygame.Rect(panel.x+pad, panel.y+int(ph*0.18), int(pw*0.55), ph-int(ph*0.28))

    controls_left = preview_area.right + pad
    input_w=int(pw*0.14); input_h=int(ph*0.075); input_gap=int(pw*0.015)
    inputs_y = panel.y + int(ph*0.20)
    width_rect  = pygame.Rect(controls_left, inputs_y, input_w, input_h)
    height_rect = pygame.Rect(controls_left + input_w + input_gap, inputs_y, input_w, input_h)

    tool_w=input_w; tool_h=int(ph*0.07); tool_gap=int(pw*0.012)
    tools_top = inputs_y + input_h + int(ph*0.04)
    t_start  = pygame.Rect(controls_left, tools_top, tool_w, tool_h)
    t_finish = pygame.Rect(controls_left+tool_w+tool_gap, tools_top, tool_w, tool_h)
    t_obst   = pygame.Rect(controls_left, tools_top+tool_h+tool_gap, tool_w, tool_h)
    t_trap   = pygame.Rect(controls_left+tool_w+tool_gap, tools_top+tool_h+tool_gap, tool_w, tool_h)

    # Save/Load controls
    save_name_rect = pygame.Rect(controls_left, t_trap.bottom + int(ph*0.05), tool_w*2 + tool_gap, input_h)
    save_btn       = pygame.Rect(controls_left, save_name_rect.bottom + tool_gap, tool_w, tool_h)
    load_dd_rect   = pygame.Rect(controls_left + tool_w + tool_gap, save_name_rect.bottom + tool_gap, tool_w, tool_h)

    done_w = tool_w*2 + tool_gap; done_h=int(ph*0.085)
    done_btn = pygame.Rect(controls_left, panel.bottom - done_h - int(ph*0.06), done_w, done_h)
    erase_w = done_w; erase_h=int(ph*0.06)
    erase_btn = pygame.Rect(controls_left, done_btn.top - erase_h - int(ph*0.03), erase_w, erase_h)

    return dict(panel=panel, preview=preview_area, width=width_rect, height=height_rect,
                t_start=t_start, t_finish=t_finish, t_obst=t_obst, t_trap=t_trap,
                save_name=save_name_rect, save_btn=save_btn, load_dd=load_dd_rect,
                done=done_btn, erase=erase_btn)

# =========================
#       SCENE DRAWS
# =========================
def compute_main_grid_layout():
    cols, rows = gridWidth, gridHeight
    cols = max(1, cols); rows = max(1, rows)
    tile_w = SCREEN_WIDTH // cols
    tile_h = (SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.22)) // rows
    tile_size = max(6, min(tile_w, tile_h))
    gpw, gph = cols*tile_size, rows*tile_size
    ox = (SCREEN_WIDTH - gpw)//2
    oy = (SCREEN_HEIGHT - gph)//2 - int(SCREEN_HEIGHT*0.03)
    return tile_size, gpw, gph, ox, oy

def draw_menu(mouse_pos, start_btn, gridmap_btn, options_btn):
    SCREEN.fill(BLACK)
    SCREEN.blit(TITLE_FONT.render("Maze RL", True, WHITE), (SCREEN_WIDTH//2-120, int(SCREEN_HEIGHT*0.18)))
    draw_button(start_btn,   "Start training (visual)", mouse_pos)
    draw_button(gridmap_btn, "GridMap",                 mouse_pos)
    draw_button(options_btn, "Options",                 mouse_pos)

# ---------- Q-values overlay state & helpers ----------
qviz_show = False
qviz_font_px = 16   # 10..36
qviz_alpha  = 200   # 40..255

def draw_q_overlay(Q, tile_size, ox, oy):
    font = pygame.font.SysFont(None, max(10, min(72, int(qviz_font_px))))
    fw = font.size("0.0")[0]
    for y in range(gridHeight):
        for x in range(gridWidth):
            q = Q[y, x]
            cx = ox + x*tile_size
            cy = oy + y*tile_size
            def blit_val(val, pos):
                s = font.render(f"{val:.1f}", True, WHITE)
                s = s.convert_alpha(); s.set_alpha(int(qviz_alpha))
                SCREEN.blit(s, pos)
            pad = max(2, tile_size//12)
            blit_val(q[0], (cx + pad,                 cy + tile_size//2 - font.get_height()//2))             # L
            blit_val(q[1], (cx + tile_size - pad - fw, cy + tile_size//2 - font.get_height()//2))            # R
            blit_val(q[2], (cx + tile_size//2 - fw//2, cy + pad))                                            # U
            blit_val(q[3], (cx + tile_size//2 - fw//2, cy + tile_size - font.get_height() - pad))           # D

def draw_path_overlay(path, ox, oy, tile_size, color=PATH_YELLOW, width=3):
    if not path or len(path) < 2: return
    pts = []
    for (x,y) in path:
        cx = ox + x*tile_size + tile_size//2
        cy = oy + y*tile_size + tile_size//2
        pts.append((cx,cy))
    pygame.draw.lines(SCREEN, color, False, pts, width)
    sx,sy = pts[0]; gx,gy = pts[-1]
    pygame.draw.circle(SCREEN, color, (sx,sy), max(3, tile_size//6))
    pygame.draw.circle(SCREEN, color, (gx,gy), max(4, tile_size//5), width=2)

def draw_training(mouse_pos, controls, hud, sliders_panel, font_slider, alpha_slider):
    (play_btn, reset_btn, stop_btn, qviz_btn, show_first_btn, show_best_btn, clear_path_btn) = controls
    SCREEN.fill(BLACK)
    tile_size, gpw, gph, ox, oy = compute_main_grid_layout()
    # grid
    for x in range(0, gpw+1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (ox+x, oy), (ox+x, oy+gph))
    for y in range(0, gph+1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (ox, oy+y), (ox+gpw, oy+y))
    # visuals
    for gy in range(gridHeight):
        for gx in range(gridWidth):
            c = map_data[gy][gx]
            if c == Cell.NONE: continue
            cell_rect = pygame.Rect(ox + gx*tile_size, oy + gy*tile_size, tile_size, tile_size)
            draw_cell_visual(SCREEN, cell_rect, c)

    # overlays / agent
    if hud["show_first"] and hud["first_path"]:
        draw_path_overlay(hud["first_path"], ox, oy, tile_size, color=PATH_YELLOW, width=4)
    if hud["show_best"] and hud["best_path"]:
        draw_path_overlay(hud["best_path"], ox, oy, tile_size, color=PATH_PURPLE, width=4)
    if hud["agent_pos"] is not None and not (hud["show_first"] or hud["show_best"]):
        ax, ay = hud["agent_pos"]
        rect = pygame.Rect(ox + ax*tile_size, oy + ay*tile_size, tile_size, tile_size)
        pygame.draw.rect(SCREEN, TEAL, rect.inflate(-int(tile_size*0.25), -int(tile_size*0.25)), border_radius=6)

    # Q overlay
    if qviz_show and Q is not None:
        draw_q_overlay(Q, tile_size, ox, oy)

    # HUD text
    info_y = max(10, oy - int(SCREEN_HEIGHT*0.07))
    text = f"Episode: {hud['episode']}/{EPISODES}   Step: {hud['step']}/{max_steps}   Reward: {hud['reward']:.1f}   Eps: {epsilon:.3f}   Alpha: {alpha:.3f}   Finishes: {hud['finishes']}"
    SCREEN.blit(LBL_FONT.render(text, True, WHITE), (int(SCREEN_WIDTH*0.05), info_y))

    # buttons (two rows)
    draw_button(play_btn,  "Pause" if hud["running"] else "Play", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)
    draw_button(reset_btn, "Reset Q", mouse_pos, base=GREY, hover=LIGHT_GREY)
    draw_button(stop_btn,  "Stop",    mouse_pos, base=RED,  hover=(220,90,90))
    draw_button(qviz_btn,  "Show Q-values" if not qviz_show else "Hide Q-values", mouse_pos, base=GREY, hover=LIGHT_GREY, active=qviz_show)

    draw_button(show_first_btn, "Show First Path", mouse_pos, base=GREY, hover=LIGHT_GREY, active=hud["show_first"])
    draw_button(show_best_btn,  "Show Optimal Path", mouse_pos, base=GREY, hover=LIGHT_GREY, active=hud["show_best"])
    draw_button(clear_path_btn, "Clear Path Overlay", mouse_pos, base=GREY, hover=LIGHT_GREY)

    # sliders panel (only for Q overlay)
    if qviz_show:
        pygame.draw.rect(SCREEN, (30,30,30), sliders_panel, border_radius=12)
        pygame.draw.rect(SCREEN, WHITE, sliders_panel, 1, border_radius=12)
        font_slider.draw("Font", fmt=lambda v: f"{int(v)} px")
        alpha_slider.draw("Alpha", fmt=lambda v: f"{int(v)}")

def draw_gridmap(mouse_pos, dt, L):
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, GREY, L["panel"], border_radius=18)
    pygame.draw.rect(SCREEN, WHITE, L["panel"], 1, border_radius=18)
    t = TITLE_FONT.render("Grid Map", True, WHITE)
    SCREEN.blit(t, t.get_rect(center=(L["panel"].centerx, L["panel"].y + int(L["panel"].h*0.09))))

    # inputs W/H
    width_input.set_rect(L["width"]);  height_input.set_rect(L["height"])
    width_input.update(dt); height_input.update(dt)
    SCREEN.blit(LBL_FONT.render("Width", True, WHITE),  (L["width"].x,  L["width"].y - 22))
    SCREEN.blit(LBL_FONT.render("Height", True, WHITE), (L["height"].x, L["height"].y - 22))
    width_input.draw(); height_input.draw()

    # preview
    cols = max(1, width_input.get_int(edit_w))
    rows = max(1, height_input.get_int(edit_h))
    pr = L["preview"]
    pygame.draw.rect(SCREEN, DARK_GREY, pr, border_radius=12)
    pygame.draw.rect(SCREEN, WHITE, pr, 1, border_radius=12)
    cell_size = max(6, min(pr.w//cols, pr.h//rows))
    preview_w = cell_size*cols; preview_h = cell_size*rows
    origin = pygame.Rect(0,0, preview_w, preview_h); origin.center = pr.center
    draw_grid_lines(SCREEN, origin, cell_size, cols, rows)

    for gy in range(min(rows, edit_h)):
        for gx in range(min(cols, edit_w)):
            c = edit_map[gy][gx]
            if c == Cell.NONE: continue
            cell_rect = pygame.Rect(origin.x + gx*cell_size, origin.y + gy*cell_size, cell_size, cell_size)
            draw_cell_visual(SCREEN, cell_rect, c)

    # tools
    SCREEN.blit(LBL_FONT.render("Tools", True, WHITE), (L["t_start"].x, L["t_start"].y - 24))
    draw_button(L["t_start"],  "Start",    mouse_pos, active=(current_tool==Cell.START))
    draw_button(L["t_finish"], "Finish",   mouse_pos, active=(current_tool==Cell.FINISH))
    draw_button(L["t_obst"],   "Obstacle", mouse_pos, active=(current_tool==Cell.OBSTACLE))
    draw_button(L["t_trap"],   "Trap",     mouse_pos, active=(current_tool==Cell.TRAP))

    # save/load
    save_name_input.set_rect(L["save_name"])
    save_name_input.update(dt)
    SCREEN.blit(LBL_FONT.render("Save Name", True, WHITE), (L["save_name"].x, L["save_name"].y - 22))
    save_name_input.draw()
    draw_button(L["save_btn"], "Save Map", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    # position only here; draw later on top
    load_dropdown.set_rect(L["load_dd"])
    SCREEN.blit(SMALL_FONT.render("Pick to load", True, LIGHT_GREY), (L["load_dd"].x, L["load_dd"].bottom + 6))

    # erase/done
    draw_button(L["erase"], "Erase All", mouse_pos, base=LIGHT_GREY, hover=WHITE, text_color=BLACK)
    draw_button(L["done"],  "Done",      mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    hint = SMALL_FONT.render("Left click = paint, Right click = erase (1..4 to switch tool)", True, LIGHT_GREY)
    SCREEN.blit(hint, hint.get_rect(midtop=(origin.centerx, origin.bottom + 10)))

    # TOP-LAYER: draw dropdown last so it stays above everything
    load_dropdown.draw(mouse_pos)

    return origin, cell_size

def layout_options():
    pw = int(SCREEN_WIDTH*0.92); ph=int(SCREEN_HEIGHT*0.86)
    panel = pygame.Rect((SCREEN_WIDTH-pw)//2, (SCREEN_HEIGHT-ph)//2, pw, ph)
    pad = int(min(pw, ph)*0.03)
    col_w = (pw - pad*3) // 2
    left_col  = pygame.Rect(panel.x+pad, panel.y+int(ph*0.18), col_w, ph-int(ph*0.30))
    right_col = pygame.Rect(left_col.right+pad, left_col.y, col_w, left_col.h)

    # NEW: algorithm dropdown rect under the title
    algo_rect = pygame.Rect(panel.x+pad, panel.y+int(ph*0.12), int(pw*0.28), int(ph*0.075))

    # bottom row: Save/Load settings + Done
    row_h = int(ph*0.085)
    btn_y = panel.bottom - row_h - int(ph*0.06)
    save_settings_btn = pygame.Rect(panel.x+pad, btn_y, int(pw*0.22), row_h)
    load_settings_btn = pygame.Rect(save_settings_btn.right + pad, btn_y, int(pw*0.22), row_h)
    done_btn = pygame.Rect(panel.right - pad - int(pw*0.22), btn_y, int(pw*0.22), row_h)

    def row_rect(col, i): return pygame.Rect(col.x+10, col.y+i*row_h, col.w-20, row_h-10)

    return dict(panel=panel, left=left_col, right=right_col, row=row_rect,
                save_btn=save_settings_btn, load_btn=load_settings_btn, done=done_btn,
                algo=algo_rect)

def draw_options(mouse_pos, dt, L):
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, GREY, L["panel"], border_radius=18)
    pygame.draw.rect(SCREEN, WHITE, L["panel"], 1, border_radius=18)
    t = TITLE_FONT.render("Options", True, WHITE)
    SCREEN.blit(t, t.get_rect(center=(L["panel"].centerx, L["panel"].y + int(L["panel"].h*0.09))))

    # NEW: Algorithm selector
    SCREEN.blit(LBL_FONT.render("Algorithm", True, WHITE), (L["algo"].x, L["algo"].y - 22))
    algorithm_dropdown.set_rect(L["algo"])

    ensure_option_inputs(L)
    assign_option_rects(L)

    pygame.draw.rect(SCREEN, DARK_GREY, L["left"],  border_radius=10)
    pygame.draw.rect(SCREEN, DARK_GREY, L["right"], border_radius=10)

    for key, ib in opt_inputs.items():
        ib.update(dt)
        SCREEN.blit(LBL_FONT.render(ib._label_text, True, WHITE), (ib._label_rect.x, ib._label_rect.y + (ib._label_rect.h-22)//2))
        ib.draw()

    draw_button(L["save_btn"], "Save Settings", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)
    draw_button(L["load_btn"], "Load Settings", mouse_pos, base=GREY, hover=LIGHT_GREY)
    draw_button(L["done"],     "Save & Back",   mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    # TOP-LAYER: draw dropdown last so it stays above everything
    algorithm_dropdown.draw(mouse_pos)

# =========================
#   SETTINGS SAVE / LOAD
# =========================
def current_settings_dict():
    return {
        "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "mi": mi, "beta": beta,
        "EPISODES": EPISODES, "max_steps": max_steps, "FPS": FPS,
        "reward_start_visual": reward_start_visual,
        "reward_finish": reward_finish,
        "reward_obstacle_block": reward_obstacle_block,
        "reward_trap": reward_trap,
        "reward_valid_move": reward_valid_move,
        "reward_invalid_move": reward_invalid_move,
        "algorithm": algorithm,  # NEW
    }

def apply_settings_dict(d):
    global alpha,gamma,epsilon,mi,beta,EPISODES,max_steps,FPS
    global reward_start_visual,reward_finish,reward_obstacle_block,reward_trap,reward_valid_move,reward_invalid_move
    global algorithm
    if not d: return
    alpha   = float(d.get("alpha", alpha))
    gamma   = float(d.get("gamma", gamma))
    epsilon = float(d.get("epsilon", epsilon))
    mi      = float(d.get("mi", mi))
    beta    = float(d.get("beta", beta))
    EPISODES  = int(d.get("EPISODES", EPISODES))
    max_steps = int(d.get("max_steps", max_steps))
    FPS       = int(d.get("FPS", FPS))
    reward_start_visual   = float(d.get("reward_start_visual", reward_start_visual))
    reward_finish         = float(d.get("reward_finish", reward_finish))
    reward_obstacle_block = float(d.get("reward_obstacle_block", reward_obstacle_block))
    reward_trap           = float(d.get("reward_trap", reward_trap))
    reward_valid_move     = float(d.get("reward_valid_move", reward_valid_move))
    reward_invalid_move   = float(d.get("reward_invalid_move", reward_invalid_move))
    algorithm = str(d.get("algorithm", algorithm))  # NEW

def read_settings():
    if not os.path.exists(SETTINGS_FILE): return None
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Failed to read settings.json:", e)
        return None

def write_settings(d):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print("Failed to write settings.json:", e)

# =========================
#       SAVE / LOAD MAPS
# =========================
def read_grids():
    if not os.path.exists(GRIDS_FILE):
        return {}
    try:
        with open(GRIDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def write_grids(data: dict):
    try:
        with open(GRIDS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Failed to write grids.json:", e)

def list_grid_names():
    data = read_grids()
    return list(data.keys())

def save_current_grid(name: str):
    if not name: return
    data = read_grids()
    data[name] = {
        "width": gridWidth,
        "height": gridHeight,
        "cells": map_data
    }
    write_grids(data)

def load_grid(name: str):
    global gridWidth, gridHeight, map_data, start_pos, finish_pos
    data = read_grids()
    if name not in data: return
    entry = data[name]
    w = int(entry.get("width", 1))
    h = int(entry.get("height", 1))
    cells = entry.get("cells", [])
    if not cells or len(cells) != h or any(len(row) != w for row in cells):
        print("Saved grid shape mismatch; ignoring.")
        return
    gridWidth, gridHeight = w, h
    map_data = [[int(c) for c in row] for row in cells]
    start = finish = None
    for y in range(gridHeight):
        for x in range(gridWidth):
            if map_data[y][x] == Cell.START:  start = (x,y)
            if map_data[y][x] == Cell.FINISH: finish = (x,y)
    start_pos, finish_pos = start, finish

load_dropdown = Dropdown((0,0,10,10), items=list_grid_names(), placeholder="Load Map ▼")
# NEW: Algorithm selector (Options scene)
ALGORITHMS = ["SARSA", "Q-Learning"]
algorithm = "SARSA"
algorithm_dropdown = Dropdown((0,0,10,10), items=ALGORITHMS, placeholder="Algorithm", max_visible=4)

def _sync_algorithm_dropdown():
    try:
        algorithm_dropdown.selected = algorithm_dropdown.items.index(algorithm)
    except ValueError:
        algorithm_dropdown.selected = -1

# =========================
#          SCENES
# =========================
class Scene:
    MENU="menu"; TRAINING="training"; GRIDMAP="gridmap"; OPTIONS="options"
state = Scene.MENU
prev_state = None

# GRIDMAP editing buffers
edit_w, edit_h = gridWidth, gridHeight
edit_map = [row[:] for row in map_data]
edit_start = None
edit_finish = None

def init_edit_from_runtime():
    global edit_w, edit_h, edit_map, edit_start, edit_finish
    edit_w, edit_h = gridWidth, gridHeight
    edit_map = [row[:] for row in map_data]
    edit_start = edit_finish = None
    for y in range(edit_h):
        for x in range(edit_w):
            if edit_map[y][x] == Cell.START:  edit_start  = (x,y)
            if edit_map[y][x] == Cell.FINISH: edit_finish = (x,y)

def apply_edit_to_runtime():
    global gridWidth, gridHeight, map_data, start_pos, finish_pos
    gridWidth, gridHeight = edit_w, edit_h
    map_data = [row[:] for row in edit_map]
    start_pos = finish_pos = None
    for y in range(gridHeight):
        for x in range(gridWidth):
            c = map_data[y][x]
            if c == Cell.START:  start_pos  = (x,y)
            if c == Cell.FINISH: finish_pos = (x,y)

# --- LAYOUT HELPERS ---
def layout_menu():
    bw = int(SCREEN_WIDTH*0.36); bh=int(SCREEN_HEIGHT*0.09)
    cx = SCREEN_WIDTH//2
    y1 = SCREEN_HEIGHT//2 - bh - int(SCREEN_HEIGHT*0.015)
    y2 = SCREEN_HEIGHT//2 + int(SCREEN_HEIGHT*0.015)
    y3 = y2 + bh + int(SCREEN_HEIGHT*0.02)
    start_btn   = pygame.Rect(cx-bw//2, y1, bw, bh)
    gridmap_btn = pygame.Rect(cx-bw//2, y2, bw, bh)
    options_btn = pygame.Rect(cx-bw//2, y3, bw, bh)
    return start_btn, gridmap_btn, options_btn

def layout_training():
    # two rows so everything is always visible
    bw = int(SCREEN_WIDTH*0.16); bh=int(SCREEN_HEIGHT*0.065); gap=int(SCREEN_WIDTH*0.012)
    left = int(SCREEN_WIDTH*0.04)
    row1_y = SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.16)
    row2_y = SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.08)

    play_btn   = pygame.Rect(left, row1_y, bw, bh)
    reset_btn  = pygame.Rect(play_btn.right + gap, row1_y, bw, bh)
    stop_btn   = pygame.Rect(reset_btn.right + gap, row1_y, bw, bh)
    qviz_btn   = pygame.Rect(stop_btn.right + gap, row1_y, bw, bh)

    show_first_btn = pygame.Rect(left, row2_y, bw, bh)
    show_best_btn  = pygame.Rect(show_first_btn.right + gap, row2_y, bw, bh)
    clear_path_btn = pygame.Rect(show_best_btn.right + gap, row2_y, bw, bh)

    # sliders panel above row1 (for Q overlay)
    panel_h = int(SCREEN_HEIGHT*0.10)
    sliders_panel = pygame.Rect(left, row1_y - panel_h - int(SCREEN_HEIGHT*0.015), int(SCREEN_WIDTH*0.92), panel_h)
    s_gap = int(sliders_panel.w*0.02)
    s_w = (sliders_panel.w - s_gap*3)//2
    font_slider_rect = pygame.Rect(sliders_panel.x + s_gap, sliders_panel.y + sliders_panel.h//2 - 10, s_w, 20)
    alpha_slider_rect= pygame.Rect(font_slider_rect.right + s_gap, font_slider_rect.y, s_w, 20)

    return (play_btn, reset_btn, stop_btn, qviz_btn,
            show_first_btn, show_best_btn, clear_path_btn,
            sliders_panel, font_slider_rect, alpha_slider_rect)

def layout_gridmap():
    pw = int(SCREEN_WIDTH*0.92); ph=int(SCREEN_HEIGHT*0.86)
    panel = pygame.Rect((SCREEN_WIDTH-pw)//2, (SCREEN_HEIGHT-ph)//2, pw, ph)
    pad = int(min(pw, ph)*0.03)
    preview_area = pygame.Rect(panel.x+pad, panel.y+int(ph*0.18), int(pw*0.55), ph-int(ph*0.28))

    controls_left = preview_area.right + pad
    input_w=int(pw*0.14); input_h=int(ph*0.075); input_gap=int(pw*0.015)
    inputs_y = panel.y + int(ph*0.20)
    width_rect  = pygame.Rect(controls_left, inputs_y, input_w, input_h)
    height_rect = pygame.Rect(controls_left + input_w + input_gap, inputs_y, input_w, input_h)

    tool_w=input_w; tool_h=int(ph*0.07); tool_gap=int(pw*0.012)
    tools_top = inputs_y + input_h + int(ph*0.04)
    t_start  = pygame.Rect(controls_left, tools_top, tool_w, tool_h)
    t_finish = pygame.Rect(controls_left+tool_w+tool_gap, tools_top, tool_w, tool_h)
    t_obst   = pygame.Rect(controls_left, tools_top+tool_h+tool_gap, tool_w, tool_h)
    t_trap   = pygame.Rect(controls_left+tool_w+tool_gap, tools_top+tool_h+tool_gap, tool_w, tool_h)

    # Save/Load controls
    save_name_rect = pygame.Rect(controls_left, t_trap.bottom + int(ph*0.05), tool_w*2 + tool_gap, input_h)
    save_btn       = pygame.Rect(controls_left, save_name_rect.bottom + tool_gap, tool_w, tool_h)
    load_dd_rect   = pygame.Rect(controls_left + tool_w + tool_gap, save_name_rect.bottom + tool_gap, tool_w, tool_h)

    done_w = tool_w*2 + tool_gap; done_h=int(ph*0.085)
    done_btn = pygame.Rect(controls_left, panel.bottom - done_h - int(ph*0.06), done_w, done_h)
    erase_w = done_w; erase_h=int(ph*0.06)
    erase_btn = pygame.Rect(controls_left, done_btn.top - erase_h - int(ph*0.03), erase_w, erase_h)

    return dict(panel=panel, preview=preview_area, width=width_rect, height=height_rect,
                t_start=t_start, t_finish=t_finish, t_obst=t_obst, t_trap=t_trap,
                save_name=save_name_rect, save_btn=save_btn, load_dd=load_dd_rect,
                done=done_btn, erase=erase_btn)

# =========================
#       SCENE DRAWS
# =========================
def compute_main_grid_layout():
    cols, rows = gridWidth, gridHeight
    cols = max(1, cols); rows = max(1, rows)
    tile_w = SCREEN_WIDTH // cols
    tile_h = (SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.22)) // rows
    tile_size = max(6, min(tile_w, tile_h))
    gpw, gph = cols*tile_size, rows*tile_size
    ox = (SCREEN_WIDTH - gpw)//2
    oy = (SCREEN_HEIGHT - gph)//2 - int(SCREEN_HEIGHT*0.03)
    return tile_size, gpw, gph, ox, oy

def draw_menu(mouse_pos, start_btn, gridmap_btn, options_btn):
    SCREEN.fill(BLACK)
    SCREEN.blit(TITLE_FONT.render("Maze RL", True, WHITE), (SCREEN_WIDTH//2-120, int(SCREEN_HEIGHT*0.18)))
    draw_button(start_btn,   "Start training (visual)", mouse_pos)
    draw_button(gridmap_btn, "GridMap",                 mouse_pos)
    draw_button(options_btn, "Options",                 mouse_pos)

# ---------- Q-values overlay state & helpers ----------
qviz_show = False
qviz_font_px = 16   # 10..36
qviz_alpha  = 200   # 40..255

def draw_q_overlay(Q, tile_size, ox, oy):
    font = pygame.font.SysFont(None, max(10, min(72, int(qviz_font_px))))
    fw = font.size("0.0")[0]
    for y in range(gridHeight):
        for x in range(gridWidth):
            q = Q[y, x]
            cx = ox + x*tile_size
            cy = oy + y*tile_size
            def blit_val(val, pos):
                s = font.render(f"{val:.1f}", True, WHITE)
                s = s.convert_alpha(); s.set_alpha(int(qviz_alpha))
                SCREEN.blit(s, pos)
            pad = max(2, tile_size//12)
            blit_val(q[0], (cx + pad,                 cy + tile_size//2 - font.get_height()//2))             # L
            blit_val(q[1], (cx + tile_size - pad - fw, cy + tile_size//2 - font.get_height()//2))            # R
            blit_val(q[2], (cx + tile_size//2 - fw//2, cy + pad))                                            # U
            blit_val(q[3], (cx + tile_size//2 - fw//2, cy + tile_size - font.get_height() - pad))           # D

def draw_path_overlay(path, ox, oy, tile_size, color=PATH_YELLOW, width=3):
    if not path or len(path) < 2: return
    pts = []
    for (x,y) in path:
        cx = ox + x*tile_size + tile_size//2
        cy = oy + y*tile_size + tile_size//2
        pts.append((cx,cy))
    pygame.draw.lines(SCREEN, color, False, pts, width)
    sx,sy = pts[0]; gx,gy = pts[-1]
    pygame.draw.circle(SCREEN, color, (sx,sy), max(3, tile_size//6))
    pygame.draw.circle(SCREEN, color, (gx,gy), max(4, tile_size//5), width=2)

def draw_training(mouse_pos, controls, hud, sliders_panel, font_slider, alpha_slider):
    (play_btn, reset_btn, stop_btn, qviz_btn, show_first_btn, show_best_btn, clear_path_btn) = controls
    SCREEN.fill(BLACK)
    tile_size, gpw, gph, ox, oy = compute_main_grid_layout()
    # grid
    for x in range(0, gpw+1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (ox+x, oy), (ox+x, oy+gph))
    for y in range(0, gph+1, tile_size):
        pygame.draw.line(SCREEN, WHITE, (ox, oy+y), (ox+gpw, oy+y))
    # visuals
    for gy in range(gridHeight):
        for gx in range(gridWidth):
            c = map_data[gy][gx]
            if c == Cell.NONE: continue
            cell_rect = pygame.Rect(ox + gx*tile_size, oy + gy*tile_size, tile_size, tile_size)
            draw_cell_visual(SCREEN, cell_rect, c)

    # overlays / agent
    if hud["show_first"] and hud["first_path"]:
        draw_path_overlay(hud["first_path"], ox, oy, tile_size, color=PATH_YELLOW, width=4)
    if hud["show_best"] and hud["best_path"]:
        draw_path_overlay(hud["best_path"], ox, oy, tile_size, color=PATH_PURPLE, width=4)
    if hud["agent_pos"] is not None and not (hud["show_first"] or hud["show_best"]):
        ax, ay = hud["agent_pos"]
        rect = pygame.Rect(ox + ax*tile_size, oy + ay*tile_size, tile_size, tile_size)
        pygame.draw.rect(SCREEN, TEAL, rect.inflate(-int(tile_size*0.25), -int(tile_size*0.25)), border_radius=6)

    # Q overlay
    if qviz_show and Q is not None:
        draw_q_overlay(Q, tile_size, ox, oy)

    # HUD text
    info_y = max(10, oy - int(SCREEN_HEIGHT*0.07))
    text = f"Episode: {hud['episode']}/{EPISODES}   Step: {hud['step']}/{max_steps}   Reward: {hud['reward']:.1f}   Eps: {epsilon:.3f}   Alpha: {alpha:.3f}   Finishes: {hud['finishes']}"
    SCREEN.blit(LBL_FONT.render(text, True, WHITE), (int(SCREEN_WIDTH*0.05), info_y))

    # buttons (two rows)
    draw_button(play_btn,  "Pause" if hud["running"] else "Play", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)
    draw_button(reset_btn, "Reset Q", mouse_pos, base=GREY, hover=LIGHT_GREY)
    draw_button(stop_btn,  "Stop",    mouse_pos, base=RED,  hover=(220,90,90))
    draw_button(qviz_btn,  "Show Q-values" if not qviz_show else "Hide Q-values", mouse_pos, base=GREY, hover=LIGHT_GREY, active=qviz_show)

    draw_button(show_first_btn, "Show First Path", mouse_pos, base=GREY, hover=LIGHT_GREY, active=hud["show_first"])
    draw_button(show_best_btn,  "Show Optimal Path", mouse_pos, base=GREY, hover=LIGHT_GREY, active=hud["show_best"])
    draw_button(clear_path_btn, "Clear Path Overlay", mouse_pos, base=GREY, hover=LIGHT_GREY)

    # sliders panel (only for Q overlay)
    if qviz_show:
        pygame.draw.rect(SCREEN, (30,30,30), sliders_panel, border_radius=12)
        pygame.draw.rect(SCREEN, WHITE, sliders_panel, 1, border_radius=12)
        font_slider.draw("Font", fmt=lambda v: f"{int(v)} px")
        alpha_slider.draw("Alpha", fmt=lambda v: f"{int(v)}")

def draw_gridmap(mouse_pos, dt, L):
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, GREY, L["panel"], border_radius=18)
    pygame.draw.rect(SCREEN, WHITE, L["panel"], 1, border_radius=18)
    t = TITLE_FONT.render("Grid Map", True, WHITE)
    SCREEN.blit(t, t.get_rect(center=(L["panel"].centerx, L["panel"].y + int(L["panel"].h*0.09))))

    # inputs W/H
    width_input.set_rect(L["width"]);  height_input.set_rect(L["height"])
    width_input.update(dt); height_input.update(dt)
    SCREEN.blit(LBL_FONT.render("Width", True, WHITE),  (L["width"].x,  L["width"].y - 22))
    SCREEN.blit(LBL_FONT.render("Height", True, WHITE), (L["height"].x, L["height"].y - 22))
    width_input.draw(); height_input.draw()

    # preview
    cols = max(1, width_input.get_int(edit_w))
    rows = max(1, height_input.get_int(edit_h))
    pr = L["preview"]
    pygame.draw.rect(SCREEN, DARK_GREY, pr, border_radius=12)
    pygame.draw.rect(SCREEN, WHITE, pr, 1, border_radius=12)
    cell_size = max(6, min(pr.w//cols, pr.h//rows))
    preview_w = cell_size*cols; preview_h = cell_size*rows
    origin = pygame.Rect(0,0, preview_w, preview_h); origin.center = pr.center
    draw_grid_lines(SCREEN, origin, cell_size, cols, rows)

    for gy in range(min(rows, edit_h)):
        for gx in range(min(cols, edit_w)):
            c = edit_map[gy][gx]
            if c == Cell.NONE: continue
            cell_rect = pygame.Rect(origin.x + gx*cell_size, origin.y + gy*cell_size, cell_size, cell_size)
            draw_cell_visual(SCREEN, cell_rect, c)

    # tools
    SCREEN.blit(LBL_FONT.render("Tools", True, WHITE), (L["t_start"].x, L["t_start"].y - 24))
    draw_button(L["t_start"],  "Start",    mouse_pos, active=(current_tool==Cell.START))
    draw_button(L["t_finish"], "Finish",   mouse_pos, active=(current_tool==Cell.FINISH))
    draw_button(L["t_obst"],   "Obstacle", mouse_pos, active=(current_tool==Cell.OBSTACLE))
    draw_button(L["t_trap"],   "Trap",     mouse_pos, active=(current_tool==Cell.TRAP))

    # save/load
    save_name_input.set_rect(L["save_name"])
    save_name_input.update(dt)
    SCREEN.blit(LBL_FONT.render("Save Name", True, WHITE), (L["save_name"].x, L["save_name"].y - 22))
    save_name_input.draw()
    draw_button(L["save_btn"], "Save Map", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    # position only here; draw later on top
    load_dropdown.set_rect(L["load_dd"])
    SCREEN.blit(SMALL_FONT.render("Pick to load", True, LIGHT_GREY), (L["load_dd"].x, L["load_dd"].bottom + 6))

    # erase/done
    draw_button(L["erase"], "Erase All", mouse_pos, base=LIGHT_GREY, hover=WHITE, text_color=BLACK)
    draw_button(L["done"],  "Done",      mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    hint = SMALL_FONT.render("Left click = paint, Right click = erase (1..4 to switch tool)", True, LIGHT_GREY)
    SCREEN.blit(hint, hint.get_rect(midtop=(origin.centerx, origin.bottom + 10)))

    # TOP-LAYER: draw dropdown last so it stays above everything
    load_dropdown.draw(mouse_pos)

    return origin, cell_size

def layout_options():
    pw = int(SCREEN_WIDTH*0.92); ph=int(SCREEN_HEIGHT*0.86)
    panel = pygame.Rect((SCREEN_WIDTH-pw)//2, (SCREEN_HEIGHT-ph)//2, pw, ph)
    pad = int(min(pw, ph)*0.03)
    col_w = (pw - pad*3) // 2
    left_col  = pygame.Rect(panel.x+pad, panel.y+int(ph*0.18), col_w, ph-int(ph*0.30))
    right_col = pygame.Rect(left_col.right+pad, left_col.y, col_w, left_col.h)

    # NEW: algorithm dropdown rect under the title
    algo_rect = pygame.Rect(panel.x+pad, panel.y+int(ph*0.12), int(pw*0.28), int(ph*0.075))

    # bottom row: Save/Load settings + Done
    row_h = int(ph*0.085)
    btn_y = panel.bottom - row_h - int(ph*0.06)
    save_settings_btn = pygame.Rect(panel.x+pad, btn_y, int(pw*0.22), row_h)
    load_settings_btn = pygame.Rect(save_settings_btn.right + pad, btn_y, int(pw*0.22), row_h)
    done_btn = pygame.Rect(panel.right - pad - int(pw*0.22), btn_y, int(pw*0.22), row_h)

    def row_rect(col, i): 
        return pygame.Rect(col.x+10, col.y+i*int(ph*0.055), col.w-20, int(ph*0.055)-10)

    return dict(panel=panel, left=left_col, right=right_col, row=row_rect,
                save_btn=save_settings_btn, load_btn=load_settings_btn, done=done_btn,
                algo=algo_rect)

def draw_options(mouse_pos, dt, L):
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, GREY, L["panel"], border_radius=18)
    pygame.draw.rect(SCREEN, WHITE, L["panel"], 1, border_radius=18)
    t = TITLE_FONT.render("Options", True, WHITE)
    SCREEN.blit(t, t.get_rect(center=(L["panel"].centerx, L["panel"].y + int(L["panel"].h*0.09))))

    # NEW: Algorithm selector
    SCREEN.blit(LBL_FONT.render("Algorithm", True, WHITE), (L["algo"].x, L["algo"].y - 22))
    algorithm_dropdown.set_rect(L["algo"])

    ensure_option_inputs(L)
    assign_option_rects(L)

    pygame.draw.rect(SCREEN, DARK_GREY, L["left"],  border_radius=10)
    pygame.draw.rect(SCREEN, DARK_GREY, L["right"], border_radius=10)

    for key, ib in opt_inputs.items():
        ib.update(dt)
        SCREEN.blit(LBL_FONT.render(ib._label_text, True, WHITE), (ib._label_rect.x, ib._label_rect.y + (ib._label_rect.h-22)//2))
        ib.draw()

    draw_button(L["save_btn"], "Save Settings", mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)
    draw_button(L["load_btn"], "Load Settings", mouse_pos, base=GREY, hover=LIGHT_GREY)
    draw_button(L["done"],     "Save & Back",   mouse_pos, base=BLUE, hover=WHITE, text_color=BLACK)

    # TOP-LAYER: draw dropdown last so it stays above everything
    algorithm_dropdown.draw(mouse_pos)

# =========================
#   INPUTS / OPTION INPUTS
# =========================
width_input  = InputBox((0,0,10,10), text=str(gridWidth),  numeric=True, float_ok=False, minv=1)
height_input = InputBox((0,0,10,10), text=str(gridHeight), numeric=True, float_ok=False, minv=1)
save_name_input = InputBox((0,0,10,10), text="", numeric=False)
current_tool = Cell.START

opt_inputs = {}
def ensure_option_inputs(L):
    global opt_inputs
    def ib(val, float_ok=True, minv=None, maxv=None):
        return InputBox((0,0,10,10), text=str(val), numeric=True, float_ok=float_ok, minv=minv, maxv=maxv)
    if opt_inputs: return
    # Rewards
    opt_inputs["start_visual"]   = ib(reward_start_visual, float_ok=True)
    opt_inputs["finish"]         = ib(reward_finish, float_ok=True)
    opt_inputs["obstacle_block"] = ib(reward_obstacle_block, float_ok=True)
    opt_inputs["trap"]           = ib(reward_trap, float_ok=True)
    opt_inputs["valid_move"]     = ib(reward_valid_move, float_ok=True)
    opt_inputs["invalid_move"]   = ib(reward_invalid_move, float_ok=True)
    # Hyperparams
    opt_inputs["alpha"]     = ib(alpha,   float_ok=True, minv=0.0)
    opt_inputs["gamma"]     = ib(gamma,   float_ok=True, minv=0.0)
    opt_inputs["epsilon"]   = ib(epsilon, float_ok=True, minv=0.0)
    opt_inputs["mi"]        = ib(mi,      float_ok=True, minv=0.0)
    opt_inputs["beta"]      = ib(beta,    float_ok=True, minv=0.0)
    opt_inputs["EPISODES"]  = InputBox((0,0,10,10), text=str(EPISODES),  numeric=True, float_ok=False, minv=1)
    opt_inputs["max_steps"] = InputBox((0,0,10,10), text=str(max_steps), numeric=True, float_ok=False, minv=1)
    # FPS
    opt_inputs["FPS"]       = InputBox((0,0,10,10), text=str(FPS), numeric=True, float_ok=False, minv=1, maxv=500)

def assign_option_rects(L):
    rows_left = ["start_visual","finish","obstacle_block","trap","valid_move","invalid_move"]
    rows_right= ["alpha","gamma","epsilon","mi","beta","EPISODES","max_steps","FPS"]
    def place(col, idx, key):
        r = L["row"](col, idx)
        label_rect = pygame.Rect(r.x, r.y, int(r.w*0.55), r.h)
        input_rect = pygame.Rect(label_rect.right+8, r.y, r.w - (label_rect.w+8), r.h)
        opt_inputs[key].set_rect(input_rect)
        opt_inputs[key]._label_rect = label_rect
        opt_inputs[key]._label_text = key.replace("_"," ").title()
    for i,k in enumerate(rows_left):  place(L["left"], i, k)
    for i,k in enumerate(rows_right): place(L["right"], i, k)

def inputs_to_settings_dict():
    return {
        "alpha": float(opt_inputs["alpha"].get_float(alpha)),
        "gamma": float(opt_inputs["gamma"].get_float(gamma)),
        "epsilon": float(opt_inputs["epsilon"].get_float(epsilon)),
        "mi": float(opt_inputs["mi"].get_float(mi)),
        "beta": float(opt_inputs["beta"].get_float(beta)),
        "EPISODES": int(opt_inputs["EPISODES"].get_int(EPISODES)),
        "max_steps": int(opt_inputs["max_steps"].get_int(max_steps)),
        "FPS": int(opt_inputs["FPS"].get_int(FPS)),
        "reward_start_visual": float(opt_inputs["start_visual"].get_float(reward_start_visual)),
        "reward_finish": float(opt_inputs["finish"].get_float(reward_finish)),
        "reward_obstacle_block": float(opt_inputs["obstacle_block"].get_float(reward_obstacle_block)),
        "reward_trap": float(opt_inputs["trap"].get_float(reward_trap)),
        "reward_valid_move": float(opt_inputs["valid_move"].get_float(reward_valid_move)),
        "reward_invalid_move": float(opt_inputs["invalid_move"].get_float(reward_invalid_move)),
        "algorithm": (algorithm_dropdown.items[algorithm_dropdown.selected]
                      if 0 <= algorithm_dropdown.selected < len(algorithm_dropdown.items)
                      else algorithm),  # NEW
    }

def apply_options():  # from inputs -> globals
    apply_settings_dict(inputs_to_settings_dict())

# =========================
#   VISUAL TRAINING STATE
# =========================
training_running = False
Q = None
episode_idx = 0
step_idx = 0
num_finishes = 0
firstFind = False
firstEpisode = -1
first_finish_path = []
best_path = []
show_first_overlay = False
show_best_overlay = False

# snapshot of settings when entering Training (to restore on STOP)
training_settings_snapshot = None

def training_reset():
    global Q, episode_idx, step_idx, num_finishes, firstFind, firstEpisode, training_running
    global first_finish_path, best_path, show_first_overlay, show_best_overlay
    Q = alloc_q()
    episode_idx = 0
    step_idx = 0
    num_finishes = 0
    firstFind = False
    firstEpisode = -1
    first_finish_path = []
    best_path = []
    show_first_overlay = False
    show_best_overlay = False
    training_running = False
    if start_pos is not None:
        agent.activeState = list(start_pos)
    agent.activeReward = 0

def epsilon_greedy_action(state):
    x,y = state
    global epsilon
    if random.random() < epsilon:
        return random.choice(agent.actionOptions)
    return int(np.argmax(Q[y,x]))

_current_action = None
episode_path = []

def training_finished():
    return episode_idx >= EPISODES

def path_from_Q(start, goal, limit=None):
    if start is None or goal is None: return []
    if limit is None: limit = gridWidth*gridHeight*2
    path = [tuple(start)]
    cur = list(start)
    for _ in range(limit):
        x,y = cur
        a = int(np.argmax(Q[y,x]))
        nx, ny = agent._next_from(cur, a)
        tried = set()
        # avoid blocked/bounds
        while not (0 <= nx < gridWidth and 0 <= ny < gridHeight) or map_data[ny][nx] == Cell.OBSTACLE:
            tried.add(a)
            if len(tried) == 4:
                return path
            qs = Q[y,x].copy()
            for t in tried: qs[t] = -1e18
            a = int(np.argmax(qs))
            nx, ny = agent._next_from(cur, a)
        cur = [nx, ny]
        if tuple(cur) in path: break
        path.append(tuple(cur))
        if tuple(cur) == tuple(goal): break
    return path

def compute_best_path():
    global best_path
    best_path = path_from_Q(start_pos, finish_pos)

def training_step():
    global episode_idx, step_idx, epsilon, alpha, num_finishes, firstFind, firstEpisode, _current_action, episode_path
    if start_pos is None or finish_pos is None:
        return
    if step_idx == 0 and _current_action is None:
        agent.activeState = list(start_pos)
        agent.activeReward = 0
        episode_path = [tuple(agent.activeState)]
        _current_action = epsilon_greedy_action(agent.activeState)

    prev_state = list(agent.activeState)
    reward, next_state = agent.ProcessNextAction(_current_action)
    next_action = epsilon_greedy_action(next_state)

    px, py = prev_state
    nx, ny = next_state

    # NEW: support SARSA and Q-Learning targets
    if algorithm == "Q-Learning":
        best_next = np.max(Q[ny, nx])
        target = reward + gamma * best_next
    else:  # default SARSA
        target = reward + gamma * Q[ny, nx, next_action]

    Q[py, px, _current_action] += alpha * (target - Q[py, px, _current_action] - (step_idx * mi))

    agent.activeState = next_state
    _current_action = next_action
    step_idx += 1
    episode_path.append(tuple(next_state))

    epsilon = abs(math.sin(episode_idx * math.pi * beta)) / 2

    reached_goal = (tuple(agent.activeState) == tuple(finish_pos))
    if reached_goal or step_idx >= max_steps:
        if reached_goal:
            num_finishes += 1
            if alpha > 0.1: alpha -= 0.0045
            if not firstFind:
                firstFind=True; firstEpisode = episode_idx
                compact=[]
                for p in episode_path:
                    if not compact or compact[-1]!=p: compact.append(p)
                globals()['first_finish_path'] = compact
        episode_idx += 1
        step_idx = 0
        _current_action = None
        episode_path = []
        if training_finished():
            compute_best_path()

# =========================
#          MAIN LOOP
# =========================
running = True
Q = alloc_q()
load_dropdown.set_items(list_grid_names())

# Sliders for Q overlay
font_slider = Slider((0,0,100,20), 10, 36, qviz_font_px, integer=True)
alpha_slider= Slider((0,0,100,20), 40, 255, qviz_alpha, integer=True)

while running:
    dt = clock.tick(FPS)
    events = pygame.event.get()
    mouse_pos = pygame.mouse.get_pos()

    for e in events:
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.VIDEORESIZE:
            SCREEN_WIDTH, SCREEN_HEIGHT = max(700, e.w), max(480, e.h)
            SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

    # scene enter hooks
    if prev_state != state and state == Scene.GRIDMAP:
        init_edit_from_runtime()
        width_input.text = str(edit_w); height_input.text = str(edit_h)
        save_name_input.text = ""
        load_dropdown.set_items(list_grid_names())

    if prev_state != state and state == Scene.TRAINING:
        # snapshot current settings to restore on STOP
        training_settings_snapshot = current_settings_dict()
        Q = np.zeros((gridHeight, gridWidth, len(agent.actionOptions)), dtype=float)
        training_reset()

    if prev_state != state and state == Scene.OPTIONS:
        opt_inputs.clear()  # rebuild next draw
        _sync_algorithm_dropdown()          # NEW: reflect current algo in dropdown
        algorithm_dropdown.open = False     # NEW: ensure closed when entering
        algorithm_dropdown.changed = False  # NEW

    prev_state = state

    if state == Scene.MENU:
        start_btn, gridmap_btn, options_btn = layout_menu()
        # events
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if start_btn.collidepoint(e.pos):   state = Scene.TRAINING
                elif gridmap_btn.collidepoint(e.pos): state = Scene.GRIDMAP
                elif options_btn.collidepoint(e.pos): state = Scene.OPTIONS
        # draw
        draw_menu(mouse_pos, start_btn, gridmap_btn, options_btn)

    elif state == Scene.TRAINING:
        (play_btn, reset_btn, stop_btn, qviz_btn,
         show_first_btn, show_best_btn, clear_path_btn,
         sliders_panel, font_rect, alpha_rect) = layout_training()
        font_slider.set_rect(font_rect); alpha_slider.set_rect(alpha_rect)

        # events
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    # back to menu without resetting settings; ESC behaves like simple back
                    state = Scene.MENU
                if e.key == pygame.K_SPACE:
                    training_running = not training_running
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if play_btn.collidepoint(e.pos):   training_running = not training_running
                elif reset_btn.collidepoint(e.pos): training_reset()
                elif stop_btn.collidepoint(e.pos):
                    # STOP: restore settings snapshot and go menu
                    if training_settings_snapshot:
                        apply_settings_dict(training_settings_snapshot)
                    training_reset()
                    state = Scene.MENU
                elif qviz_btn.collidepoint(e.pos):
                    qviz_show = not qviz_show
                elif show_first_btn.collidepoint(e.pos):
                    if training_finished():
                        show_first_overlay = not show_first_overlay
                        if show_first_overlay: show_best_overlay = False
                elif show_best_btn.collidepoint(e.pos):
                    if training_finished():
                        show_best_overlay = not show_best_overlay
                        if show_best_overlay: show_first_overlay = False
                elif clear_path_btn.collidepoint(e.pos):
                    show_first_overlay = False
                    show_best_overlay = False

            if qviz_show:
                font_slider.handle_event(e)
                alpha_slider.handle_event(e)
                qviz_font_px = int(font_slider.value)
                qviz_alpha   = int(alpha_slider.value)

        # progress training
        if training_running and not training_finished():
            training_step()

        hud = {
            "episode": episode_idx,
            "step": step_idx,
            "reward": agent.activeReward,
            "running": training_running and not training_finished(),
            "finishes": num_finishes,
            "agent_pos": tuple(agent.activeState) if start_pos is not None else None,
            "first_path": first_finish_path,
            "best_path": best_path,
            "show_first": show_first_overlay,
            "show_best": show_best_overlay
        }
        draw_training(mouse_pos,
                      (play_btn, reset_btn, stop_btn, qviz_btn, show_first_btn, show_best_btn, clear_path_btn),
                      hud, sliders_panel, font_slider, alpha_slider)

        if (start_pos is None or finish_pos is None):
            msg = "Set Start and Finish in GridMap to run training."
            overlay = TITLE_FONT.render(msg, True, RED)
            SCREEN.blit(overlay, overlay.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))

    elif state == Scene.GRIDMAP:
        L = layout_gridmap()
        origin, cell_size = draw_gridmap(mouse_pos, dt, L)

        # events
        # live resize from inputs
        tw = max(1, width_input.get_int(edit_w))
        th = max(1, height_input.get_int(edit_h))
        if (tw, th) != (edit_w, edit_h):
            # resize buffer
            new_arr = [[Cell.NONE for _ in range(tw)] for _ in range(th)]
            for y in range(min(th, edit_h)):
                for x in range(min(tw, edit_w)):
                    new_arr[y][x] = edit_map[y][x]
            edit_map[:] = new_arr
            globals()['edit_w'], globals()['edit_h'] = tw, th

        for e in events:
            width_input.handle_event(e); height_input.handle_event(e); save_name_input.handle_event(e)
            load_dropdown.handle_event(e)  # supports wheel scrolling now
            if load_dropdown.changed:
                # Load once per new selection
                name = load_dropdown.items[load_dropdown.selected]
                load_grid(name)
                init_edit_from_runtime()
                width_input.text = str(edit_w)
                height_input.text = str(edit_h)
                load_dropdown.changed = False

            input_focus = width_input.has_focus() or height_input.has_focus() or save_name_input.has_focus()

            if e.type == pygame.KEYDOWN and not input_focus:
                if e.key == pygame.K_1: current_tool = Cell.START
                elif e.key == pygame.K_2: current_tool = Cell.FINISH
                elif e.key == pygame.K_3: current_tool = Cell.OBSTACLE
                elif e.key == pygame.K_4: current_tool = Cell.TRAP
                elif e.key == pygame.K_ESCAPE: state = Scene.MENU

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if L["t_start"].collidepoint(e.pos):  current_tool = Cell.START;  continue
                if L["t_finish"].collidepoint(e.pos): current_tool = Cell.FINISH; continue
                if L["t_obst"].collidepoint(e.pos):   current_tool = Cell.OBSTACLE;continue
                if L["t_trap"].collidepoint(e.pos):   current_tool = Cell.TRAP;    continue

                if L["erase"].collidepoint(e.pos):
                    for y in range(edit_h):
                        for x in range(edit_w):
                            edit_map[y][x] = Cell.NONE
                    globals()['edit_start'] = None
                    globals()['edit_finish'] = None
                    continue

                if L["save_btn"].collidepoint(e.pos):
                    name = save_name_input.text.strip()
                    apply_edit_to_runtime()
                    save_current_grid(name)
                    load_dropdown.set_items(list_grid_names())
                    continue

                if L["done"].collidepoint(e.pos):
                    apply_edit_to_runtime()
                    training_reset()
                    state = Scene.MENU
                    continue

        # painting
        if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            if e.type == pygame.MOUSEMOTION and not (e.buttons[0] or e.buttons[2]): 
                pass
            else:
                mx,my = e.pos
                if origin.collidepoint(mx,my):
                    gx = (mx - origin.x)//cell_size
                    gy = (my - origin.y)//cell_size
                    if 0 <= gx < edit_w and 0 <= gy < edit_h:
                        if (e.type == pygame.MOUSEBUTTONDOWN and e.button == 3) or (e.type == pygame.MOUSEMOTION and e.buttons[2]):
                            if edit_start == (gx,gy): globals()['edit_start']=None
                            if edit_finish== (gx,gy): globals()['edit_finish']=None
                            edit_map[gy][gx] = Cell.NONE
                        elif (e.type == pygame.MOUSEBUTTONDOWN and e.button == 1) or (e.type == pygame.MOUSEMOTION and e.buttons[0]):
                            if current_tool == Cell.START:
                                if edit_start:
                                    sx,sy = edit_start; edit_map[sy][sx] = Cell.NONE
                                globals()['edit_start']=(gx,gy); edit_map[gy][gx]=Cell.START
                            elif current_tool == Cell.FINISH:
                                if edit_finish:
                                    fx,fy = edit_finish; edit_map[fy][fx] = Cell.NONE
                                globals()['edit_finish']=(gx,gy); edit_map[gy][gx]=Cell.FINISH
                            elif current_tool == Cell.OBSTACLE:
                                if edit_start == (gx,gy): globals()['edit_start']=None
                                if edit_finish== (gx,gy): globals()['edit_finish']=None
                                edit_map[gy][gx]=Cell.OBSTACLE
                            elif current_tool == Cell.TRAP:
                                if edit_start == (gx,gy): globals()['edit_start']=None
                                if edit_finish== (gx,gy): globals()['edit_finish']=None
                                edit_map[gy][gx]=Cell.TRAP