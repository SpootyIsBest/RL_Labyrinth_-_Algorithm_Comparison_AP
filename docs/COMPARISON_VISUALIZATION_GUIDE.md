# Comparison Visualization Mode - Implementation Guide

## Overview
The comparison visualization mode allows you to run all 6 algorithms (2 RL + 4 Non-RL) simultaneously and view their progress in real-time.

## Features Implemented

### 1. Adaptive Grid View (2x3 Layout)
- All 6 algorithms are displayed in a 2x3 grid
- Each cell shows:
  - Algorithm name
  - Scaled maze visualization
  - Agent position (red dot)
  - Current statistics (steps taken)
  - Algorithm-specific progress

### 2. Expanded View
- **Click any grid cell** to expand that algorithm to full screen
- **Press ESC** to return to grid view
- Expanded view provides detailed visualization of a single algorithm

### 3. Runtime Controls

#### Keyboard Shortcuts:
- **C** - Toggle settings overlay (shows controls and current status)
- **F** - Freeze/Unfreeze visualization (stops rendering but training continues)
- **T** - Pause/Resume training (pauses all algorithm execution)
- **ESC** - Return to grid view (when expanded) or exit comparison mode (when in grid)
- **I** - Toggle path visualization (when supported by algorithm)

#### Speed Control:
- **1** - 1x speed
- **2** - 2x speed
- **3** - 5x speed
- **4** - 10x speed
- **5** - 20x speed
- **6** - 50x speed
- **7** - 100x speed
- **8** - 200x speed
- **9** - 500x speed
- **0** - 1000x speed (maximum)

### 4. Status Indicators
When not showing the settings overlay, the top of the screen displays:
- **VIZ: RUNNING/PAUSED** - Visualization status (green/yellow)
- **TRAIN: RUNNING/PAUSED** - Training status (green/yellow)
- **Speed: Nx** - Current speed multiplier (cyan)
- **Press C for controls** - Hint text (gray)

### 5. Settings Overlay
Press **C** to open the settings overlay, which shows:
- Current speed multiplier and how to adjust it
- Complete list of keyboard controls
- Current status of visualization and training
- How to close the overlay

## Usage Flow

### Starting Comparison Mode:
1. From Main Menu → Select "Mode Selection"
2. Choose "Record EVERY Algorithm"
3. Configure comparison settings (maze size, episodes, rewards)
4. Click "Continue to Load Maze"
5. Select a maze layout
6. The comparison visualization starts automatically

### During Comparison:
- Watch all 6 algorithms solve the maze simultaneously
- Click any algorithm to see it in detail
- Press **F** to speed up training by freezing visualization
- Press **T** to pause and examine current state
- Adjust speed with number keys (1-0)
- Press **C** anytime to see controls

### Performance Optimization:
The **F key (freeze visualization)** is particularly useful for:
- Speeding up training significantly (10-100x faster)
- Allowing RL algorithms to explore without rendering overhead
- Reducing CPU usage while algorithms train

### Completion:
- When all algorithms finish, results are automatically collected
- The system returns to the Main Menu
- Comprehensive comparison data is saved to JSON

## Technical Details

### Parallel Execution:
- Each RL algorithm has its own:
  - Agent instance
  - Q-table
  - Training coroutine
- Each Non-RL algorithm has its own:
  - Visualizer instance
  - Path tracking

### Grid Layout:
- Screen divided into 2 rows × 3 columns
- Each cell: ~427×360 pixels (on 1280×720 screen)
- Mazes are automatically scaled to fit

### Expanded View:
- Selected algorithm uses full screen (1280×720)
- Larger visualization for detailed inspection
- All other algorithms continue running in background

## Algorithms Displayed:
1. **Q-Learning** (RL)
2. **SARSA** (RL)
3. **BFS** (Non-RL)
4. **Wall Follower** (Non-RL)
5. **Random Walk** (Non-RL)
6. **Greedy Best-First** (Non-RL)

## Tips:
- Use speed multipliers (6-0 keys) for faster training
- Press **F** during training to maximize speed without visualization
- Click algorithms to inspect their behavior in detail
- Press **C** if you forget any controls
- Use **T** to pause and study the current state

## Future Enhancements:
- Result collection and comprehensive JSON export
- Real-time performance comparison charts
- Configurable grid layouts (2×2, 3×3, etc.)
- Custom algorithm selection for comparison
