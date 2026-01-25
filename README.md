# RL Labyrinth – Algorithm Comparison

Compare reinforcement learning algorithms on a grid-based labyrinth environment. The project provides a simple maze, an agent, and training/visualization utilities to evaluate how different RL approaches learn to solve the maze.

## Features
- Grid-based labyrinth environment with customizable layout
- Agent abstraction and state representation
- Visualization/monitoring utilities
- CSV logging for experiments
- Legacy implementations in oldCode/ for reference

## Project Structure
- Agent.py – agent logic and policy updates
- Maze.py – environment and transition rules
- State.py – state representation
- Main.py – entry point for running experiments
- Monitors.py – logging/metrics helpers
- output.csv – sample results
- oldCode/ – archived/legacy implementations

## Requirements
- Python 3.10+ (recommended)

Install dependencies:
1. Create a virtual environment (optional but recommended)
2. Install packages:

```
pip install -r requirements.txt
```

## Usage
Run the main experiment:

```
python Main.py
```

Results are typically written to output.csv and/or displayed in the console/visualizer depending on configuration.

## Configuration
Adjust parameters directly in Main.py or related modules (e.g., learning rate, discount factor, exploration rate, number of episodes).

## Data Output
Experiment results are saved to output.csv for comparison and plotting.

## Legacy Code
The oldCode/ folder contains earlier or alternative implementations for experimentation and reference.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
NON, for now
