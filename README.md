# Robot Pathfinding Course Design

This repository contains a course design project on robot obstacle avoidance and path planning.

The project focuses on grid-based path planning and compares multiple classical and learning-based methods, including:

- BFS
- A*
- Improved A*
- Dijkstra
- Q-learning
- DQN-based deep reinforcement learning

## Project Overview

The task is to plan a feasible path for a robot from a start point to a goal point while avoiding obstacles on grid maps of different difficulty levels.

The implementation includes:

- basic map and challenge map experiments
- shortest path comparison among classical graph search algorithms
- dynamic visualization of the search process
- reinforcement learning path planning attempts
- deep reinforcement learning extension for more complex maps
- course report materials based on TongjiThesis

## Main Files

- `pathfinder.py`
  Main experiment script containing BFS, A*, Improved A*, Q-learning, visualization, and map generation.

- `Dijikstra.py`
  Dijkstra algorithm implementation and comparison experiments.

- `deep_rl_pathfinder.py`
  DQN-based path planning script used for neural-network experiments on larger grid maps.

- `q_learning.py`
  Independent Q-learning related code.

- `apf.py`
  Artificial potential field related exploratory code.

- `TongjiThesis-master/`
  LaTeX report template and report-related materials.

## Generated Results

The repository also includes generated experiment outputs, such as:

- search-process GIF animations
- comparison figures
- training curves
- DQN result images
- trained model weights

## Environment

Recommended Python environment:

- Python 3.10 or above
- `numpy`
- `matplotlib`
- `pillow`
- `torch` for DQN experiments

Example installation:

```bash
pip install numpy matplotlib pillow torch
```

## Running the Code

Run the main classical pathfinding experiments:

```bash
python pathfinder.py
```

Run Dijkstra experiments:

```bash
python Dijikstra.py
```

Run deep reinforcement learning experiments:

```bash
python deep_rl_pathfinder.py
```

## Notes

- The file name `Dijikstra.py` is preserved intentionally to stay consistent with the existing project structure.
- Some generated result files are kept in the repository for direct course report use.
- The DQN part is an exploratory extension and may require parameter tuning or GPU acceleration for larger maps.

## Repository Purpose

This repository is mainly used for:

- course design development
- experiment record keeping
- cross-device synchronization between macOS and Windows
- report writing and result organization
