# RL Maze & Path-Finding Visual Simulator
**ENSIA — Reinforcement Learning Project 2025/2026**  
Team: Imene Fatma Djelili · Amira Mereddef · Aya Khaldi · Lina Wafaa Bentiba · Melynda Hadj Ali


---

## Installation

```bash
pip install -r requirements.txt
```

## Running the Test

```bash
python test_env.py
```

## Maze Environment (`env/maze_env.py`)

The `MazeEnv` class provides a standard RL interface:

| Method | Description |
|---|---|
| `reset()` | Starts a new episode, returns initial state |
| `step(action)` | Takes action, returns `(next_state, reward, done)` |
| `render()` | Prints the maze to the terminal |
| `num_states` | Total states (grid size²) |
| `num_actions` | 4 (UP, DOWN, LEFT, RIGHT) |

### Reward Structure
- Goal reached: **+100**
- Normal step: **−1**
