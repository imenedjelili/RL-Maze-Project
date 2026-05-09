# RL Maze & Path-Finding Visual Simulator
**ENSIA — Reinforcement Learning Project 2025/2026**  
Team: Imene Fatma Djelili · Aya Khaldi · Amira Mereddef · Lina Wafaa Bentiba · Melynda Hadj Ali

---

## Project Overview

An interactive reinforcement learning simulator that visualizes Q-Learning and SARSA agents solving mazes in real time. Agents are trained on procedurally generated mazes using tabular RL methods, with full training curves, side-by-side comparison, and GIF export.

---

## Project Structure

```
RL-Maze-Project/
│
├── env/
│   └── maze_env.py              # Phase 01 — Maze environment (Imene)
│
├── agent/
│   ├── q_agent.py               # Phase 02 — Q-Learning agent (Aya)
│   ├── sarsa_agent.py           # Phase 03 — SARSA agent (Amira)
│   ├── q_table.npy              # Saved trained Q-table ← DO NOT DELETE
│   ├── sarsa_table.npy          # Saved trained SARSA table ← DO NOT DELETE
│   ├── q_table_stats.json       # Q-Learning training history
│   ├── training_curves.png      # Q-Learning training plots
│   └── sarsa_training_curves.png # SARSA training plots
│
├── visualizer/
│   └── maze_visualizer.py       # Phase 03 — Pygame visualization UI (Amira)
│
├── main.py                      # Train agents + plot results
├── visualize.py                 # Launch the visual simulator
├── dashboard.py                 # Phase 04 — Comparison dashboard (Lina)
├── dashboard.html               # Phase 04 — HTML report dashboard (Lina)
├── export_gif.py                # Phase 04 — GIF export utility (Lina)
├── test_env.py                  # Quick environment sanity check
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/imenedjelili/RL-Maze-Project.git
cd RL-Maze-Project
pip install -r requirements.txt
```

---

## How to Run the Full Project

### Step 1 — Sanity check the environment
```bash
python test_env.py
```

### Step 2 — Train Q-Learning agent
```bash
python main.py
```

### Step 3 — Train SARSA agent
```bash
python main.py sarsa
```

### Step 4 — Launch the visual simulator (Pygame)
```bash
python visualize.py
```

### Step 5 — Open the comparison dashboard
```bash
python dashboard.py
# Then open dashboard.html in your browser
```

### Step 6 — Export a GIF of the agent solving the maze
```bash
python export_gif.py
```

---

## Phase 01 — Maze Environment (`env/maze_env.py`) — *Imene*

The `MazeEnv` class provides a standard RL interface for a 10×10 grid maze. The maze is generated using the **Recursive Backtracker (DFS)** algorithm, which guarantees a solvable maze with exactly one path initially carved — then additional openings ensure the goal corner is reachable.

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

### State Encoding
Each cell `(row, col)` maps to a unique integer: `state = row × size + col`

---

## Phase 02 — Q-Learning Agent (`agent/q_agent.py`) — *Aya*

### What is Q-Learning?
Q-Learning is an **off-policy**, model-free TD algorithm. The agent learns a Q-table (states × actions) by exploring the maze and updating Q-values using the Bellman equation:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

### Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.95 | Discount factor (future rewards) |
| `epsilon` | 1.0 → 0.01 | Exploration rate (ε-greedy, decays over time) |
| `num_episodes` | 2000 | Training episodes |

### Train the agent
```bash
python main.py
```
This will:
1. Train the Q-Learning agent for 2000 episodes
2. Print progress every 200 episodes
3. Evaluate on 20 test episodes and report success rate
4. Save `agent/q_table.npy` and `agent/q_table_stats.json`
5. Save `agent/training_curves.png`

### Run a demo
```bash
python main.py demo
```

### Use in your own code
```python
from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent

env   = MazeEnv(size=10, random_maze=False)
agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
agent.load("agent/q_table.npy")
agent.epsilon = 0.0  # pure exploitation

path = agent.get_learned_path(env)
```

---

## Phase 03 — SARSA + Visualization UI — *Amira*

### SARSA Agent (`agent/sarsa_agent.py`)

SARSA is an **on-policy** TD algorithm. Unlike Q-Learning, the update uses the *actual* next action taken (not the greedy max):

```
Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]
```

This makes SARSA more cautious — it learns a safer policy under ε-greedy exploration.

| Feature | Q-Learning | SARSA |
|---|---|---|
| Policy type | Off-policy | On-policy |
| Update target | max Q(s', a') | Q(s', a') actually taken |
| Behavior | More aggressive | More conservative |

### Visualization UI (`visualizer/maze_visualizer.py` + `visualize.py`)

A **Pygame-based** real-time simulator that:
- Renders the maze with color-coded walls, free cells, agent, and goal
- Animates the agent step-by-step as it follows its learned policy
- Displays episode count, steps, and cumulative reward live

```bash
python visualize.py
```

---

## Phase 04 — Dashboard + Comparison + GIF Export — *Lina*

### Dashboard (`dashboard.py` / `dashboard.html`)

An HTML comparison dashboard that loads training statistics from both agents and displays:
- Side-by-side reward curves (Q-Learning vs SARSA)
- Steps-to-goal over episodes
- Epsilon decay curves
- Final performance metrics table

```bash
python dashboard.py        # generates/updates dashboard.html
# Open dashboard.html in any browser
```

### GIF Export (`export_gif.py`)

Exports an animated GIF of the trained agent navigating the maze — useful for the project video and report.

```bash
python export_gif.py
# Output: agent_demo.gif
```

---

## Phase 05 — Report + Video + Polish — *Melynda*

Final deliverables:
- Written project report (this README + PDF report)
- Demo video walkthrough
- Code cleanup, comments, and final polish

---

## Algorithm Comparison Summary

| Metric | Q-Learning | SARSA |
|---|---|---|
| Update rule | Off-policy (greedy) | On-policy (actual action) |
| Convergence | Faster on deterministic mazes | Safer, smoother convergence |
| Path quality | Optimal in deterministic settings | Near-optimal, avoids risky edges |
| Training stability | Slightly more volatile | More stable reward curve |

---

## Dependencies

```
numpy
pygame
matplotlib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Team & Timeline

| Phase | Owner | Milestone | Deliverable | Status |
|---|---|---|---|---|
| Phase 01 | Imene | Maze Environment Setup | Working Maze Environment | ✅ Done |
| Phase 02 | Aya | Q-Learning Implementation | Trained Q-Learning Agent | ✅ Done |
| Phase 03 | Amira | SARSA + Visualization UI | Animated Simulator | ✅ Done |
| Phase 04 | Lina | Dashboard + Comparison | Full Application Running | ✅ Done |
| Phase 05 | Melynda | Report + Video + Polish | Final Submission | ✅ Done |

---

*ENSIA Reinforcement Learning Course — 2025/2026*
