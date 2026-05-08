# RL Maze & Path-Finding Visual Simulator
**ENSIA — Reinforcement Learning Project 2025/2026**  
Team: Imene Fatma Djelili · Aya Khaldi · Amira Mereddef · Lina Wafaa Bentiba · Melynda Hadj Ali

---

## Project Structure

```
RL-Maze-Project/
│
├── env/
│   └── maze_env.py         # Phase 01 — Maze environment (Imene)
│
├── agent/
│   ├── q_agent.py          # Phase 02 — Q-Learning agent (Aya)
│   ├── q_table.npy         # Saved trained Q-table ← DO NOT DELETE
│   └── q_table_stats.json  # Training history (rewards, steps, epsilon)
│
├── main.py                 # Train the agent + plot results
├── test_env.py             # Quick environment sanity check
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Phase 01 — Maze Environment (`env/maze_env.py`) — *Imene*

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

---

## Phase 02 — Q-Learning Agent (`agent/q_agent.py`) — *Aya*

### What is Q-Learning?
Q-Learning is a model-free RL algorithm. The agent learns a Q-table
(states × actions) by exploring the maze and updating values using:

```
Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') − Q(s,a)]
```

### Hyperparameters (set in `main.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.95 | Discount factor (future rewards) |
| `epsilon` | 1.0 → 0.01 | Exploration rate (decays over episodes) |
| `num_episodes` | 2000 | Training episodes |

### Train the agent

```bash
python main.py
```

This will:
1. Train the Q-Learning agent for 2000 episodes
2. Print progress every 200 episodes
3. Evaluate on 20 episodes and print success rate
4. Save `agent/q_table.npy` and `agent/q_table_stats.json`
5. Save `agent/training_curves.png` (reward / steps / epsilon plots)

### Run a demo with the trained agent

```bash
python main.py demo
```

### Use the agent in your own code (Phase 03+)

```python
from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent

env   = MazeEnv(size=10, random_maze=False)
agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
agent.load("agent/q_table.npy")   # load trained Q-table
agent.epsilon = 0.0               # no exploration, pure exploitation

# Get the learned path as list of (row, col) positions
path = agent.get_learned_path(env)

# Or step manually
state = env.reset()
action = agent.best_action(state)
next_state, reward, done = env.step(action)
```

---

## Phase 03 — SARSA + Visualization UI (`agent/sarsa_agent.py`) — *Amira* ⬅️ TODO

---

## Phase 04 — Dashboard + Comparison — *Lina* ⬅️ TODO

---

## Phase 05 — Report + Video + Polish — *Melynda* ⬅️ TODO
