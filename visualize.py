import sys
import pygame
import matplotlib.pyplot as plt
import numpy as np

from env.maze_env import MazeEnv
from agent.sarsa_agent import SARSAAgent
from agent.q_agent import QLearningAgent
from visualizer.maze_visualizer import MazeVisualizer


# ── Config ────────────────────────────────────────────────────────────────
MAZE_SIZE    = 10
NUM_EPISODES = 2000
CELL_SIZE    = 60        
ANIM_DELAY   = 0.18     


# ── Helpers ───────────────────────────────────────────────────────────────

def train_sarsa(env):
    print("=" * 55)
    print("   Training SARSA Agent...")
    print("=" * 55)
    agent = SARSAAgent(
        num_states  = env.num_states,
        num_actions = env.num_actions,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
    )
    agent.train(env, num_episodes=NUM_EPISODES, verbose=True)
    agent.evaluate(env, num_episodes=20)
    agent.save("agent/sarsa_table.npy")
    return agent


def plot_training_curves(sarsa_agent):
    """Plot SARSA training curves (reward / steps / epsilon)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("SARSA Training Progress", fontsize=14, fontweight="bold")

    # Smooth helper
    def smooth(data, w=100):
        return np.convolve(data, np.ones(w) / w, mode="valid")

    # Reward
    axes[0].plot(sarsa_agent.rewards_history, alpha=0.3, color="royalblue", label="raw")
    axes[0].plot(smooth(sarsa_agent.rewards_history), color="royalblue", label="100-ep avg")
    axes[0].set_title("Total Reward per Episode")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].legend()

    # Steps
    axes[1].plot(sarsa_agent.steps_history, alpha=0.3, color="darkorange", label="raw")
    axes[1].plot(smooth(sarsa_agent.steps_history), color="darkorange", label="100-ep avg")
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Steps")
    axes[1].legend()

    # Epsilon
    axes[2].plot(sarsa_agent.epsilon_history, color="green")
    axes[2].set_title("Epsilon Decay (Exploration Rate)")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig("agent/sarsa_training_curves.png", dpi=150)
    print("\nSARSA training curves saved to agent/sarsa_training_curves.png")
    plt.show()


# ── Modes ─────────────────────────────────────────────────────────────────

def mode_default():
    """Train SARSA → show training curves → animate learned path."""
    env   = MazeEnv(size=MAZE_SIZE, random_maze=False)
    agent = train_sarsa(env)

    plot_training_curves(agent)

    path = agent.get_learned_path(env)
    print(f"\nAnimating SARSA path ({len(path)} steps)...")
    print("A Pygame window will open — watch the agent solve the maze!")

    pygame.display.set_caption("RL Maze — SARSA Agent (Phase 03)")
    viz = MazeVisualizer(env, cell_size=CELL_SIZE)
    viz.animate_path(path, title="SARSA Agent — learned path", delay=ANIM_DELAY)
    viz.close()

    print("\nDone! Check agent/sarsa_training_curves.png for the plots.")


def mode_heatmap():
    """Train SARSA → show Q-value heatmap overlay."""
    env   = MazeEnv(size=MAZE_SIZE, random_maze=False)
    agent = train_sarsa(env)

    print("\nOpening Q-value heatmap (blue=low, red=high)...")
    pygame.display.set_caption("RL Maze — SARSA Q-Value Heatmap")
    viz = MazeVisualizer(env, cell_size=CELL_SIZE)
    viz.show_heatmap(agent.q_table, title="SARSA — Q-Value Heatmap")
    viz.close()


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"

    if mode == "heatmap":
        mode_heatmap()
    else:
        mode_default()