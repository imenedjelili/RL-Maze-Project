import matplotlib.pyplot as plt
import numpy as np

from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

ENV_SIZE      = 10       # maze grid size (10x10)
NUM_EPISODES  = 2000     # training episodes
MAX_STEPS     = 500      # max steps per episode

# Q-Learning hyperparameters
ALPHA         = 0.1      # learning rate
GAMMA         = 0.95     # discount factor
EPSILON       = 1.0      # initial exploration rate
EPSILON_MIN   = 0.01     # minimum exploration rate
EPSILON_DECAY = 0.995    # epsilon decay per episode

SAVE_PATH     = "agent/q_table.npy"


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train():
    print("=" * 55)
    print("   RL Maze Project — Phase 02: Q-Learning Training")
    print("=" * 55)

    # Use fixed maze for training so the agent learns a consistent layout
    env = MazeEnv(size=ENV_SIZE, random_maze=False)

    agent = QLearningAgent(
        num_states    = env.num_states,
        num_actions   = env.num_actions,
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON,
        epsilon_min   = EPSILON_MIN,
        epsilon_decay = EPSILON_DECAY,
    )

    rewards, steps = agent.train(env, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)

    # Save trained agent
    agent.save(SAVE_PATH)

    # Evaluate
    print()
    agent.evaluate(env, num_episodes=20, render=False)

    # Show learned path
    print("\nLearned path (greedy episode):")
    path = agent.get_learned_path(env)
    env.render()
    print(f"Path length: {len(path)} steps")

    # Plot training curves
    plot_training(rewards, steps, agent.epsilon_history)

    return agent, env


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_training(rewards, steps, epsilons, window=100):
    """Plot reward, steps, and epsilon curves over training."""

    def moving_avg(data, w):
        return np.convolve(data, np.ones(w) / w, mode='valid')

    episodes = np.arange(len(rewards))
    avg_r = moving_avg(rewards, window)
    avg_s = moving_avg(steps, window)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Q-Learning Training Progress", fontsize=14, fontweight='bold')

    # Reward
    axes[0].plot(episodes, rewards, alpha=0.2, color='steelblue', label='raw')
    axes[0].plot(np.arange(len(avg_r)) + window // 2, avg_r,
                 color='steelblue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_title("Total Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Steps
    axes[1].plot(episodes, steps, alpha=0.2, color='darkorange', label='raw')
    axes[1].plot(np.arange(len(avg_s)) + window // 2, avg_s,
                 color='darkorange', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Epsilon
    axes[2].plot(epsilons, color='seagreen', linewidth=2)
    axes[2].set_title("Epsilon Decay (Exploration Rate)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("agent/training_curves.png", dpi=150, bbox_inches='tight')
    print("\nTraining curves saved to agent/training_curves.png")
    plt.show()


# -----------------------------------------------------------------------
# Demo: run a single greedy episode with a saved agent
# -----------------------------------------------------------------------

def run_demo():
    """Load a trained agent and watch it solve the maze."""
    env = MazeEnv(size=ENV_SIZE, random_maze=False)

    agent = QLearningAgent(
        num_states  = env.num_states,
        num_actions = env.num_actions,
    )
    agent.load(SAVE_PATH)
    agent.epsilon = 0.0   # pure exploitation

    print("\nRunning greedy demo episode...\n")
    state = env.reset()
    env.render()

    total_reward = 0
    for step in range(MAX_STEPS):
        action = agent.best_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            print(f"Goal reached in {step + 1} steps! Total reward: {total_reward}")
            env.render()
            break
    else:
        print("Max steps reached without solving the maze.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        train()