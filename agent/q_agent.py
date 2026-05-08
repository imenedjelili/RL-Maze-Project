import numpy as np
import json
import os


class QLearningAgent:
    """
    Q-Learning Agent for the MazeEnv.

    Learns a policy by maintaining a Q-table (states x actions)
    and updating it using the Bellman equation after each step.
    """

    def __init__(self, num_states, num_actions,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):
        """
        num_states    : total number of states in the environment
        num_actions   : number of possible actions
        alpha         : learning rate
        gamma         : discount factor (how much future rewards matter)
        epsilon       : initial exploration rate (1.0 = fully random)
        epsilon_min   : minimum exploration rate
        epsilon_decay : multiplicative decay applied after each episode
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: rows = states, cols = actions, initialized to 0
        self.q_table = np.zeros((num_states, num_actions))

        # Track training history for plotting
        self.rewards_history = []
        self.steps_history = []
        self.epsilon_history = []

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """
        Epsilon-greedy action selection:
        - With probability epsilon  → random action (explore)
        - With probability 1-epsilon → best known action (exploit)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def best_action(self, state):
        """Always returns the greedy best action (no exploration). Used after training."""
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        """
        Bellman update:
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, env, num_episodes=2000, max_steps=500, verbose=True):
        """
        Full training loop.

        env          : MazeEnv instance
        num_episodes : number of episodes to run
        max_steps    : max steps per episode (prevents infinite loops)
        verbose      : print progress every 200 episodes
        """
        print(f"Training Q-Learning agent for {num_episodes} episodes...\n")

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            steps = 0

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            self.decay_epsilon()

            # Record history
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)
            self.epsilon_history.append(self.epsilon)

            if verbose and episode % 200 == 0:
                avg_reward = np.mean(self.rewards_history[-200:])
                avg_steps  = np.mean(self.steps_history[-200:])
                print(f"Episode {episode:>5} | "
                      f"Avg Reward (last 200): {avg_reward:>8.1f} | "
                      f"Avg Steps: {avg_steps:>6.1f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        print("\nTraining complete!")
        return self.rewards_history, self.steps_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, num_episodes=10, max_steps=500, render=False):
        """
        Evaluate the trained agent (no exploration, epsilon=0).
        Returns average reward and success rate.
        """
        rewards = []
        successes = 0

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.best_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    successes += 1
                    break

            rewards.append(total_reward)

            if render:
                print(f"\n--- Eval Episode {episode + 1} ---")
                env.render()
                print(f"Total reward: {total_reward}")

        avg_reward = np.mean(rewards)
        success_rate = (successes / num_episodes) * 100
        print(f"\nEvaluation over {num_episodes} episodes:")
        print(f"  Avg Reward   : {avg_reward:.1f}")
        print(f"  Success Rate : {success_rate:.0f}%")
        return avg_reward, success_rate

    # ------------------------------------------------------------------
    # Extract Learned Path
    # ------------------------------------------------------------------

    def get_learned_path(self, env, max_steps=500):
        """
        Runs one greedy episode and returns the path taken by the agent.
        Returns list of (row, col) positions.
        Used by Phase 03 for visualization.
        """
        state = env.reset()
        path = [env.agent_pos]

        for _ in range(max_steps):
            action = self.best_action(state)
            next_state, reward, done = env.step(action)
            path.append(env.agent_pos)
            state = next_state
            if done:
                break

        return path

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path="agent/q_table.npy"):
        """Save Q-table and training stats to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

        # Save training history as JSON for easy plotting later
        stats_path = path.replace(".npy", "_stats.json")
        stats = {
            "rewards_history": self.rewards_history,
            "steps_history":   self.steps_history,
            "epsilon_history": self.epsilon_history,
            "hyperparams": {
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            }
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Q-table saved to {path}")
        print(f"Training stats saved to {stats_path}")

    def load(self, path="agent/q_table.npy"):
        """Load a previously saved Q-table."""
        self.q_table = np.load(path)
        print(f"Q-table loaded from {path}")