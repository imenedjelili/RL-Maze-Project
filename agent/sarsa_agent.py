import numpy as np
import json
import os


class SARSAAgent:
    """
    SARSA Agent for the MazeEnv.
    it updates using the action it will ACTUALLY take next,
    not the theoretical best action.

    Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a') - Q(s,a)]
                                        ↑
                               actual next action (not max)
    """

    def __init__(self, num_states, num_actions,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):

        self.num_states    = num_states
        self.num_actions   = num_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

       
        self.q_table = np.zeros((num_states, num_actions))

        # Training history
        self.rewards_history = []
        self.steps_history   = []
        self.epsilon_history = []

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """Epsilon-greedy: explore randomly or exploit best known action."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def best_action(self, state):
        """Greedy best action — used after training (no exploration)."""
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Learning  
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA Bellman update:
        Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a') - Q(s,a)]

        Key difference from Q-Learning:
        - Q-Learning uses:  γ * MAX Q(s', any_action)   ← best possible
        - SARSA uses:       γ * Q(s', next_action)       ← what we'll actually do
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]

        self.q_table[state, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training Loop  
    # ------------------------------------------------------------------

    def train(self, env, num_episodes=2000, max_steps=500, verbose=True):
        """
        SARSA training loop.
        Note the structure: action is chosen at START of step,
        next_action chosen BEFORE the update (unlike Q-Learning).
        """
        print(f"Training SARSA agent for {num_episodes} episodes...\n")

        for episode in range(1, num_episodes + 1):
            state  = env.reset()
            action = self.choose_action(state)   

            total_reward = 0
            steps        = 0

            for _ in range(max_steps):
                next_state, reward, done = env.step(action)

                next_action = self.choose_action(next_state)  

                self.update(state, action, reward,
                            next_state, next_action, done)    

                state  = next_state
                action = next_action                       
                total_reward += reward
                steps        += 1

                if done:
                    break

            self.decay_epsilon()
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

        print("\nSARSA Training complete!")
        return self.rewards_history, self.steps_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, num_episodes=10, max_steps=500):
        """Evaluate trained agent — no exploration."""
        rewards   = []
        successes = 0

        for _ in range(num_episodes):
            state        = env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action               = self.best_action(state)
                next_state, reward, done = env.step(action)
                total_reward        += reward
                state                = next_state
                if done:
                    successes += 1
                    break

            rewards.append(total_reward)

        avg_reward   = np.mean(rewards)
        success_rate = (successes / num_episodes) * 100
        print(f"\nSARSA Evaluation over {num_episodes} episodes:")
        print(f"  Avg Reward   : {avg_reward:.1f}")
        print(f"  Success Rate : {success_rate:.0f}%")
        return avg_reward, success_rate

    # ------------------------------------------------------------------
    # Get Learned Path  
    # ------------------------------------------------------------------

    def get_learned_path(self, env, max_steps=500):
        """
        Runs one greedy episode, returns list of (row, col) positions.
        Same interface as QLearningAgent — visualizer works with both.
        """
        state = env.reset()
        path  = [tuple(env.agent_pos)]

        for _ in range(max_steps):
            action               = self.best_action(state)
            next_state, reward, done = env.step(action)
            path.append(tuple(env.agent_pos))
            state = next_state
            if done:
                break

        return path

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path="agent/sarsa_table.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

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

        print(f"SARSA table saved to {path}")

    def load(self, path="agent/sarsa_table.npy"):
        self.q_table = np.load(path)
        print(f"SARSA table loaded from {path}")