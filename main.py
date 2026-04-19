from env.maze_env import MazeEnv


def run_demo():
    """
    Quick demo: runs one full episode with a random policy.
    Replace the random action selection with your RL agent later.
    """
    import random

    env = MazeEnv(size=10, random_maze=True)
    state = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 200

    print("Starting demo episode...\n")
    env.render()

    while steps < max_steps:
        action = random.choice(env.ACTIONS)
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state

        if done:
            print(f"Goal reached in {steps} steps!")
            break

    print(f"\nEpisode finished — Steps: {steps} | Total reward: {total_reward}")
    env.render()


if __name__ == "__main__":
    run_demo()
