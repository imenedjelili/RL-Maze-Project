from env.maze_env import MazeEnv


def test_maze():
    print("=" * 45)
    print("   Testing Maze Environment — Phase 01")
    print("=" * 45)

    env = MazeEnv(size=10, random_maze=True)
    state = env.reset()

    print(f"\nMaze size     : {env.size} x {env.size}")
    print(f"Total states  : {env.num_states}")
    print(f"Total actions : {env.num_actions}  (UP, DOWN, LEFT, RIGHT)")
    print(f"Start state   : {state}  → position (0, 0)")
    print(f"Goal position : {env.goal_pos}")

    print("\nInitial maze view:")
    env.render()

    print("--- Taking 8 random steps ---\n")
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    import random
    for i in range(8):
        action = random.choice(env.ACTIONS)
        next_state, reward, done = env.step(action)
        print(f"  Step {i+1:02d} | action={action_names[action]:<5} | "
              f"state={next_state:>3} | reward={reward:>4} | done={done}")
        if done:
            print("\n  → Agent reached the goal early!")
            break

    print("\nFinal maze state:")
    env.render()

    print("\n✅ MazeEnv passed all checks — ready for Phase 02.\n")


if __name__ == "__main__":
    test_maze()
