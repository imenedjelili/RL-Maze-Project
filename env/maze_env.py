import numpy as np
import random


class MazeEnv:
    """
    A grid-based maze environment for reinforcement learning.
    The agent starts at the top-left and tries to reach the bottom-right exit.
    """

    # Action constants
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    # Cell type constants
    FREE  = 0
    WALL  = 1

    def __init__(self, size=10, random_maze=True):
        """
        size        : grid is (size x size)
        random_maze : if True, generates a random maze; if False, uses a default layout
        """
        self.size = size
        self.random_maze = random_maze
        self.maze = None
        self.agent_pos = None
        self.start_pos = (0, 0)
        self.goal_pos  = (size - 1, size - 1)
        self._build_maze()

    # ------------------------------------------------------------------
    # Maze Generation
    # ------------------------------------------------------------------

    def _build_maze(self):
        """Creates the maze grid using Recursive Backtracker (DFS)."""
        if self.random_maze:
            self.maze = self._generate_random_maze()
        else:
            self.maze = self._default_maze()

        # Make sure start and goal are always accessible
        self.maze[self.start_pos] = self.FREE
        self.maze[self.goal_pos]  = self.FREE

    def _generate_random_maze(self):
        """
        Recursive Backtracker algorithm.
        Starts with all walls, then carves passages.
        Always produces a solvable maze.
        """
        size = self.size
        maze = np.ones((size, size), dtype=int)

        def carve(r, c):
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and maze[nr][nc] == self.WALL:
                    maze[r + dr // 2][c + dc // 2] = self.FREE
                    maze[nr][nc] = self.FREE
                    carve(nr, nc)

        maze[0][0] = self.FREE
        carve(0, 0)

        # Ensure the goal corner is reachable
        maze[size - 1][size - 1] = self.FREE
        maze[size - 1][size - 2] = self.FREE
        maze[size - 2][size - 1] = self.FREE

        return maze

    def _default_maze(self):
        """A fixed maze layout — useful for reproducible testing."""
        size = self.size
        maze = np.zeros((size, size), dtype=int)
        for i in range(1, size - 1, 2):
            maze[i][1:size - 1] = self.WALL
            maze[i][i % size] = self.FREE
        return maze

    # ------------------------------------------------------------------
    # RL Interface
    # ------------------------------------------------------------------

    def reset(self):
        """
        Resets the environment for a new episode.
        Returns the initial state as an integer.
        """
        self._build_maze()
        self.agent_pos = self.start_pos
        return self._pos_to_state(self.agent_pos)

    def step(self, action):
        """
        Executes one action in the environment.

        action  : integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        Returns : (next_state, reward, done)
        """
        row, col = self.agent_pos

        if action == self.UP:
            new_pos = (row - 1, col)
        elif action == self.DOWN:
            new_pos = (row + 1, col)
        elif action == self.LEFT:
            new_pos = (row, col - 1)
        elif action == self.RIGHT:
            new_pos = (row, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        if self._is_valid(new_pos):
            self.agent_pos = new_pos

        reward, done = self._get_reward()
        next_state = self._pos_to_state(self.agent_pos)

        return next_state, reward, done

    def _is_valid(self, pos):
        """Returns True if pos is inside the grid and not a wall."""
        row, col = pos
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return False
        return self.maze[row][col] != self.WALL

    def _get_reward(self):
        """
        Reward structure:
        - Goal reached : +100, episode ends
        - Any other step : -1  (encourages finding shorter paths)
        """
        if self.agent_pos == self.goal_pos:
            return 100, True
        return -1, False

    def _pos_to_state(self, pos):
        """Converts (row, col) to a unique integer state ID."""
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state):
        """Converts a state integer back to (row, col)."""
        return (state // self.size, state % self.size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_states(self):
        """Total number of cells (states) in the maze."""
        return self.size * self.size

    @property
    def num_actions(self):
        """Number of available actions."""
        return len(self.ACTIONS)

    def get_maze(self):
        """Returns a copy of the maze grid (read-only access for teammates)."""
        return self.maze.copy()

    def render(self):
        """Prints the maze in the terminal. Useful for debugging."""
        print()
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row_str += " A "
                elif (r, c) == self.goal_pos:
                    row_str += " G "
                elif self.maze[r][c] == self.WALL:
                    row_str += "███"
                else:
                    row_str += " . "
            print(row_str)
        print()
