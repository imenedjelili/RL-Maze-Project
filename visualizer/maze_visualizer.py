import pygame
import numpy as np
import time


# ── Colours ────────────────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (20,  20,  20)
GRAY       = (200, 200, 200)
WALL_COLOR = (40,  40,  40)
PATH_COLOR = (230, 240, 255)   
AGENT_COL  = (50,  180, 255)   
GOAL_COLOR = (50,  220, 100)   
START_COL  = (255, 200,  50)   
TRAIL_COL  = (150, 200, 255)   
TEXT_COLOR = (255, 255, 255)
BG_COLOR   = (15,  15,  25)


class MazeVisualizer:
    """
    Pygame visualizer for the RL Maze project.
    Works with BOTH QLearningAgent and SARSAAgent — same interface.

    Usage:
        viz = MazeVisualizer(env, cell_size=60)
        viz.animate_path(path, title="SARSA Agent", delay=0.15)
        viz.close()
    """

    def __init__(self, env, cell_size=60):
        self.env       = env
        self.maze      = env.maze          
        self.size      = env.size          
        self.cell_size = cell_size

        # Window dimensions — maze + bottom info bar
        self.win_w = self.size * cell_size
        self.win_h = self.size * cell_size + 80

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("monospace", 18, bold=True)
        self.small  = pygame.font.SysFont("monospace", 14)

    # ── Internal helpers ──────────────────────────────────────────────

    def _cell_rect(self, row, col):
        """Return the pygame.Rect for a grid cell."""
        x = col * self.cell_size
        y = row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _draw_maze(self, visited=None, path=None, agent_pos=None,
                   title="", step=0, total=0):
        """Render one frame of the maze."""
        self.screen.fill(BG_COLOR)

        for r in range(self.size):
            for c in range(self.size):
                rect = self._cell_rect(r, c)

                if self.maze[r, c] == 1:
                    color = WALL_COLOR
                elif (r, c) == tuple(self.env.goal_pos):
                    color = GOAL_COLOR
                elif (r, c) == (0, 0):
                    color = START_COL
                elif visited and (r, c) in visited:
                    color = TRAIL_COL
                else:
                    color = PATH_COLOR

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BG_COLOR, rect, 1) 

        # Draw agent as a circle
        if agent_pos is not None:
            r, c   = agent_pos
            rect   = self._cell_rect(r, c)
            center = rect.center
            radius = self.cell_size // 2 - 6
            pygame.draw.circle(self.screen, AGENT_COL, center, radius)

        # Bottom info bar
        bar_y = self.size * self.cell_size
        pygame.draw.rect(self.screen, (30, 30, 50),
                         pygame.Rect(0, bar_y, self.win_w, 80))

        title_surf = self.font.render(title, True, TEXT_COLOR)
        step_surf  = self.small.render(
            f"Step {step} / {total}   |   Press ESC to skip", True, GRAY)

        self.screen.blit(title_surf, (12, bar_y + 8))
        self.screen.blit(step_surf,  (12, bar_y + 38))

        pygame.display.flip()

    # ── Public API ────────────────────────────────────────────────────

    def animate_path(self, path, title="Agent", delay=0.15):
        """
        Animate the agent walking along `path`.

        path  : list of (row, col) tuples — from agent.get_learned_path(env)
        title : shown in the info bar
        delay : seconds between frames (lower = faster)
        """
        visited = set()
        total   = len(path) - 1

        for step, pos in enumerate(path):
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            visited.add(pos)
            self._draw_maze(visited=visited,
                            agent_pos=pos,
                            title=title,
                            step=step,
                            total=total)
            self.clock.tick(60)
            time.sleep(delay)

        
        time.sleep(2)

    def show_heatmap(self, q_table, title="Q-Value Heatmap"):
        """
        Overlay Q-value heatmap on the maze.
        Each cell is coloured by its max Q-value (blue=low, red=high).
        """
        
        max_q  = np.max(q_table, axis=1).reshape(self.size, self.size)
        q_min  = max_q.min()
        q_max  = max_q.max()
        q_range = q_max - q_min if q_max != q_min else 1

        self.screen.fill(BG_COLOR)

        for r in range(self.size):
            for c in range(self.size):
                rect = self._cell_rect(r, c)

                if self.maze[r, c] == 1:
                    color = WALL_COLOR
                elif (r, c) == tuple(self.env.goal_pos):
                    color = GOAL_COLOR
                else:
                    
                    t     = (max_q[r, c] - q_min) / q_range
                    red   = int(t * 255)
                    blue  = int((1 - t) * 255)
                    color = (red, 30, blue)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BG_COLOR, rect, 1)

        # Info bar
        bar_y = self.size * self.cell_size
        pygame.draw.rect(self.screen, (30, 30, 50),
                         pygame.Rect(0, bar_y, self.win_w, 80))
        surf = self.font.render(title, True, TEXT_COLOR)
        hint = self.small.render("Blue = low Q   |   Red = high Q   |   Press any key to close",
                                 True, GRAY)
        self.screen.blit(surf, (12, bar_y + 8))
        self.screen.blit(hint, (12, bar_y + 38))
        pygame.display.flip()

        
 
        time.sleep(1)  
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    waiting = False
            pygame.display.flip() 

    def close(self):
        pygame.quit()