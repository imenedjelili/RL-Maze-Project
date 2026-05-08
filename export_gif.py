"""
RL Maze — GIF Exporter
========================
ENSIA · RL Project 2025/2026

Exports animated GIFs of the trained agents solving the maze.

Usage:
    python export_gif.py          # exports both agents
    python export_gif.py q        # Q-Learning only
    python export_gif.py sarsa    # SARSA only

Output:
    agent/q_learning.gif
    agent/sarsa.gif
    agent/comparison.gif   (both side by side)
"""

import sys, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
sys.setrecursionlimit(10000)

from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent

# ── Config ────────────────────────────────────────────────────────────────
CELL       = 52          # pixels per cell
FPS        = 6           # frames per second
PAUSE_END  = 2.0         # seconds to hold last frame
TRAIL_FADE = True        # show trail behind agent

# ── Colours ───────────────────────────────────────────────────────────────
C_BG      = (7,   9,  15)
C_WALL    = (15,  20,  35)
C_FREE    = (20,  28,  48)
C_GOAL_BG = (14,  42,  28)
C_START_BG= (30,  26,   8)
C_GRID    = (10,  14,  24)
C_TRAIL_Q = (30,  90, 160)
C_TRAIL_S = (160, 60,  20)
C_AGENT_Q = (79, 179, 246)
C_AGENT_S = (249,112,  72)
C_GOAL    = (61, 220, 132)
C_START   = (255,209,  80)
C_TEXT    = (208,232,255)
C_MUTED   = (60,  85, 120)
C_BAR     = (10,  14,  28)

BAR_H     = 54           # info bar height at bottom


# ── Load agents ───────────────────────────────────────────────────────────
def load_agents():
    env = MazeEnv(size=10, random_maze=False)

    q = QLearningAgent(env.num_states, env.num_actions)
    q.load("agent/q_table.npy")
    q.epsilon = 0.0

    sa = SARSAAgent(env.num_states, env.num_actions)
    if not os.path.exists("agent/sarsa_table.npy"):
        print("SARSA table not found — training now...")
        sa.train(env, num_episodes=2000, verbose=True)
        sa.save("agent/sarsa_table.npy")
    else:
        sa.load("agent/sarsa_table.npy")
    sa.epsilon = 0.0

    return env, q, sa


# ── Drawing helpers ───────────────────────────────────────────────────────
def make_base_frame(env, cell=CELL):
    """Draw the static maze (walls, free cells, S, G labels)."""
    sz  = env.size
    w   = sz * cell
    h   = sz * cell + BAR_H
    img = Image.new("RGB", (w, h), C_BG)
    d   = ImageDraw.Draw(img)

    for r in range(sz):
        for c in range(sz):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            if env.maze[r, c] == 1:
                fc = C_WALL
            elif (r, c) == tuple(env.goal_pos):
                fc = C_GOAL_BG
            elif (r, c) == (0, 0):
                fc = C_START_BG
            else:
                fc = C_FREE
            d.rectangle([x0, y0, x1-1, y1-1], fill=fc)
            d.rectangle([x0, y0, x1-1, y1-1], outline=C_GRID, width=1)

    # S and G labels
    font_sz = max(12, cell // 3)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_sz)
    except:
        font = ImageFont.load_default()

    def label(text, r, c, color):
        cx = c * cell + cell // 2
        cy = r * cell + cell // 2
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((cx - tw//2, cy - th//2), text, fill=color, font=font)

    label("S", 0, 0, C_START)
    gr, gc = env.goal_pos
    label("G", gr, gc, C_GOAL)

    return img


def draw_frame(base_img, env, path, step, agent_color, trail_color,
               title, cell=CELL):
    """
    Overlay trail + agent on a copy of base_img for the given step.
    """
    img = base_img.copy()
    d   = ImageDraw.Draw(img)
    sz  = env.size

    visited = path[:step + 1]

    # Trail
    for i, (r, c) in enumerate(visited[:-1]):
        alpha = 0.25 + 0.75 * (i / max(len(visited), 1))
        tc = tuple(int(C_FREE[k] + (trail_color[k] - C_FREE[k]) * alpha)
                   for k in range(3))
        x0, y0 = c * cell + 2, r * cell + 2
        x1, y1 = x0 + cell - 4, y0 + cell - 4
        d.rectangle([x0, y0, x1, y1], fill=tc)

    # Agent circle
    ar, ac = path[step]
    cx = ac * cell + cell // 2
    cy = ar * cell + cell // 2
    rad = cell // 2 - 5
    # Glow
    for g in range(4, 0, -1):
        gc_col = tuple(min(255, int(agent_color[k] * 0.4)) for k in range(3))
        d.ellipse([cx-rad-g, cy-rad-g, cx+rad+g, cy+rad+g], fill=gc_col)
    d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=agent_color)

    # Info bar
    bar_y = sz * cell
    d.rectangle([0, bar_y, sz*cell, bar_y + BAR_H], fill=C_BAR)

    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_big = ImageFont.load_default()
        font_sm  = font_big

    d.text((12, bar_y + 6),  title,                          fill=agent_color, font=font_big)
    d.text((12, bar_y + 28), f"Step {step} / {len(path)-1}", fill=C_MUTED,     font=font_sm)

    return img


def frames_for_agent(env, path, agent_color, trail_color, title, cell=CELL):
    """Generate all PIL frames for one agent's path."""
    base   = make_base_frame(env, cell)
    frames = []
    for step in range(len(path)):
        frames.append(draw_frame(base, env, path, step,
                                 agent_color, trail_color, title, cell))
    return frames


def save_gif(frames, path, fps=FPS, pause_end=PAUSE_END):
    imgs = [np.array(f) for f in frames]
    # Hold last frame for pause_end seconds
    hold = max(1, int(fps * pause_end))
    imgs += [imgs[-1]] * hold
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    print(f"  Saved -> {path}  ({len(imgs)} frames @ {fps} fps)")


# ── Side-by-side comparison GIF ───────────────────────────────────────────
def make_comparison_gif(env, q_path, s_path, cell=CELL):
    """Both agents side-by-side, synced by step."""
    base = make_base_frame(env, cell)
    sz   = env.size
    w, h = sz * cell, sz * cell + BAR_H

    max_steps = max(len(q_path), len(s_path))
    frames    = []

    # Pad shorter path by repeating its last position
    def pad(p, n):
        return p + [p[-1]] * (n - len(p))

    qp = pad(q_path, max_steps)
    sp = pad(s_path, max_steps)

    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_big = ImageFont.load_default()
        font_sm  = font_big

    sep = 6   # separator width between the two mazes
    cw  = w * 2 + sep
    ch  = h + 30   # extra top strip for column headers

    for step in range(max_steps):
        left  = draw_frame(base, env, qp,  min(step, len(q_path)-1),
                           C_AGENT_Q, C_TRAIL_Q, "Q-Learning", cell)
        right = draw_frame(base, env, sp, min(step, len(s_path)-1),
                           C_AGENT_S, C_TRAIL_S, "SARSA", cell)

        canvas = Image.new("RGB", (cw, ch), C_BG)
        # Header
        d = ImageDraw.Draw(canvas)
        d.text((w // 2 - 40, 6),   "Q-Learning", fill=C_AGENT_Q, font=font_big)
        d.text((w + sep + w//2 - 25, 6), "SARSA", fill=C_AGENT_S, font=font_big)
        # Step counter centred
        d.text((cw // 2 - 30, 8), f"step {step}", fill=C_MUTED, font=font_sm)

        canvas.paste(left,  (0,     30))
        canvas.paste(right, (w+sep, 30))
        frames.append(canvas)

    return frames


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("\n  RL Maze GIF Exporter")
    print("  " + "─" * 38)

    env, q_agent, sa_agent = load_agents()
    q_path  = q_agent.get_learned_path(env)
    sa_path = sa_agent.get_learned_path(env)

    os.makedirs("agent", exist_ok=True)

    if mode in ("all", "q"):
        print("\n  Rendering Q-Learning GIF...")
        frames = frames_for_agent(env, q_path,
                                  C_AGENT_Q, C_TRAIL_Q,
                                  "Q-Learning Agent")
        save_gif(frames, "agent/q_learning.gif")

    if mode in ("all", "sarsa"):
        print("\n  Rendering SARSA GIF...")
        frames = frames_for_agent(env, sa_path,
                                  C_AGENT_S, C_TRAIL_S,
                                  "SARSA Agent")
        save_gif(frames, "agent/sarsa.gif")

    if mode == "all":
        print("\n  Rendering side-by-side comparison GIF...")
        frames = make_comparison_gif(env, q_path, sa_path)
        save_gif(frames, "agent/comparison.gif")

    print("\n  Done! Files saved in agent/")
    print("  - agent/q_learning.gif")
    print("  - agent/sarsa.gif")
    print("  - agent/comparison.gif")
    print()
