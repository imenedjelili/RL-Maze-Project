"""
RL Maze & Path-Finding Visual Simulator
========================================
ENSIA · RL Project 2025/2026
Team: Imene · Aya · Amira · Lina · Melynda

Single entry point for the entire application.

Usage:
    python main.py               Full pipeline:
                                   1. Train Q-Learning  (skips if already trained)
                                   2. Train SARSA       (skips if already trained)
                                   3. Animate both agents in Pygame
                                   4. Export GIFs  ->  agent/*.gif
                                   5. Open browser dashboard

    python main.py train         Retrain BOTH agents from scratch
    python main.py demo          Pygame animation only
    python main.py dashboard     Open browser dashboard only
    python main.py gif           Export GIFs only
"""

import sys, os, json
import numpy as np
sys.setrecursionlimit(10000)

from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent

# ── Config ────────────────────────────────────────────────────────────────
ENV_SIZE      = 10
NUM_EPISODES  = 2000
MAX_STEPS     = 500
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.995

# ── Helpers ───────────────────────────────────────────────────────────────
def banner(text):
    print("\n" + "=" * 54)
    print(f"   {text}")
    print("=" * 54)

def agents_trained():
    return (os.path.exists("agent/q_table.npy") and
            os.path.exists("agent/q_table_stats.json") and
            os.path.exists("agent/sarsa_table.npy") and
            os.path.exists("agent/sarsa_table_stats.json"))

def load_env():
    return MazeEnv(size=ENV_SIZE, random_maze=False)

def load_agents(env):
    q = QLearningAgent(env.num_states, env.num_actions,
                       alpha=ALPHA, gamma=GAMMA,
                       epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                       epsilon_decay=EPSILON_DECAY)
    q.load("agent/q_table.npy")
    qs = json.load(open("agent/q_table_stats.json"))
    q.rewards_history  = qs["rewards_history"]
    q.steps_history    = qs["steps_history"]
    q.epsilon_history  = qs["epsilon_history"]
    q.epsilon = 0.0

    sa = SARSAAgent(env.num_states, env.num_actions)
    sa.load("agent/sarsa_table.npy")
    ss = json.load(open("agent/sarsa_table_stats.json"))
    sa.rewards_history = ss["rewards_history"]
    sa.steps_history   = ss["steps_history"]
    sa.epsilon_history = ss["epsilon_history"]
    sa.epsilon = 0.0

    return q, sa

# ── Step 1 & 2 — Train ────────────────────────────────────────────────────
def train_q(env, force=False):
    if not force and os.path.exists("agent/q_table.npy"):
        print("  Q-Learning: saved model found, skipping.")
        print("  (use  python main.py train  to retrain from scratch)\n")
        return

    banner("Training Q-Learning Agent")
    agent = QLearningAgent(env.num_states, env.num_actions,
                           alpha=ALPHA, gamma=GAMMA,
                           epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                           epsilon_decay=EPSILON_DECAY)
    agent.train(env, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
    agent.save("agent/q_table.npy")
    agent.evaluate(env, num_episodes=20, render=False)
    json.dump({"rewards_history": agent.rewards_history,
               "steps_history":   agent.steps_history,
               "epsilon_history": agent.epsilon_history,
               "hyperparams": {"alpha": ALPHA, "gamma": GAMMA,
                               "epsilon_min": EPSILON_MIN,
                               "epsilon_decay": EPSILON_DECAY}},
              open("agent/q_table_stats.json", "w"))
    print("  Saved -> agent/q_table.npy\n")

def train_sarsa(env, force=False):
    if not force and os.path.exists("agent/sarsa_table.npy"):
        print("  SARSA: saved model found, skipping.")
        print("  (use  python main.py train  to retrain from scratch)\n")
        return

    banner("Training SARSA Agent")
    agent = SARSAAgent(env.num_states, env.num_actions)
    agent.train(env, num_episodes=NUM_EPISODES, verbose=True)
    agent.save("agent/sarsa_table.npy")
    json.dump({"rewards_history": agent.rewards_history,
               "steps_history":   agent.steps_history,
               "epsilon_history": agent.epsilon_history},
              open("agent/sarsa_table_stats.json", "w"))
    print("  Saved -> agent/sarsa_table.npy\n")

# ── Step 3 — Pygame animation ─────────────────────────────────────────────
def run_demo(env, q, sa):
    banner("Pygame Visualization")
    try:
        from visualizer.maze_visualizer import MazeVisualizer
    except ImportError:
        print("  Pygame not available, skipping animation.")
        return

    q_path  = q.get_learned_path(env)
    sa_path = sa.get_learned_path(env)

    print("  Showing Q-Learning agent...")
    viz = MazeVisualizer(env, cell_size=60)
    viz.animate_path(q_path, title="Q-Learning Agent", delay=0.18)
    viz.close()

    print("  Showing SARSA agent...")
    viz = MazeVisualizer(env, cell_size=60)
    viz.animate_path(sa_path, title="SARSA Agent", delay=0.18)
    viz.close()

# ── Step 4 — GIF export ───────────────────────────────────────────────────
def export_gifs(env, q, sa):
    banner("Exporting Animated GIFs")
    try:
        from PIL import Image, ImageDraw, ImageFont
        import imageio
    except ImportError:
        print("  Pillow / imageio not found, skipping.")
        print("  Install: pip install imageio Pillow\n")
        return

    CELL=52; FPS=6; PAUSE=2.0
    C_BG=(7,9,15); C_WALL=(15,20,35); C_FREE=(20,28,48)
    C_GOAL_BG=(14,42,28); C_START_BG=(30,26,8); C_GRID=(10,14,24)
    C_TQ=(30,90,160); C_TS=(160,60,20)
    C_AQ=(79,179,246); C_AS=(249,112,72)
    C_GOAL=(61,220,132); C_START=(255,209,80)
    C_MUTED=(60,85,120); C_BAR=(10,14,28)
    BAR=54

    def font(sz, bold=False):
        for p in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
                  f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'Bold' if bold else ''}.ttf"]:
            if os.path.exists(p):
                try: return ImageFont.truetype(p, sz)
                except: pass
        return ImageFont.load_default()

    def base_frame(env):
        sz=env.size; img=Image.new("RGB",(sz*CELL,sz*CELL+BAR),C_BG)
        d=ImageDraw.Draw(img)
        for r in range(sz):
            for c in range(sz):
                x0,y0=c*CELL,r*CELL
                fc=(C_WALL if env.maze[r,c]==1 else
                    C_GOAL_BG if (r,c)==tuple(env.goal_pos) else
                    C_START_BG if (r,c)==(0,0) else C_FREE)
                d.rectangle([x0,y0,x0+CELL-1,y0+CELL-1],fill=fc,outline=C_GRID,width=1)
        f=font(max(12,CELL//3),bold=True)
        def lbl(t,r,c,col):
            cx,cy=c*CELL+CELL//2,r*CELL+CELL//2
            bb=d.textbbox((0,0),t,font=f); tw,th=bb[2]-bb[0],bb[3]-bb[1]
            d.text((cx-tw//2,cy-th//2),t,fill=col,font=f)
        lbl("S",0,0,C_START); gr,gc=env.goal_pos; lbl("G",gr,gc,C_GOAL)
        return img

    def draw(base,env,path,step,ac,tc,title):
        img=base.copy(); d=ImageDraw.Draw(img); sz=env.size
        for i,(r,c) in enumerate(path[:step]):
            a=0.2+0.8*(i/max(step,1))
            col=tuple(int(C_FREE[k]+(tc[k]-C_FREE[k])*a) for k in range(3))
            d.rectangle([c*CELL+2,r*CELL+2,c*CELL+CELL-3,r*CELL+CELL-3],fill=col)
        ar,ac2=path[step]; cx,cy=ac2*CELL+CELL//2,ar*CELL+CELL//2; rad=CELL//2-5
        for g in range(4,0,-1):
            glow=tuple(min(255,int(ac[k]*0.3)) for k in range(3))
            d.ellipse([cx-rad-g,cy-rad-g,cx+rad+g,cy+rad+g],fill=glow)
        d.ellipse([cx-rad,cy-rad,cx+rad,cy+rad],fill=ac)
        by=sz*CELL; d.rectangle([0,by,sz*CELL,by+BAR],fill=C_BAR)
        d.text((12,by+6),title,fill=ac,font=font(15,True))
        d.text((12,by+28),f"Step {step} / {len(path)-1}",fill=C_MUTED,font=font(12))
        return img

    def save(frames,path):
        imgs=[np.array(f) for f in frames]+[np.array(frames[-1])]*max(1,int(FPS*PAUSE))
        imageio.mimsave(path,imgs,fps=FPS,loop=0)
        print(f"  Saved -> {path}")

    qp=q.get_learned_path(env); sp=sa.get_learned_path(env)
    b=base_frame(env)

    print("  Q-Learning GIF...")
    save([draw(b,env,qp,i,C_AQ,C_TQ,"Q-Learning Agent") for i in range(len(qp))],
         "agent/q_learning.gif")

    print("  SARSA GIF...")
    save([draw(b,env,sp,i,C_AS,C_TS,"SARSA Agent") for i in range(len(sp))],
         "agent/sarsa.gif")

    print("  Comparison GIF (side by side)...")
    sz=env.size; w=sz*CELL; h=sz*CELL+BAR; sep=6
    cw=w*2+sep; ch=h+30; mx=max(len(qp),len(sp))
    def pad(p,n): return p+[p[-1]]*(n-len(p))
    qpad=pad(qp,mx); spad=pad(sp,mx)
    fb=font(15,True); fs=font(12)
    comp=[]
    for step in range(mx):
        L=draw(b,env,qpad,min(step,len(qp)-1),C_AQ,C_TQ,"Q-Learning Agent")
        R=draw(b,env,spad,min(step,len(sp)-1),C_AS,C_TS,"SARSA Agent")
        canvas=Image.new("RGB",(cw,ch),C_BG); d=ImageDraw.Draw(canvas)
        d.text((w//2-45,6),"Q-Learning",fill=C_AQ,font=fb)
        d.text((w+sep+w//2-28,6),"SARSA",fill=C_AS,font=fb)
        d.text((cw//2-28,8),f"step {step}",fill=C_MUTED,font=fs)
        canvas.paste(L,(0,30)); canvas.paste(R,(w+sep,30))
        comp.append(canvas)
    save(comp,"agent/comparison.gif")
    print()

# ── Step 5 — Dashboard ────────────────────────────────────────────────────
def open_dashboard(env, q, sa):
    banner("Opening Browser Dashboard")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dashboard", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_dashboard(env, q, sa)

# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    os.makedirs("agent", exist_ok=True)

    print()
    print("  RL Maze & Path-Finding Visual Simulator")
    print("  ENSIA · RL Project 2025/2026")
    print("  Team: Imene · Aya · Amira · Lina · Melynda")

    env = load_env()

    if mode == "train":
        train_q(env, force=True)
        train_sarsa(env, force=True)
        print("\n  Both agents retrained successfully.")

    elif mode == "demo":
        if not agents_trained():
            train_q(env); train_sarsa(env)
        q, sa = load_agents(env)
        run_demo(env, q, sa)

    elif mode == "dashboard":
        if not agents_trained():
            train_q(env); train_sarsa(env)
        q, sa = load_agents(env)
        open_dashboard(env, q, sa)

    elif mode == "gif":
        if not agents_trained():
            train_q(env); train_sarsa(env)
        q, sa = load_agents(env)
        export_gifs(env, q, sa)

    else:  # full pipeline
        train_q(env)
        train_sarsa(env)
        q, sa = load_agents(env)
        run_demo(env, q, sa)
        export_gifs(env, q, sa)
        open_dashboard(env, q, sa)
        print("  All done!")
        print("  GIFs      -> agent/q_learning.gif, sarsa.gif, comparison.gif")
        print("  Dashboard -> opens in your browser\n")