"""
RL Maze — Q-Learning vs SARSA Dashboard
========================================
ENSIA · RL Project 2025/2026

Usage:
    python dashboard.py           # loads trained agents -> opens browser
    python dashboard.py retrain   # retrains SARSA first, then opens browser
"""

import sys, json, os, webbrowser
import numpy as np
sys.setrecursionlimit(10000)

from env.maze_env import MazeEnv
from agent.q_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent

# ── Load / train agents ───────────────────────────────────────────────────
def load_agents():
    env = MazeEnv(size=10, random_maze=False)

    q = QLearningAgent(env.num_states, env.num_actions)
    q.load("agent/q_table.npy")
    qs = json.load(open("agent/q_table_stats.json"))
    q.rewards_history  = qs["rewards_history"]
    q.steps_history    = qs["steps_history"]
    q.epsilon_history  = qs["epsilon_history"]
    q.epsilon = 0.0

    sa = SARSAAgent(env.num_states, env.num_actions)
    retrain = "retrain" in sys.argv or not os.path.exists("agent/sarsa_table_stats.json")
    if retrain:
        print("Training SARSA agent...")
        sa.train(env, num_episodes=2000, verbose=True)
        sa.save("agent/sarsa_table.npy")
        json.dump({"rewards_history": sa.rewards_history,
                   "steps_history":   sa.steps_history,
                   "epsilon_history": sa.epsilon_history},
                  open("agent/sarsa_table_stats.json", "w"))
    else:
        sa.load("agent/sarsa_table.npy")
        ss = json.load(open("agent/sarsa_table_stats.json"))
        sa.rewards_history = ss["rewards_history"]
        sa.steps_history   = ss["steps_history"]
        sa.epsilon_history = ss["epsilon_history"]
    sa.epsilon = 0.0
    return env, q, sa

# ── Evaluate ──────────────────────────────────────────────────────────────
def eval_agent(agent, env, n=100):
    rews, steps, wins = [], [], 0
    for _ in range(n):
        s, tot, st = env.reset(), 0, 0
        for _ in range(500):
            a = agent.best_action(s)
            s, r, done = env.step(a)
            tot += r; st += 1
            if done: wins += 1; break
        rews.append(tot); steps.append(st)
    return float(np.mean(rews)), float(wins / n * 100), float(np.mean(steps))

def convergence(h, w=100):
    s = np.convolve(h, np.ones(w) / w, mode="valid")
    thr = 0.9 * s.max()
    idx = np.where(s >= thr)[0]
    return int(idx[0]) + w // 2 if len(idx) else None

# ── Build & open ──────────────────────────────────────────────────────────
def build_dashboard(env, q, sa):
    print("Evaluating agents...")
    qr, qsr, qst = eval_agent(q,  env)
    sr, ssr, sst = eval_agent(sa, env)

    qpath = q.get_learned_path(env)
    spath = sa.get_learned_path(env)
    qconv = convergence(q.rewards_history)
    sconv = convergence(sa.rewards_history)

    data = {
        "maze":          env.maze.tolist(),
        "maze_size":     env.size,
        "q_rewards":     q.rewards_history,
        "q_steps":       q.steps_history,
        "q_epsilon":     q.epsilon_history,
        "sarsa_rewards": sa.rewards_history,
        "sarsa_steps":   sa.steps_history,
        "sarsa_epsilon": sa.epsilon_history,
        "q_path":        qpath,
        "sarsa_path":    spath,
        "q_table":       q.q_table.tolist(),
        "sarsa_table":   sa.q_table.tolist(),
        "stats": {
            "q_reward":   round(qr,  1),
            "q_success":  round(qsr, 1),
            "q_steps":    round(qst, 1),
            "q_path_len": len(qpath) - 1,
            "q_conv":     qconv,
            "s_reward":   round(sr,  1),
            "s_success":  round(ssr, 1),
            "s_steps":    round(sst, 1),
            "s_path_len": len(spath) - 1,
            "s_conv":     sconv,
        }
    }

    # Read the HTML template and inject data
    html = get_html_template().replace("__DATA_PLACEHOLDER__", json.dumps(data))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent", "dashboard.html")
    with open(out, "w") as f:
        f.write(html)

    print(f"\n  Dashboard saved -> {out}")
    print("  Opening in browser...\n")
    webbrowser.open("file://" + out)


# ── HTML template (no f-string, uses placeholder) ─────────────────────────
def get_html_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RL Maze Dashboard · ENSIA 2025/2026</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#07090f; --panel:#0d1220; --border:#1b2640;
  --q:#4fb3f6;  --s:#f97048;    --goal:#3ddc84;
  --start:#ffd166; --wall:#09111e; --free:#111828;
  --text:#d0e8ff; --muted:#3d5470;
}
* { box-sizing:border-box; margin:0; padding:0 }
body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif; min-height:100vh }

header {
  padding:2rem 3rem 1.6rem;
  border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between; gap:2rem;
  background:linear-gradient(180deg,#09101e 0%,var(--bg) 100%);
}
header h1 { font-size:clamp(1.3rem,2.5vw,2rem); font-weight:800; letter-spacing:-.02em; line-height:1.15 }
header h1 em { font-style:normal; color:#7c6af7 }
header p { color:var(--muted); font-size:.8rem; margin-top:.35rem; font-family:'Space Mono',monospace }
.legend { display:flex; gap:1.6rem; flex-shrink:0 }
.legend-item { display:flex; align-items:center; gap:.5rem; font-size:.85rem; font-weight:700 }
.dot { width:10px; height:10px; border-radius:50%; flex-shrink:0 }

main { padding:1.8rem 3rem 3rem; display:grid; gap:1.4rem }

.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem }
.card {
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:1.2rem 1.4rem; position:relative; overflow:hidden;
}
.card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px }
.card.cq::before { background:var(--q) }
.card.cs::before { background:var(--s) }
.card .lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; font-family:'Space Mono',monospace }
.card .val { font-size:2rem; font-weight:800; margin-top:.3rem; line-height:1 }
.card.cq .val { color:var(--q) }
.card.cs .val { color:var(--s) }
.card .sub { font-size:.72rem; color:var(--border); margin-top:.3rem; font-family:'Space Mono',monospace }

.panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1.4rem 1.6rem 1.6rem }
.panel-title { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); font-family:'Space Mono',monospace; margin-bottom:1.1rem }
.chart-wrap { position:relative; height:220px }

.row2 { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem }
.row4 { display:grid; grid-template-columns:repeat(4,1fr); gap:1.4rem }

.maze-wrap { display:flex; flex-direction:column; align-items:center; gap:.6rem }
canvas.maze { border-radius:6px; display:block }
.maze-label { font-size:.72rem; font-family:'Space Mono',monospace; color:var(--muted) }

.tbl { width:100%; border-collapse:collapse }
.tbl th, .tbl td { padding:.55rem .8rem; font-size:.82rem; font-family:'Space Mono',monospace; text-align:left; border-bottom:1px solid var(--border) }
.tbl th { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.07em }
.tbl td.vq { color:var(--q); font-weight:700 }
.tbl td.vs { color:var(--s); font-weight:700 }
.badge { background:#3ddc84; color:#000; font-size:.65rem; border-radius:4px; padding:.1rem .35rem; font-weight:700; margin-left:.4rem; vertical-align:middle }

.algo-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem }
.algo-card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1.4rem 1.6rem }
.algo-card h3 { font-size:.95rem; font-weight:700; margin-bottom:.7rem }
.algo-card h3.cq { color:var(--q) }
.algo-card h3.cs { color:var(--s) }
.formula {
  background:#07090f; border:1px solid var(--border); border-radius:6px;
  padding:.8rem 1rem; font-family:'Space Mono',monospace; font-size:.75rem;
  color:#8ab4d8; margin-bottom:.8rem; line-height:1.65; white-space:pre
}
.algo-card p { font-size:.83rem; color:var(--muted); line-height:1.65 }

footer { text-align:center; padding:1.4rem; color:var(--muted); font-size:.75rem;
  font-family:'Space Mono',monospace; border-top:1px solid var(--border) }
</style>
</head>
<body>

<header>
  <div>
    <h1>Q-Learning <em>vs</em> SARSA &mdash; <em>RL Maze Dashboard</em></h1>
    <p>ENSIA &middot; RL Project 2025/2026 &middot; 10&times;10 Maze &middot; 2000 Training Episodes</p>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:var(--q)"></div> Q-Learning</div>
    <div class="legend-item"><div class="dot" style="background:var(--s)"></div> SARSA</div>
  </div>
</header>

<main>
  <div class="cards">
    <div class="card cq"><div class="lbl">Q-Learning &middot; Avg Reward</div><div class="val" id="qr">-</div><div class="sub">100 eval episodes</div></div>
    <div class="card cs"><div class="lbl">SARSA &middot; Avg Reward</div><div class="val" id="sr">-</div><div class="sub">100 eval episodes</div></div>
    <div class="card cq"><div class="lbl">Q-Learning &middot; Success Rate</div><div class="val" id="qsr">-</div><div class="sub">goal reached</div></div>
    <div class="card cs"><div class="lbl">SARSA &middot; Success Rate</div><div class="val" id="ssr">-</div><div class="sub">goal reached</div></div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-title">Training Reward &mdash; 100-episode moving average</div>
      <div class="chart-wrap"><canvas id="rewardChart"></canvas></div>
    </div>
    <div class="panel">
      <div class="panel-title">Steps per Episode &mdash; smoothed</div>
      <div class="chart-wrap"><canvas id="stepsChart"></canvas></div>
    </div>
  </div>

  <div class="row4">
    <div class="panel">
      <div class="panel-title" style="color:var(--q)">Q-Learning Path</div>
      <div class="maze-wrap">
        <canvas id="mazeQ" class="maze" width="240" height="240"></canvas>
        <span class="maze-label" id="qpathlen">- steps</span>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title" style="color:var(--s)">SARSA Path</div>
      <div class="maze-wrap">
        <canvas id="mazeSARSA" class="maze" width="240" height="240"></canvas>
        <span class="maze-label" id="spathlen">- steps</span>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title" style="color:var(--q)">Q-Learning &middot; Q-value Heatmap</div>
      <div class="maze-wrap"><canvas id="heatQ" class="maze" width="240" height="240"></canvas></div>
    </div>
    <div class="panel">
      <div class="panel-title" style="color:var(--s)">SARSA &middot; Q-value Heatmap</div>
      <div class="maze-wrap"><canvas id="heatS" class="maze" width="240" height="240"></canvas></div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-title">Performance Summary</div>
      <table class="tbl">
        <thead><tr><th>Metric</th><th>Q-Learning</th><th>SARSA</th></tr></thead>
        <tbody id="summaryTbl"></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="panel-title">Epsilon Decay &mdash; Exploration Rate</div>
      <div class="chart-wrap"><canvas id="epsilonChart"></canvas></div>
    </div>
  </div>

  <div class="algo-grid">
    <div class="algo-card">
      <h3 class="cq">Q-Learning &mdash; Off-Policy</h3>
      <div class="formula">Q(s,a) &lt;- Q(s,a) + &alpha; &middot; [r + &gamma; &middot; max Q(s&apos;,a&apos;) &minus; Q(s,a)]
                                    &uarr;
                           best possible next action</div>
      <p>Off-policy: updates toward the <strong>optimal</strong> next action even if the agent does not take it. Learns the greedy policy regardless of exploration &mdash; tends to converge faster. Ideal when finding the optimal path is the priority.</p>
    </div>
    <div class="algo-card">
      <h3 class="cs">SARSA &mdash; On-Policy</h3>
      <div class="formula">Q(s,a) &lt;- Q(s,a) + &alpha; &middot; [r + &gamma; &middot; Q(s&apos;, a&apos;) &minus; Q(s,a)]
                                    &uarr;
                           actual next action taken</div>
      <p>On-policy: updates using the action the agent <strong>will actually take</strong>, including exploratory moves. Learns a safer, more conservative policy. Slightly slower to converge but more robust in stochastic environments.</p>
    </div>
  </div>
</main>

<footer>
  Team: Imene Fatma Djelili &middot; Aya Khaldi &middot; Amira Mereddef &middot; Lina Wafaa Bentiba &middot; Melynda Hadj Ali &nbsp;&middot;&nbsp; ENSIA RL Project 2025/2026
</footer>

<script>
const D = __DATA_PLACEHOLDER__;

function smooth(arr, w) {
  w = w || 100;
  var out = [];
  for (var i = w-1; i < arr.length; i++) {
    var s = 0;
    for (var j = i-w+1; j <= i; j++) s += arr[j];
    out.push(s/w);
  }
  return out;
}

var ep  = Array.from({length: D.q_rewards.length}, function(_,i){ return i; });
var qS  = smooth(D.q_rewards);
var sS  = smooth(D.sarsa_rewards);
var qSt = smooth(D.q_steps);
var sSt = smooth(D.sarsa_steps);
var xS  = ep.slice(99);
var st  = D.stats;

document.getElementById("qr").textContent   = st.q_reward;
document.getElementById("sr").textContent   = st.s_reward;
document.getElementById("qsr").textContent  = st.q_success + "%";
document.getElementById("ssr").textContent  = st.s_success + "%";
document.getElementById("qpathlen").textContent = st.q_path_len + " steps";
document.getElementById("spathlen").textContent = st.s_path_len + " steps";

var rows = [
  ["Avg Reward",       st.q_reward,                st.s_reward,                "reward"],
  ["Success Rate",     st.q_success + "%",          st.s_success + "%",         "success"],
  ["Avg Steps (eval)", st.q_steps.toFixed(1),       st.s_steps.toFixed(1),      "steps"],
  ["Path Length",      st.q_path_len + " steps",    st.s_path_len + " steps",   "path"],
  ["Convergence",      "ep " + st.q_conv,           "ep " + st.s_conv,          "conv"],
];
var tbody = document.getElementById("summaryTbl");
rows.forEach(function(row) {
  var metric = row[0], qv = row[1], sv = row[2], key = row[3];
  var qFaster = key === "conv" && st.q_conv <= st.s_conv;
  var sFaster = key === "conv" && st.s_conv < st.q_conv;
  var tr = document.createElement("tr");
  tr.innerHTML = "<td>" + metric + "</td>"
    + "<td class='vq'>" + qv + (qFaster ? "<span class='badge'>FASTER</span>" : "") + "</td>"
    + "<td class='vs'>" + sv + (sFaster ? "<span class='badge'>FASTER</span>" : "") + "</td>";
  tbody.appendChild(tr);
});

Chart.defaults.color = "#3d5470";
Chart.defaults.borderColor = "#1b2640";
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;

var baseOpts = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 900, easing: "easeOutQuart" },
  plugins: {
    legend: { display: true, labels: { color: "#8ab4d8", font: { size: 10 }, boxWidth: 12, padding: 16 } },
    tooltip: {
      backgroundColor: "rgba(10,15,28,.96)", borderColor: "#1b2640", borderWidth: 1,
      titleFont: { family: "'Space Mono',monospace", size: 10 },
      bodyFont:  { family: "'Space Mono',monospace", size: 10 },
      padding: 10
    }
  },
  scales: {
    x: { grid: { color: "rgba(27,38,64,.5)" }, ticks: { maxTicksLimit: 6, color: "#3d5470" } },
    y: { grid: { color: "rgba(27,38,64,.5)" }, ticks: { maxTicksLimit: 5, color: "#3d5470" } }
  }
};

new Chart(document.getElementById("rewardChart"), {
  type: "line",
  data: {
    labels: xS,
    datasets: [
      { label: "Q-Learning", data: qS, borderColor: "#4fb3f6", backgroundColor: "rgba(79,179,246,.07)", borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3 },
      { label: "SARSA",      data: sS, borderColor: "#f97048", backgroundColor: "rgba(249,112,72,.07)",  borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3 }
    ]
  },
  options: Object.assign({}, baseOpts, {
    scales: Object.assign({}, baseOpts.scales, {
      x: Object.assign({}, baseOpts.scales.x, { title: { display: true, text: "Episode", color: "#3d5470" } }),
      y: Object.assign({}, baseOpts.scales.y, { title: { display: true, text: "Reward",  color: "#3d5470" } })
    })
  })
});

new Chart(document.getElementById("stepsChart"), {
  type: "line",
  data: {
    labels: xS,
    datasets: [
      { label: "Q-Learning", data: qSt, borderColor: "#4fb3f6", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0.3 },
      { label: "SARSA",      data: sSt, borderColor: "#f97048", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0.3 }
    ]
  },
  options: Object.assign({}, baseOpts, {
    scales: Object.assign({}, baseOpts.scales, {
      x: Object.assign({}, baseOpts.scales.x, { title: { display: true, text: "Episode", color: "#3d5470" } }),
      y: Object.assign({}, baseOpts.scales.y, { title: { display: true, text: "Steps",   color: "#3d5470" } })
    })
  })
});

new Chart(document.getElementById("epsilonChart"), {
  type: "line",
  data: {
    labels: ep,
    datasets: [
      { label: "Q-Learning", data: D.q_epsilon,     borderColor: "#4fb3f6", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0 },
      { label: "SARSA",      data: D.sarsa_epsilon, borderColor: "#f97048", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0, borderDash: [5,4] }
    ]
  },
  options: Object.assign({}, baseOpts, {
    scales: Object.assign({}, baseOpts.scales, {
      x: Object.assign({}, baseOpts.scales.x, { title: { display: true, text: "Episode", color: "#3d5470" } }),
      y: Object.assign({}, baseOpts.scales.y, { title: { display: true, text: "epsilon", color: "#3d5470" } })
    })
  })
});

var S = D.maze_size;

function drawMaze(canvasId, path, pathColor) {
  var canvas = document.getElementById(canvasId);
  var ctx = canvas.getContext("2d");
  var cell = canvas.width / S;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (var r = 0; r < S; r++) {
    for (var c = 0; c < S; c++) {
      if (D.maze[r][c] === 1)           ctx.fillStyle = "#09111e";
      else if (r===S-1 && c===S-1)      ctx.fillStyle = "#0b2218";
      else if (r===0   && c===0)        ctx.fillStyle = "#1a1600";
      else                              ctx.fillStyle = "#111828";
      ctx.fillRect(c*cell, r*cell, cell, cell);
      ctx.strokeStyle = "#06090f"; ctx.lineWidth = 0.5;
      ctx.strokeRect(c*cell, r*cell, cell, cell);
    }
  }
  ctx.font = "bold " + Math.round(cell*.52) + "px Syne";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#ffd166"; ctx.fillText("S", cell*.5, cell*.5);
  ctx.fillStyle = "#3ddc84"; ctx.fillText("G", (S-1)*cell+cell*.5, (S-1)*cell+cell*.5);
  if (!path || path.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(path[0][1]*cell+cell/2, path[0][0]*cell+cell/2);
  for (var i = 1; i < path.length; i++)
    ctx.lineTo(path[i][1]*cell+cell/2, path[i][0]*cell+cell/2);
  ctx.strokeStyle = pathColor; ctx.lineWidth = 3;
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.globalAlpha = 0.9; ctx.stroke(); ctx.globalAlpha = 1;
  for (var i = 0; i < path.length; i++) {
    var r2 = path[i][0], c2 = path[i][1];
    var isEnd = i === 0 || i === path.length-1;
    ctx.beginPath();
    ctx.arc(c2*cell+cell/2, r2*cell+cell/2, cell*(isEnd ? .22 : .13), 0, Math.PI*2);
    ctx.fillStyle = i===0 ? "#ffd166" : i===path.length-1 ? "#3ddc84" : pathColor;
    ctx.fill();
  }
}

function drawHeatmap(canvasId, qtable, rgb) {
  var canvas = document.getElementById(canvasId);
  var ctx = canvas.getContext("2d");
  var cell = canvas.width / S;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  var maxQ = qtable.map(function(row){ return Math.max.apply(null, row); });
  var lo = Math.min.apply(null, maxQ), hi = Math.max.apply(null, maxQ), rng = hi-lo||1;
  for (var r = 0; r < S; r++) {
    for (var c = 0; c < S; c++) {
      var state = r*S+c;
      if (D.maze[r][c] === 1) {
        ctx.fillStyle = "#09111e";
      } else {
        var t = (maxQ[state]-lo)/rng;
        var R = Math.round(7+(rgb[0]-7)*t);
        var G = Math.round(9+(rgb[1]-9)*t);
        var B = Math.round(15+(rgb[2]-15)*t);
        ctx.fillStyle = "rgba("+R+","+G+","+B+","+(0.25+t*0.75)+")";
      }
      ctx.fillRect(c*cell, r*cell, cell, cell);
      ctx.strokeStyle = "#06090f"; ctx.lineWidth = 0.4;
      ctx.strokeRect(c*cell, r*cell, cell, cell);
    }
  }
  ctx.font = "bold " + Math.round(cell*.52) + "px Syne";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#3ddc84"; ctx.fillText("G",(S-1)*cell+cell*.5,(S-1)*cell+cell*.5);
  ctx.fillStyle = "#ffd166"; ctx.fillText("S",cell*.5,cell*.5);
}

drawMaze("mazeQ",    D.q_path,    "#4fb3f6");
drawMaze("mazeSARSA",D.sarsa_path,"#f97048");
drawHeatmap("heatQ", D.q_table,    [79,179,246]);
drawHeatmap("heatS", D.sarsa_table,[249,112,72]);
</script>
</body>
</html>'''


if __name__ == "__main__":
    env, q, sa = load_agents()
    build_dashboard(env, q, sa)