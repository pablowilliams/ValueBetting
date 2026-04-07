"""
Web Dashboard — FastAPI server serving a real-time browser dashboard.
Run with: python -m dashboard.web
"""

import sys
import os

# CRITICAL: Set up paths and env BEFORE any project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_dir)
os.chdir(_project_dir)

# Force all .env vars into os.environ
_env_path = os.path.join(_project_dir, ".env")
with open(_env_path) as _f:
    for _line in _f:
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.split("#")[0].strip()
            if _v:
                os.environ[_k] = _v

import asyncio
import json
import time
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from config import settings
from config.categories import CATEGORY_CONFIG
from core.classifier import classify_market
from core.consensus import compute_consensus
from core.edge import compute_edge
from core.market import Market, ProbEstimate, EdgeSignal
from estimators.registry import run_estimators
from execution.risk_manager import RiskManager
from execution.executor import get_executor
from execution.position_sizer import compute_position_size
from sources.polymarket import PolymarketScanner
from dashboard.logger import TradeLogger
from ml.calibration import CalibrationModel
from ml.edge_decay import EdgeDecayModel
from ml.trade_scorer import TradeQualityScorer, build_trade_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_project_dir, "valuebetting.log")),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="ValueBetting Dashboard")

# ── Bot state (shared across websocket connections) ──
scanner = PolymarketScanner()
risk = RiskManager()
executor = get_executor()
trade_logger = TradeLogger()
calibration = CalibrationModel()
edge_decay = EdgeDecayModel()
trade_scorer = TradeQualityScorer()

bot_state = {
    "scan_count": 0,
    "start_time": time.time(),
    "latest_signals": [],
    "markets": [],
    "running": False,
}

connected_clients: list[WebSocket] = []
_trade_features: dict = {}  # condition_id -> TradeFeatures for outcome recording


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ValueBetting Dashboard</title>
<style>
  :root {
    --bg: #0a0e17; --card: #111827; --border: #1e293b;
    --text: #e2e8f0; --dim: #64748b; --accent: #38bdf8;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
    --orange: #f97316; --blue: #3b82f6; --purple: #a855f7;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; padding: 16px; }
  h1 { text-align: center; color: var(--accent); font-size: 22px; margin-bottom: 4px; letter-spacing: 2px; }
  .subtitle { text-align: center; color: var(--dim); font-size: 12px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .full { grid-column: 1 / -1; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--dim); margin-bottom: 10px; }
  .stat { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .stat:last-child { border-bottom: none; }
  .stat-label { color: var(--dim); }
  .stat-value { font-weight: bold; }
  .green { color: var(--green); } .red { color: var(--red); } .yellow { color: var(--yellow); }
  .orange { color: var(--orange); } .blue { color: var(--accent); } .purple { color: var(--purple); }
  .big-number { font-size: 28px; font-weight: bold; }
  .big-label { font-size: 11px; color: var(--dim); margin-top: 2px; }
  .metric-row { display: flex; gap: 20px; margin-bottom: 8px; }
  .metric { text-align: center; flex: 1; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--dim); font-weight: 600; padding: 6px 8px; border-bottom: 2px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 6px 8px; border-bottom: 1px solid var(--border); }
  tr:hover { background: rgba(56, 189, 248, 0.04); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; }
  .badge-sports { background: #064e3b; color: #34d399; }
  .badge-politics { background: #312e81; color: #a78bfa; }
  .badge-weather { background: #1e3a5f; color: #7dd3fc; }
  .badge-crypto { background: #422006; color: #fbbf24; }
  .badge-economics { background: #1c1917; color: #a8a29e; }
  .badge-other { background: #1e293b; color: #94a3b8; }
  .badge-buy { background: #064e3b; color: #22c55e; }
  .badge-skip { background: #1e293b; color: #64748b; }
  .heat-bar { height: 6px; border-radius: 3px; background: var(--border); overflow: hidden; margin-top: 4px; }
  .heat-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .brier-bar { display: flex; align-items: center; gap: 6px; }
  .brier-track { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .brier-fill { height: 100%; border-radius: 4px; }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  @keyframes titleGlow {
    0%   { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.3); }
    20%  { color: #a78bfa; text-shadow: 0 0 20px rgba(167,139,250,0.3); }
    40%  { color: #22c55e; text-shadow: 0 0 20px rgba(34,197,94,0.3); }
    60%  { color: #f97316; text-shadow: 0 0 20px rgba(249,115,22,0.3); }
    80%  { color: #ec4899; text-shadow: 0 0 20px rgba(236,72,153,0.3); }
    100% { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.3); }
  }
  h1 { animation: titleGlow 25s ease-in-out infinite; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-green { background: var(--green); } .dot-red { background: var(--red); }
  .dot-yellow { background: var(--yellow); } .dot-dim { background: var(--dim); }
  .log-line { font-size: 11px; color: var(--dim); padding: 2px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .log-line.info { color: var(--text); }
  .log-line.edge { color: var(--green); }
  .log-line.warn { color: var(--yellow); }
  #log-container { max-height: 180px; overflow-y: auto; }
  .empty { text-align: center; color: var(--dim); padding: 20px; font-style: italic; }
</style>
</head>
<body>

<h1>VALUEBETTING</h1>
<p class="subtitle">Polymarket Value Arbitrage Engine &mdash; <span id="mode">PAPER</span> MODE &mdash; <span id="clock"></span></p>

<div class="grid">
  <!-- Status -->
  <div class="card">
    <div class="card-title">Portfolio</div>
    <div class="metric-row">
      <div class="metric">
        <div class="big-number blue" id="bankroll">$500.00</div>
        <div class="big-label">Bankroll</div>
      </div>
      <div class="metric">
        <div class="big-number" id="pnl">$0.00</div>
        <div class="big-label">Daily P&L</div>
      </div>
      <div class="metric">
        <div class="big-number" id="drawdown">0.0%</div>
        <div class="big-label">Drawdown</div>
      </div>
    </div>
    <div class="stat"><span class="stat-label">Heat Level</span><span class="stat-value" id="heat">GREEN</span></div>
    <div class="heat-bar"><div class="heat-fill" id="heat-bar" style="width:0%; background:var(--green)"></div></div>
  </div>

  <!-- Stats -->
  <div class="card">
    <div class="card-title">Session Stats</div>
    <div class="stat"><span class="stat-label">Scans</span><span class="stat-value" id="scans">0</span></div>
    <div class="stat"><span class="stat-label">Trades</span><span class="stat-value" id="trades">0</span></div>
    <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value" id="winrate">0%</span></div>
    <div class="stat"><span class="stat-label">Avg P&L/Trade</span><span class="stat-value" id="avg-pnl">$0.00</span></div>
    <div class="stat"><span class="stat-label">Open Positions</span><span class="stat-value" id="open-pos">0</span></div>
    <div class="stat"><span class="stat-label">Uptime</span><span class="stat-value" id="uptime">0m</span></div>
  </div>
</div>

<!-- Open Positions -->
<div class="card" style="margin-bottom:12px">
  <div class="card-title">Open Positions</div>
  <table>
    <thead><tr><th>Market</th><th>Category</th><th>Side</th><th>Entry</th><th>Size</th><th>True P</th><th>Age</th><th>P&L</th></tr></thead>
    <tbody id="positions-body"><tr><td colspan="8" class="empty">No open positions</td></tr></tbody>
  </table>
</div>

<div class="grid">
  <!-- Opportunities -->
  <div class="card">
    <div class="card-title">Latest Opportunities</div>
    <table>
      <thead><tr><th>Market</th><th>Cat</th><th>Side</th><th>Edge</th><th>True P</th><th>Sources</th><th>Action</th></tr></thead>
      <tbody id="opps-body"><tr><td colspan="7" class="empty">Scanning...</td></tr></tbody>
    </table>
  </div>

  <!-- ML Models -->
  <div class="card">
    <div class="card-title">Machine Learning</div>
    <div class="stat">
      <span class="stat-label"><span class="status-dot" id="cal-dot"></span>Probability Calibration</span>
      <span class="stat-value" id="cal-status">Collecting data...</span>
    </div>
    <div class="stat">
      <span class="stat-label"><span class="status-dot" id="edge-dot"></span>Edge Decay Predictor</span>
      <span class="stat-value" id="edge-status">Collecting data...</span>
    </div>
    <div class="stat">
      <span class="stat-label"><span class="status-dot" id="scorer-dot"></span>Trade Quality Scorer (LightGBM)</span>
      <span class="stat-value" id="scorer-status">Initializing...</span>
    </div>
    <div style="margin-top:12px">
      <div class="card-title" style="margin-bottom:6px">Source Calibration (Brier Score)</div>
      <div id="brier-scores"><span class="empty">No data yet</span></div>
    </div>
  </div>
</div>

<div class="grid">
  <!-- Category Performance -->
  <div class="card">
    <div class="card-title">Category Performance</div>
    <table>
      <thead><tr><th>Category</th><th>Trades</th><th>Win%</th><th>P&L</th><th>Avg Edge</th></tr></thead>
      <tbody id="cat-body"><tr><td colspan="5" class="empty">No data</td></tr></tbody>
    </table>
  </div>

  <!-- Recent Trades -->
  <div class="card">
    <div class="card-title">Recent Trades</div>
    <table>
      <thead><tr><th>Market</th><th>Cat</th><th>P&L</th><th>Edge</th><th>Exit Reason</th></tr></thead>
      <tbody id="trades-body"><tr><td colspan="5" class="empty">No trades yet</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Live Log -->
<div class="card" style="margin-top:12px">
  <div class="card-title"><span class="status-dot dot-green pulse"></span>Live Activity Log</div>
  <div id="log-container"></div>
</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);

function catBadge(cat) {
  return `<span class="badge badge-${cat || 'other'}">${cat || 'other'}</span>`;
}
function pnlColor(v) { return v >= 0 ? 'green' : 'red'; }
function pct(v) { return (v * 100).toFixed(1) + '%'; }

ws.onmessage = (e) => {
  const d = JSON.parse(e.data);

  // Portfolio
  document.getElementById('bankroll').textContent = '$' + d.bankroll.toFixed(2);
  const pnlEl = document.getElementById('pnl');
  pnlEl.textContent = '$' + (d.daily_pnl >= 0 ? '+' : '') + d.daily_pnl.toFixed(2);
  pnlEl.className = 'big-number ' + pnlColor(d.daily_pnl);
  const ddEl = document.getElementById('drawdown');
  ddEl.textContent = pct(d.drawdown);
  ddEl.className = 'big-number ' + (d.drawdown > 0.1 ? 'red' : d.drawdown > 0.05 ? 'yellow' : 'green');

  // Heat
  const heatEl = document.getElementById('heat');
  const heatColors = {GREEN:'green',YELLOW:'yellow',ORANGE:'orange',RED:'red'};
  heatEl.textContent = d.heat_level;
  heatEl.className = 'stat-value ' + (heatColors[d.heat_level] || 'green');
  const heatBar = document.getElementById('heat-bar');
  const heatPct = {GREEN:5,YELLOW:33,ORANGE:66,RED:100}[d.heat_level] || 0;
  heatBar.style.width = heatPct + '%';
  heatBar.style.background = 'var(--' + (heatColors[d.heat_level] || 'green') + ')';

  // Stats
  document.getElementById('scans').textContent = d.scan_count;
  document.getElementById('trades').textContent = d.trade_count;
  document.getElementById('winrate').textContent = (d.win_rate * 100).toFixed(0) + '%';
  const avgEl = document.getElementById('avg-pnl');
  avgEl.textContent = '$' + (d.avg_pnl >= 0 ? '+' : '') + d.avg_pnl.toFixed(2);
  avgEl.className = 'stat-value ' + pnlColor(d.avg_pnl);
  document.getElementById('open-pos').textContent = d.positions.length;
  const mins = Math.floor(d.uptime / 60);
  document.getElementById('uptime').textContent = mins < 60 ? mins + 'm' : (mins/60).toFixed(1) + 'h';
  document.getElementById('mode').textContent = d.live_mode ? 'LIVE' : 'PAPER';

  // Positions
  const pb = document.getElementById('positions-body');
  if (d.positions.length === 0) {
    pb.innerHTML = '<tr><td colspan="8" class="empty">No open positions</td></tr>';
  } else {
    pb.innerHTML = d.positions.map(p => {
      const age = p.age_seconds < 3600 ? Math.floor(p.age_seconds/60)+'m' : (p.age_seconds/3600).toFixed(1)+'h';
      const pnl = ((p.current_price||p.entry_price) - p.entry_price) * p.num_contracts;
      return `<tr>
        <td>${p.question.slice(0,40)}</td>
        <td>${catBadge(p.category)}</td>
        <td class="blue">${p.side}</td>
        <td>$${p.entry_price.toFixed(3)}</td>
        <td>$${p.size_usd.toFixed(2)}</td>
        <td>${p.true_prob.toFixed(2)}</td>
        <td>${age}</td>
        <td class="${pnlColor(pnl)}">$${pnl>=0?'+':''}${pnl.toFixed(2)}</td>
      </tr>`;
    }).join('');
  }

  // Opportunities
  const ob = document.getElementById('opps-body');
  const opps = d.opportunities || [];
  if (opps.length === 0) {
    ob.innerHTML = '<tr><td colspan="7" class="empty">No opportunities this scan</td></tr>';
  } else {
    ob.innerHTML = opps.slice(0,8).map(o => `<tr>
      <td>${o.question.slice(0,35)}</td>
      <td>${catBadge(o.category)}</td>
      <td class="blue">${o.side}</td>
      <td class="${o.net_edge >= 0.08 ? 'green' : 'yellow'}">${pct(o.net_edge)}</td>
      <td>${o.true_prob.toFixed(3)}</td>
      <td>${o.sources}</td>
      <td><span class="badge badge-${o.action.toLowerCase()}">${o.action}</span></td>
    </tr>`).join('');
  }

  // ML
  const calDot = document.getElementById('cal-dot');
  const edgeDot = document.getElementById('edge-dot');
  document.getElementById('cal-status').textContent = d.ml_calibration ? 'ACTIVE' : 'Collecting data...';
  calDot.className = 'status-dot ' + (d.ml_calibration ? 'dot-green' : 'dot-dim');
  document.getElementById('edge-status').textContent = d.ml_edge_decay ? 'ACTIVE' : 'Collecting data...';
  edgeDot.className = 'status-dot ' + (d.ml_edge_decay ? 'dot-green' : 'dot-dim');
  // Trade scorer
  const ts = d.ml_trade_scorer || {};
  const tsEl = document.getElementById('scorer-status');
  if (tsEl) {
    tsEl.textContent = ts.phase || 'Initializing...';
    const tsDot = document.getElementById('scorer-dot');
    if (tsDot) tsDot.className = 'status-dot ' + (ts.is_trained ? 'dot-green' : 'dot-dim');
  }

  // Brier scores
  const bs = document.getElementById('brier-scores');
  if (d.brier_scores && Object.keys(d.brier_scores).length > 0) {
    bs.innerHTML = Object.entries(d.brier_scores).sort((a,b)=>a[1]-b[1]).map(([src,score]) => {
      const pctW = Math.max(5, (1-score)*100);
      const col = score < 0.15 ? 'var(--green)' : score < 0.25 ? 'var(--yellow)' : 'var(--red)';
      return `<div class="brier-bar" style="margin-bottom:4px">
        <span style="width:120px;display:inline-block;color:var(--dim)">${src}</span>
        <span style="width:40px;color:${col};font-weight:bold">${score.toFixed(3)}</span>
        <div class="brier-track"><div class="brier-fill" style="width:${pctW}%;background:${col}"></div></div>
      </div>`;
    }).join('');
  }

  // Category performance
  const cb = document.getElementById('cat-body');
  const cats = d.category_stats || [];
  if (cats.length === 0) {
    cb.innerHTML = '<tr><td colspan="5" class="empty">No data</td></tr>';
  } else {
    cb.innerHTML = cats.map(c => {
      const wr = c.trades > 0 ? (c.wins/c.trades*100).toFixed(0)+'%' : '0%';
      return `<tr>
        <td>${catBadge(c.category)}</td>
        <td>${c.trades}</td>
        <td>${wr}</td>
        <td class="${pnlColor(c.pnl)}">$${c.pnl>=0?'+':''}${c.pnl.toFixed(2)}</td>
        <td>${pct(c.avg_edge || 0)}</td>
      </tr>`;
    }).join('');
  }

  // Recent trades
  const tb = document.getElementById('trades-body');
  const trades = d.recent_trades || [];
  if (trades.length === 0) {
    tb.innerHTML = '<tr><td colspan="5" class="empty">No trades yet</td></tr>';
  } else {
    tb.innerHTML = trades.slice(0,8).map(t => `<tr>
      <td>${(t.market_question||'').slice(0,28)}</td>
      <td>${catBadge(t.category)}</td>
      <td class="${pnlColor(t.pnl_usd)}">$${t.pnl_usd>=0?'+':''}${t.pnl_usd.toFixed(2)}</td>
      <td>${pct(t.net_edge_at_entry||0)}</td>
      <td>${(t.exit_reason||'').slice(0,18)}</td>
    </tr>`).join('');
  }

  // Log
  if (d.log_lines) {
    const lc = document.getElementById('log-container');
    d.log_lines.forEach(line => {
      const div = document.createElement('div');
      div.className = 'log-line' + (line.includes('EDGE') ? ' edge' : line.includes('WARNING') ? ' warn' : ' info');
      div.textContent = line;
      lc.appendChild(div);
    });
    while (lc.children.length > 100) lc.removeChild(lc.firstChild);
    lc.scrollTop = lc.scrollHeight;
  }
};

// Clock
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}, 1000);

ws.onclose = () => {
  document.querySelector('.subtitle').innerHTML += ' &mdash; <span class="red">DISCONNECTED</span>';
};
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"Dashboard client connected ({len(connected_clients)} total)")

    try:
        while True:
            # Send state every 2 seconds
            await websocket.send_text(json.dumps(build_state()))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"Dashboard client disconnected ({len(connected_clients)} total)")


log_buffer: list[str] = []


class WebLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_buffer.append(msg)
        if len(log_buffer) > 200:
            log_buffer.pop(0)


def build_state() -> dict:
    """Build current bot state for the dashboard."""
    positions = []
    for cid, pos in risk.positions.items():
        market = scanner.get_market(cid)
        current_price = market.yes_price if market else pos.entry_price
        positions.append({
            "condition_id": cid,
            "question": pos.market_question,
            "category": pos.category,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "current_price": current_price,
            "size_usd": pos.size_usd,
            "num_contracts": pos.num_contracts,
            "true_prob": pos.true_prob_at_entry,
            "age_seconds": pos.age_seconds,
        })

    opportunities = []
    for sig in bot_state.get("latest_signals", []):
        opportunities.append({
            "question": sig.market.question,
            "category": sig.market.category,
            "side": sig.side,
            "net_edge": sig.net_edge,
            "true_prob": sig.true_prob,
            "sources": sig.consensus.sources,
            "action": sig.action,
        })

    # Sort: actionable first, then by edge
    opportunities.sort(key=lambda x: (x["action"] != "BUY", -x["net_edge"]))

    stats = trade_logger.get_stats()
    brier = calibration.get_source_brier_scores() if calibration.is_trained else {}

    # Get new log lines
    new_lines = list(log_buffer[-20:])

    return {
        "bankroll": risk.bankroll,
        "daily_pnl": risk.daily_pnl,
        "drawdown": risk.drawdown,
        "heat_level": risk.heat_level,
        "scan_count": bot_state["scan_count"],
        "trade_count": risk.trade_count,
        "win_rate": risk.win_rate,
        "avg_pnl": risk.avg_pnl,
        "uptime": time.time() - bot_state["start_time"],
        "live_mode": settings.LIVE_MODE,
        "positions": positions,
        "opportunities": opportunities[:10],
        "ml_calibration": calibration.is_trained,
        "ml_edge_decay": edge_decay.is_trained,
        "ml_trade_scorer": trade_scorer.get_stats(),
        "brier_scores": brier,
        "category_stats": stats.get("by_category", []),
        "recent_trades": stats.get("recent_trades", []),
        "log_lines": new_lines,
    }


async def bot_loop():
    """Background bot scanning loop."""
    bot_state["start_time"] = time.time()

    # Add web log handler
    handler = WebLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)

    while True:
        try:
            bot_state["scan_count"] += 1
            scan_num = bot_state["scan_count"]
            logger.info(f"{'─'*30} SCAN #{scan_num} {'─'*30}")

            can_trade, reason = risk.can_trade()
            if not can_trade:
                logger.warning(f"Trading blocked: {reason}")
                await asyncio.sleep(settings.SCAN_INTERVAL_SECONDS)
                continue

            # Fetch markets
            markets = await scanner.fetch_active_markets(limit=50)
            if not markets:
                logger.warning("No markets fetched")
                await asyncio.sleep(settings.SCAN_INTERVAL_SECONDS)
                continue

            # Classify
            category_counts = {}
            for market in markets:
                market.category = classify_market(market)
                category_counts[market.category] = category_counts.get(market.category, 0) + 1
            logger.info(f"Markets: {len(markets)} | Categories: {category_counts}")

            # Estimate + edge
            signals = []
            for market in markets:
                if market.condition_id in risk.positions:
                    continue
                estimates = await run_estimators(market)
                if not estimates:
                    continue
                ml_model = calibration if calibration.is_trained else None
                consensus = compute_consensus(estimates, market.category, ml_model=ml_model)
                # Skip orderbook for paper mode — use yes_price as entry
                signal = compute_edge(market, consensus, min_edge=risk.effective_min_edge)
                signals.append(signal)

            bot_state["latest_signals"] = signals

            opportunities = [s for s in signals if s.is_actionable]
            opportunities.sort(key=lambda s: s.net_edge, reverse=True)

            trade_logger.log_scan(len(markets), len(opportunities), 0, category_counts)

            if opportunities:
                logger.info(f"Found {len(opportunities)} opportunities!")
                for i, opp in enumerate(opportunities[:3]):
                    logger.info(f"  #{i+1}: {opp.market.question[:45]} | {opp.side} | edge={opp.net_edge:.1%}")

                for opp in opportunities:
                    ct, reason = risk.can_trade(category=opp.market.category)
                    if not ct:
                        break
                    if opp.market.condition_id in risk.positions:
                        continue

                    # Score trade with ML/heuristic before executing
                    trade_feat = build_trade_features(
                        opp.market, opp.consensus, opp,
                        risk, calibration, edge_decay, trade_logger,
                    )
                    score, score_conf, score_reason = trade_scorer.predict(trade_feat)
                    logger.info(f"  Trade score: {score:.3f} | {score_reason}")

                    # Store features for post-trade recording
                    market = opp.market
                    size_usd = compute_position_size(
                        risk.bankroll,
                        opp.true_prob if opp.side == "YES" else (1 - opp.true_prob),
                        opp.entry_price, risk.sizing_multiplier,
                    )
                    if size_usd <= 0:
                        continue

                    token_id = market.token_id_yes if opp.side == "YES" else market.token_id_no
                    result = executor.buy(token_id, opp.side, size_usd, opp.entry_price, market.question)
                    if result.success:
                        from core.market import Position
                        pos = Position(
                            condition_id=market.condition_id, token_id=token_id,
                            side=opp.side, entry_price=result.fill_price,
                            size_usd=size_usd, num_contracts=result.filled_size,
                            true_prob_at_entry=opp.true_prob,
                            consensus_at_entry=opp.consensus,
                            category=market.category,
                            market_question=market.question,
                            entry_time=time.time(),
                        )
                        risk.open_position(pos)
                        # Store features for recording outcome later
                        _trade_features[market.condition_id] = trade_feat
            else:
                logger.info("No actionable opportunities")

            # Monitor positions
            if risk.positions:
                logger.info(f"Monitoring {len(risk.positions)} positions...")
                closes = []
                for cid, pos in risk.positions.items():
                    mkt = scanner.get_market(cid)
                    if not mkt:
                        continue
                    # Use Gamma API prices ONLY (orderbook is unreliable for paper)
                    if pos.side == "YES":
                        cp = mkt.yes_price
                    else:
                        cp = mkt.no_price if mkt.no_price > 0.01 else (1.0 - mkt.yes_price)

                    # Paper mode: only exit on take-profit or time stop, NO stop loss
                    pnl = cp - pos.entry_price
                    edge = pos.true_prob_at_entry - cp if pos.side == "YES" else (1 - pos.true_prob_at_entry) - cp

                    # Take profit: edge shrunk below 0.5% and we're profitable
                    if edge < 0.005 and pnl > 0.01:
                        closes.append((cid, cp, f"Take profit: edge={edge:.1%}, P&L={pnl:+.3f}"))
                    # Time stop: held >6 hours
                    elif pos.age_seconds > 21600:
                        closes.append((cid, cp, f"Time stop: held {pos.age_seconds/3600:.1f}h, P&L={pnl:+.3f}"))

                    pnl_usd = pnl * pos.num_contracts
                    logger.info(
                        f"  {pos.market_question[:40]} | {pos.side} | "
                        f"Entry={pos.entry_price:.3f} Now={cp:.3f} P&L=${pnl_usd:+.2f}"
                    )

                for cid, ep, reason in closes:
                    pos = risk.positions.get(cid)
                    if not pos:
                        continue
                    result = executor.sell(pos.token_id, pos.side, pos.num_contracts, ep, pos.market_question)
                    if result.success:
                        record = risk.close_position(cid, result.fill_price, reason)
                        if record:
                            trade_logger.log_trade(record)
                            # Record outcome for trade scorer ML training
                            feat = _trade_features.pop(cid, None)
                            if feat:
                                pnl_per_dollar = record.pnl_usd / record.size_usd if record.size_usd > 0 else 0
                                trade_scorer.record_outcome(
                                    feat, pnl_per_dollar,
                                    won=record.pnl_usd > 0,
                                    hold_seconds=record.exit_time - record.entry_time,
                                    exit_reason=record.exit_reason,
                                )

            # Retrain ML
            if bot_state["scan_count"] % 50 == 0:
                calibration.train()
                edge_decay.train()
                trade_scorer.retrain()

        except Exception as e:
            logger.error(f"Scan error: {e}", exc_info=True)

        await asyncio.sleep(settings.SCAN_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup():
    import os as _os
    _key = _os.environ.get("ANTHROPIC_API_KEY", "")
    logger.info(f"Startup: ANTHROPIC_API_KEY in env = {bool(_key)}, len={len(_key)}")
    logger.info(f"Startup: settings.ANTHROPIC_API_KEY len={len(settings.ANTHROPIC_API_KEY)}")
    asyncio.create_task(bot_loop())
    logger.info("Bot loop started")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")
