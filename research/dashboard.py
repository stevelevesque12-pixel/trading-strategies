"""Render research/results.jsonl into a self-contained local HTML dashboard (research/dashboard.html).

No external scripts or fonts, so it opens offline straight from disk.
"""

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
OUT = HERE / "dashboard.html"


def _slim(d):
    """Keep only what the page needs from one result record."""
    def m(x):
        if not x:
            return None
        lu = x.get("lucid", {})
        return {k: x.get(k) for k in ("trades", "win_rate", "profit_factor", "net", "max_dd", "avg_trade", "sharpe",
                                       "trades_per_week", "pct_full_size", "pnl_full_size", "pnl_small_size",
                                       "exit_mix", "avg_qty", "years")} | {
            "pass": lu.get("pass_rate"), "bust": lu.get("bust_rate"), "days": lu.get("median_days_to_pass"),
            "evals": lu.get("evals")}
    return {
        "id": d["id"], "v": d.get("v", 1), "created": d["created"], "track": d["track"], "tf": d["tf_min"], "split": d["split"],
        "name": d["name"], "spec": d["spec"], "tried": d["configs_tried"], "robust": d["robust"],
        "is": m(d["is"]), "oos": m(d["oos"]), "full": m(d["full"]), "mes": m(d.get("mes_check")),
        "eq": d["full"].get("equity", []),
    }


def build():
    recs = []
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                recs.append(_slim(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    payload = json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "records": recs}, separators=(",", ":"))
    OUT.write_text(TEMPLATE.replace("/*__DATA__*/null", payload))
    return OUT


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MES Trend Lab</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --surface-2: #f0efec; --border: #e2e1dc;
  --text: #0b0b0b; --text-2: #52514e; --muted: #7d7c77;
  --series: #2a78d6; --series-fill: rgba(42,120,214,.10); --oos: rgba(235,104,52,.07);
  --good: #008300; --bad: #d03b3b; --warn: #b07400; --grid: #e9e8e4; --row-hover: #f1f4f9; --sel: #e4edf9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --surface-2: #232321; --border: #33332f;
    --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86;
    --series: #3987e5; --series-fill: rgba(57,135,229,.16); --oos: rgba(217,89,38,.12);
    --good: #3fb950; --bad: #e66767; --warn: #d9a400; --grid: #2a2a27; --row-hover: #20242b; --sel: #1d2a3d;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --surface-2: #232321; --border: #33332f;
  --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86;
  --series: #3987e5; --series-fill: rgba(57,135,229,.16); --oos: rgba(217,89,38,.12);
  --good: #3fb950; --bad: #e66767; --warn: #d9a400; --grid: #2a2a27; --row-hover: #20242b; --sel: #1d2a3d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }
h2 { font-size: 15px; margin: 0 0 12px; }
.sub { color: var(--text-2); margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .k { color: var(--text-2); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile .n { color: var(--muted); font-size: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; }
.filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
select, input[type=search] { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font: inherit; }
label.chk { display: inline-flex; gap: 6px; align-items: center; color: var(--text-2); }
.tbl-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 8px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text-2); font-weight: 500; font-size: 12px; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--surface); }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--row-hover); }
tbody tr.sel { background: var(--sel); }
.pos { color: var(--good); } .neg { color: var(--bad); }
.badge { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; padding: 1px 6px; border-radius: 999px; border: 1px solid var(--border); color: var(--text-2); }
.badge.ok { border-color: var(--good); color: var(--good); }
.spark { display: block; }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 12px 0; }
.chart { position: relative; }
.chart svg { width: 100%; height: 300px; display: block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; white-space: nowrap; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .cols { grid-template-columns: 1fr; } }
.kv { width: 100%; } .kv td { text-align: left; border: 0; padding: 3px 6px; white-space: normal; } .kv td:first-child { color: var(--text-2); width: 40%; }
.pager { display: flex; gap: 8px; align-items: center; margin-top: 10px; color: var(--text-2); }
button { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; font: inherit; cursor: pointer; }
.note { color: var(--muted); font-size: 12px; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--text-2); margin-bottom: 6px; flex-wrap: wrap; }
.sw { display: inline-block; width: 12px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>MES Trend Lab</h1>
  <p class="sub" id="sub"></p>
  <div class="tiles" id="tiles"></div>

  <div class="card" id="detail" style="display:none"></div>

  <div class="card">
    <h2>Every strategy tested</h2>
    <div class="filters">
      <select id="f-track"><option value="">All tracks</option><option value="15m_full">15m, full history (10y)</option><option value="5m_recent">5m, recent MES</option></select>
      <label class="chk"><input type="checkbox" id="f-robust"> Robust only</label>
      <label class="chk"><input type="checkbox" id="f-latest" checked> Latest engine only</label>
      <input type="search" id="f-q" placeholder="Filter by component…">
      <span class="note" id="count"></span>
    </div>
    <div class="tbl-wrap"><table>
      <thead><tr id="thead"></tr></thead>
      <tbody id="tbody"></tbody>
    </table></div>
    <div class="pager"><button id="prev">Prev</button><span id="page"></span><button id="next">Next</button></div>
  </div>

  <div class="card note">
    <b>How to read this.</b> Each row is one strategy family (trend + 2 confirmations + chop filter, ATR-band stop/target) after optimisation.
    Parameters are fit on the <b>in-sample</b> period only; <b>OOS</b> columns are the untouched later period and are the numbers to trust.
    Fees: $2.50/contract round turn + 1 tick slippage on every market fill. Sizing: risk $ per trade from the ATR stop, full size in strong-trend regime, reduced in weak-trend regime, no trade when the filter reads sideways.
    Lucid pass % = share of simulated LucidFlex 50K evals (one started every 5 sessions) that hit +$3,000 with the 50% consistency rule before touching the $2,000 EOD-trailing MLL, within ~250 sessions (LucidFlex has no time limit, so an unresolved eval is "still open", not failed). v1 rows used a 60-session cutoff and no trailing stop; untick "Latest engine only" to see them.
    "Robust" = IS and OOS profit factor &gt; 1.05, OOS Sharpe &gt; 0.3 and enough trades in both. With thousands of configurations tried, expect some OOS winners to be luck: favour families where many variants are robust.
  </div>
</div>
<script>
const DATA = /*__DATA__*/null;
const R = DATA.records;
const VMAX = Math.max(1, ...R.map(r => r.v || 1));
const $ = s => document.querySelector(s);
const fmt$ = v => v == null ? "–" : (v < 0 ? "−$" : "$") + Math.abs(Math.round(v)).toLocaleString();
const pct = v => v == null ? "–" : (v * 100).toFixed(0) + "%";
const f2 = v => v == null ? "–" : (v >= 99 ? "∞" : v.toFixed(2));
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";

const score = r => (r.robust ? 1000 : 0) + (r.oos?.pass ?? 0) * 100 + (r.oos?.sharpe ?? -9);
R.forEach(r => r._score = score(r));

// header tiles
(function () {
  const f15 = R.filter(r => r.track === "15m_full"), rob = R.filter(r => r.robust);
  const tried = R.reduce((a, r) => a + (r.tried || 0), 0);
  const best = [...R].sort((a, b) => b._score - a._score)[0];
  $("#sub").textContent = `Intraday trend-following research on MES · Lucid 50K rules · updated ${DATA.generated}`;
  const tiles = [
    ["Strategies logged", R.length.toLocaleString(), `${f15.length} on 10-year 15m`],
    ["Configurations backtested", tried.toLocaleString(), "in-sample optimisation"],
    ["Robust (IS + OOS)", rob.length.toLocaleString(), R.length ? pct(rob.length / R.length) + " of logged" : ""],
    ["Best OOS profit factor", best ? f2(best.oos?.profit_factor) : "–", best ? best.name : ""],
    ["Best OOS Lucid pass", best ? pct(best.oos?.pass) : "–", best ? `bust ${pct(best.oos?.bust)}` : ""],
  ];
  $("#tiles").innerHTML = tiles.map(([k, v, n]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`).join("");
})();

const COLS = [
  ["Strategy", r => r.name, r => `${r.name} <span class="note">${r.tf}m · v${r.v}</span>`, "l"],
  ["Equity (full)", null, r => spark(r.eq), "l"],
  ["Robust", r => r.robust ? 1 : 0, r => r.robust ? '<span class="badge ok">✓ robust</span>' : '<span class="badge">– no</span>', "l"],
  ["IS PF", r => r.is?.profit_factor, r => f2(r.is?.profit_factor)],
  ["OOS PF", r => r.oos?.profit_factor, r => f2(r.oos?.profit_factor)],
  ["OOS win %", r => r.oos?.win_rate, r => pct(r.oos?.win_rate)],
  ["OOS net", r => r.oos?.net, r => `<span class="${cls(r.oos?.net)}">${fmt$(r.oos?.net)}</span>`],
  ["OOS max DD", r => r.oos?.max_dd, r => fmt$(r.oos?.max_dd)],
  ["OOS Sharpe", r => r.oos?.sharpe, r => f2(r.oos?.sharpe)],
  ["OOS Lucid pass", r => r.oos?.pass, r => pct(r.oos?.pass)],
  ["OOS bust", r => r.oos?.bust, r => pct(r.oos?.bust)],
  ["Trades/wk", r => r.full?.trades_per_week, r => r.full?.trades_per_week?.toFixed(1) ?? "–"],
  ["Logged", r => r.created, r => r.created.slice(5, 16).replace("T", " "), "l"],
];
let sortCol = -1, sortDir = -1, page = 0, selId = null;
const PER = 50;
$("#thead").innerHTML = COLS.map((c, i) => `<th class="${c[3] || ""}" data-i="${i}">${c[0]}</th>`).join("");
document.querySelectorAll("th").forEach(th => th.onclick = () => {
  const i = +th.dataset.i; if (!COLS[i][1]) return;
  sortDir = sortCol === i ? -sortDir : -1; sortCol = i; page = 0; render();
});
["#f-track", "#f-robust", "#f-q", "#f-latest"].forEach(s => $(s).oninput = () => { page = 0; render(); });
$("#prev").onclick = () => { page = Math.max(0, page - 1); render(); };
$("#next").onclick = () => { page++; render(); };

function rows() {
  const t = $("#f-track").value, rob = $("#f-robust").checked, q = $("#f-q").value.toLowerCase();
  const lat = $("#f-latest").checked;
  let rs = R.filter(r => (!lat || r.v === VMAX) && (!t || r.track === t) && (!rob || r.robust) && (!q || r.name.toLowerCase().includes(q)));
  const key = sortCol >= 0 ? COLS[sortCol][1] : r => r._score;
  rs.sort((a, b) => { const x = key(a), y = key(b); return (x == null) - (y == null) || (x < y ? -1 : x > y ? 1 : 0) * sortDir; });
  return rs;
}
function spark(eq) {
  if (!eq || eq.length < 2) return "";
  const w = 110, h = 26, ys = eq.map(p => p[1]), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), sp = hi - lo || 1;
  const pts = eq.map((p, i) => `${(i / (eq.length - 1) * w).toFixed(1)},${(h - 2 - (p[1] - lo) / sp * (h - 4)).toFixed(1)}`).join(" ");
  const z = (h - 2 - (0 - lo) / sp * (h - 4)).toFixed(1);
  return `<svg class="spark" width="${w}" height="${h}" aria-hidden="true"><line x1="0" x2="${w}" y1="${z}" y2="${z}" stroke="var(--grid)"/><polyline fill="none" stroke="var(--series)" stroke-width="1.5" points="${pts}"/></svg>`;
}
function render() {
  const rs = rows(), pages = Math.max(1, Math.ceil(rs.length / PER));
  page = Math.min(page, pages - 1);
  $("#count").textContent = `${rs.length.toLocaleString()} shown`;
  $("#page").textContent = `Page ${page + 1} of ${pages}`;
  $("#tbody").innerHTML = rs.slice(page * PER, page * PER + PER).map(r =>
    `<tr data-id="${r.id}" class="${r.id === selId ? "sel" : ""}">${COLS.map(c => `<td class="${c[3] || ""}">${c[2](r)}</td>`).join("")}</tr>`).join("");
  document.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => select(tr.dataset.id));
  if (selId === null && rs.length) select(rs[0].id, false);
}
function select(id, scroll = true) {
  selId = id;
  document.querySelectorAll("tbody tr").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === id));
  const r = R.find(x => x.id === id), d = $("#detail");
  d.style.display = "";
  const tile = (k, v, n = "") => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`;
  const s = r.spec, p = o => Object.entries(o).map(([k, v]) => `${k}=${v}`).join(", ") || "–";
  const per = (lab, m) => m ? `<tr><td>${lab}</td><td>${m.trades} trades · win ${pct(m.win_rate)} · PF ${f2(m.profit_factor)} · net ${fmt$(m.net)} · DD ${fmt$(m.max_dd)} · Sharpe ${f2(m.sharpe)} · Lucid pass ${pct(m.pass)} / bust ${pct(m.bust)}${m.days ? ` · ${m.days} sessions to pass` : ""}</td></tr>` : "";
  d.innerHTML = `
    <h2>${r.name} <span class="note">· ${r.tf}m · ${r.track === "15m_full" ? "ES-as-MES 15m, 2016–2026" : "MES 5m, recent"} · ${r.robust ? "✓ robust" : "not robust"}</span></h2>
    <div class="detail-grid">
      ${tile("OOS win rate", pct(r.oos?.win_rate), `IS ${pct(r.is?.win_rate)}`)}
      ${tile("OOS profit factor", f2(r.oos?.profit_factor), `IS ${f2(r.is?.profit_factor)}`)}
      ${tile("OOS net P&L", fmt$(r.oos?.net), `${r.oos?.trades} trades`)}
      ${tile("OOS max drawdown", fmt$(r.oos?.max_dd), `IS ${fmt$(r.is?.max_dd)}`)}
      ${tile("OOS Lucid pass", pct(r.oos?.pass), `bust ${pct(r.oos?.bust)}`)}
      ${tile("Full-period PF", f2(r.full?.profit_factor), `win ${pct(r.full?.win_rate)}`)}
    </div>
    <div class="legend"><span><span class="sw" style="background:var(--series)"></span>Cumulative net P&L, 1 strategy, MES $</span><span><span class="sw" style="background:var(--oos);border:1px solid var(--border)"></span>Out-of-sample period (from ${r.split})</span></div>
    <div class="chart" id="chart"><svg id="eqsvg" role="img" aria-label="Equity curve"></svg><div class="tip" id="tip"></div></div>
    <div class="cols" style="margin-top:12px">
      <table class="kv">
        <tr><td>Trend</td><td>${s.trend} (${p(s.trend_p)})</td></tr>
        <tr><td>Confirmation 1</td><td>${s.conf1} (${p(s.conf1_p)})</td></tr>
        <tr><td>Confirmation 2</td><td>${s.conf2} (${p(s.conf2_p)})</td></tr>
        <tr><td>Chop filter</td><td>${s.regime} (${p(s.regime_p)})</td></tr>
        <tr><td>ATR bands</td><td>ATR(${s.atr_n}) · stop ${s.sl_k}× · target ${s.tp_k}×${s.trail_k ? ` · trailing ${s.trail_k}×` : ""}</td></tr>
        <tr><td>Entry / exit</td><td>${s.trigger === "fresh" ? "first aligned bar" : "any aligned bar"} · ${s.exit_on_flip ? "exit on trend flip" : "bracket only"} · window ${s.window}</td></tr>
        <tr><td>Sizing</td><td>risk $${s.risk_usd} strong trend · ×${s.small_mult} weak trend · day stop −$${s.daily_loss_limit} · day cap ${s.daily_profit_cap > 9999 ? "none" : "+$" + s.daily_profit_cap}</td></tr>
        <tr><td>Configs tried</td><td>${r.tried.toLocaleString()}</td></tr>
      </table>
      <table class="kv">
        ${per("In-sample", r.is)}${per("Out-of-sample", r.oos)}${per("Full period", r.full)}${per("Real MES 15m 2025–26", r.mes)}
        <tr><td>Size buckets (full)</td><td>full-size trades ${pct(r.full?.pct_full_size)} · P&L full ${fmt$(r.full?.pnl_full_size)} · reduced ${fmt$(r.full?.pnl_small_size)}</td></tr>
        <tr><td>Exits (full)</td><td>${r.full?.exit_mix ? Object.entries(r.full.exit_mix).map(([k, v]) => `${k} ${v}`).join(" · ") : "–"}</td></tr>
      </table>
    </div>`;
  drawEquity(r);
  if (scroll) d.scrollIntoView({ behavior: "smooth", block: "start" });
}
function drawEquity(r) {
  const svg = $("#eqsvg"), tip = $("#tip"), eq = r.eq;
  const W = svg.clientWidth || 900, H = 300, m = { l: 64, r: 12, t: 10, b: 26 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!eq.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="var(--muted)">No trades</text>`; return; }
  const xs = eq.map(p => p[0]), ys = eq.map(p => p[1]);
  const x0 = xs[0], x1 = xs[xs.length - 1] || x0 + 1, lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), sp = (hi - lo) || 1;
  const X = t => m.l + (t - x0) / ((x1 - x0) || 1) * (W - m.l - m.r), Y = v => m.t + (hi - v) / sp * (H - m.t - m.b);
  const split = Date.parse(r.split);
  let g = "";
  if (split > x0 && split < x1) g += `<rect x="${X(split)}" y="${m.t}" width="${W - m.r - X(split)}" height="${H - m.t - m.b}" fill="var(--oos)"/>`;
  const ticks = 4; for (let i = 0; i <= ticks; i++) { const v = lo + sp * i / ticks;
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmt$(v)}</text>`; }
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)" stroke-dasharray="3 3"/>`;
  const yrs = new Set(); xs.forEach(t => yrs.add(new Date(t).getFullYear()));
  const yl = [...yrs], stepY = Math.ceil(yl.length / 8);
  yl.forEach((y, i) => { const t = Date.UTC(y, 0, 1); if (i % stepY || t < x0) return;
    g += `<text x="${X(t)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">${y}</text>`; });
  if (yl.length <= 2) { [x0, x1].forEach((t, i) => g += `<text x="${X(t)}" y="${H - 8}" text-anchor="${i ? "end" : "start"}" font-size="11" fill="var(--muted)">${new Date(t).toISOString().slice(0, 10)}</text>`); }
  const path = eq.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  g += `<path d="${path}L${X(x1)},${Y(0)}L${X(x0)},${Y(0)}Z" fill="var(--series-fill)"/><path d="${path}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>`;
  g += `<line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" visibility="hidden"/><circle id="dot" r="4" fill="var(--series)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>`;
  g += `<rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" id="hit"/>`;
  svg.innerHTML = g;
  const hit = svg.querySelector("#hit"), xh = svg.querySelector("#xh"), dot = svg.querySelector("#dot");
  hit.onmousemove = e => {
    const b = svg.getBoundingClientRect(), px = (e.clientX - b.left) * W / b.width;
    let k = 0, best = 1e18; eq.forEach((p, i) => { const dd = Math.abs(X(p[0]) - px); if (dd < best) { best = dd; k = i; } });
    const p = eq[k], cx = X(p[0]), cy = Y(p[1]);
    xh.setAttribute("x1", cx); xh.setAttribute("x2", cx); xh.setAttribute("visibility", "visible");
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); dot.setAttribute("visibility", "visible");
    tip.style.display = "block";
    tip.innerHTML = `<b>${fmt$(p[1])}</b><br><span class="note">${new Date(p[0]).toISOString().slice(0, 10)} · ${p[0] >= split ? "out-of-sample" : "in-sample"}</span>`;
    const tx = cx * b.width / W; tip.style.left = Math.min(tx + 12, b.width - 150) + "px"; tip.style.top = (cy * b.height / H - 10) + "px";
  };
  hit.onmouseleave = () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); };
}
window.addEventListener("resize", () => { const r = R.find(x => x.id === selId); if (r) drawEquity(r); });
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(build())
