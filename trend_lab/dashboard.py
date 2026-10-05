"""
Render trend_lab/results/registry.json into a self-contained HTML dashboard
(no network needed): dashboard/index.html at the repo root.

  python -m trend_lab.dashboard
"""

import json
from pathlib import Path

from .optimize import load_registry

OUT = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


def verdict(r):
    o = r["oos"]
    if r["is"].get("profit_factor", 0) <= 1.0:
        return "reject"
    if o["trades"] >= 25 and o.get("profit_factor", 0) >= 1.25 and (r.get("oos_pf_top10_median") or 0) >= 1.05:
        return "candidate"
    if o["trades"] >= 15 and o.get("profit_factor", 0) > 1.0:
        return "watch"
    return "reject"


def slim(r):
    return {
        "id": r["id"], "family": r["family"], "tf": r["tf"], "iteration": r["iteration"],
        "description": r["description"], "params": r["params"], "configs": r["configs_tested"],
        "split": r["split_day"], "start": r["data_start"], "end": r["data_end"],
        "is": r["is"], "oos": r["oos"], "full": r["full"], "top10med": r.get("oos_pf_top10_median"),
        "equity": r["equity"], "verdict": verdict(r), "created": r["created"], "notes": r.get("notes", ""),
    }


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCL Trend Lab</title>
<style>
:root{color-scheme:light;--bg:#f6f6f4;--surface:#fcfcfb;--line:#e4e3de;--grid:#ecebe7;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;
--s1:#2a78d6;--oos:rgba(42,120,214,.07);--good:#008300;--warn:#b07800;--bad:#c93837;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2f2f2d;--grid:#262624;--text:#fff;--text2:#c3c2b7;--muted:#8d8c85;
--s1:#3987e5;--oos:rgba(57,135,229,.10);--good:#3fae3f;--warn:#d9a21b;--bad:#e66767;}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2f2f2d;--grid:#262624;--text:#fff;--text2:#c3c2b7;--muted:#8d8c85;
--s1:#3987e5;--oos:rgba(57,135,229,.10);--good:#3fae3f;--warn:#d9a21b;--bad:#e66767;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.sub{color:var(--text2);margin:0 0 20px;max-width:900px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .l{color:var(--text2);font-size:12px}.kpi .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.kpi .d{color:var(--muted);font-size:12px}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0 12px}
select,input[type=search]{background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font:inherit}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
th{color:var(--text2);font-weight:600;font-size:12px;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--surface)}
th:first-child,td:first-child{text-align:left}tr:hover td{background:var(--oos)}
tr.sel td{background:var(--oos)}
.badge{display:inline-flex;gap:4px;align-items:center;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor}
.candidate{color:var(--good)}.watch{color:var(--warn)}.reject{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
@media (max-width:420px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.card h3{font-size:14px;margin:0;display:flex;justify-content:space-between;gap:8px}
.card .desc{color:var(--text2);font-size:12px;margin:4px 0 8px}
.m{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.m div{font-size:11px;color:var(--text2)}.m b{display:block;font-size:14px;color:var(--text);font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9}
details{margin-top:8px}summary{cursor:pointer;color:var(--text2);font-size:12px}
pre{font-size:11px;white-space:pre-wrap;color:var(--text2);margin:6px 0 0}
.log li{margin:4px 0;color:var(--text2)}.log b{color:var(--text)}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--surface);color:var(--text2);font-size:13px;margin:16px 0;border-radius:0 8px 8px 0}
</style></head><body><main>
<h1>MCL Trend Lab</h1>
<p class="sub">Trend-following research on Micro WTI Crude (MCL) for a Lucid 50K Flex account. Every strategy is optimized on the first 60% of trading days only; <b>out-of-sample (OOS)</b> numbers come from the untouched last 40%. Costs: $1.24 RT commission + 1 tick slippage per side. Size = $200 risk/trade. Updated <span id="upd"></span>.</p>
<div class="kpis" id="kpis"></div>
<div class="note">Read OOS columns, not IS. With thousands of configs tried, in-sample results are inflated by selection. "Top-10 med" = median OOS profit factor of the 10 best in-sample configs: if that is below 1, the winner is probably luck. Lucid % = share of simulated evals (one started every trading day) that hit +$3,000 with the 50% consistency rule before breaching the $2,000 EOD trailing drawdown.</div>
<h2>Leaderboard</h2>
<div class="filters">
<select id="ftf"><option value="">All timeframes</option></select>
<select id="fv"><option value="">All verdicts</option><option>candidate</option><option>watch</option><option>reject</option></select>
<input type="search" id="fq" placeholder="Filter by family…">
</div>
<div class="tablewrap"><table id="tbl"><thead><tr></tr></thead><tbody></tbody></table></div>
<h2>Equity curves <span style="color:var(--muted);font-weight:400;font-size:12px">— net $ after costs, shaded region = out-of-sample</span></h2>
<div class="grid" id="cards"></div>
<h2>Iteration log</h2><ul class="log" id="log"></ul>
</main><div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const ITER = __ITER__;
const $ = s => document.querySelector(s);
const fmt$ = v => v==null?'–':(v<0?'−$':'$')+Math.abs(Math.round(v)).toLocaleString();
const f2 = v => v==null?'–':(+v).toFixed(2);
const pct = v => v==null?'–':(+v).toFixed(1)+'%';
const ICON = {candidate:'✓',watch:'◐',reject:'✕'};
$('#upd').textContent = new Date().toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'});

function kpis(){
  const n = DATA.length, cfgs = DATA.reduce((a,r)=>a+r.configs,0);
  const ok = DATA.filter(r=>r.oos.trades>=25);
  const best = ok.slice().sort((a,b)=>(b.oos.profit_factor||0)-(a.oos.profit_factor||0))[0];
  const luc = DATA.filter(r=>r.oos.lucid_pass_pct!=null&&r.oos.lucid_attempts>=15).sort((a,b)=>b.oos.lucid_pass_pct-a.oos.lucid_pass_pct)[0];
  const cand = DATA.filter(r=>r.verdict==='candidate').length;
  const k = [
    ['Strategies logged', n, `${cfgs.toLocaleString()} configs backtested`],
    ['Candidates', cand, 'OOS PF ≥ 1.25, ≥ 25 trades, robust top-10'],
    ['Best OOS profit factor', best?f2(best.oos.profit_factor):'–', best?`${best.id} · ${best.oos.trades} trades`:''],
    ['Best Lucid pass rate (OOS)', luc?pct(luc.oos.lucid_pass_pct):'–', luc?`${luc.id} · ${luc.oos.lucid_attempts} simulated evals`:'needs ≥ 15 OOS sims (15m data)'],
  ];
  $('#kpis').innerHTML = k.map(([l,v,d])=>`<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
}

const COLS = [
  ['Strategy', r=>r.id, r=>`${r.family}`],
  ['TF', r=>r.tf, r=>r.tf],
  ['Verdict', r=>({candidate:0,watch:1,reject:2})[r.verdict], r=>`<span class="badge ${r.verdict}">${ICON[r.verdict]} ${r.verdict}</span>`],
  ['OOS PF', r=>r.oos.profit_factor||0, r=>f2(r.oos.profit_factor)],
  ['OOS win %', r=>r.oos.win_rate||0, r=>pct(r.oos.win_rate)],
  ['OOS net', r=>r.oos.net||0, r=>fmt$(r.oos.net)],
  ['OOS max DD', r=>r.oos.max_dd||0, r=>fmt$(r.oos.max_dd)],
  ['OOS trades', r=>r.oos.trades, r=>r.oos.trades],
  ['Top-10 med', r=>r.top10med||0, r=>f2(r.top10med)],
  ['IS PF', r=>r.is.profit_factor||0, r=>f2(r.is.profit_factor)],
  ['Full win %', r=>r.full.win_rate||0, r=>pct(r.full.win_rate)],
  ['Full PF', r=>r.full.profit_factor||0, r=>f2(r.full.profit_factor)],
  ['Lucid pass (OOS)', r=>r.oos.lucid_pass_pct??-1, r=>pct(r.oos.lucid_pass_pct)],
  ['Iter', r=>r.iteration, r=>r.iteration],
];
let sortI = 3, sortDir = -1;
function rows(){
  const tf=$('#ftf').value, v=$('#fv').value, q=$('#fq').value.toLowerCase();
  return DATA.filter(r=>(!tf||r.tf===tf)&&(!v||r.verdict===v)&&(!q||r.id.toLowerCase().includes(q)))
    .sort((a,b)=>{const x=COLS[sortI][1](a),y=COLS[sortI][1](b);return (x>y?1:x<y?-1:0)*sortDir;});
}
function render(){
  $('#tbl thead tr').innerHTML = COLS.map((c,i)=>`<th data-i="${i}">${c[0]}${i===sortI?(sortDir<0?' ↓':' ↑'):''}</th>`).join('');
  const rs = rows();
  $('#tbl tbody').innerHTML = rs.map(r=>`<tr data-id="${r.id}">${COLS.map((c,i)=>`<td>${i===0?`<a href="#c-${css(r.id)}" style="color:inherit">${r.family}</a>`:c[2](r)}</td>`).join('')}</tr>`).join('');
  $('#cards').innerHTML = rs.map(card).join('');
  rs.forEach(r=>chart(r));
}
const css = s => s.replace(/[^a-z0-9]/gi,'_');
function card(r){
  const o=r.oos,f=r.full;
  return `<div class="card" id="c-${css(r.id)}"><h3><span>${r.family} · ${r.tf} <span style="color:var(--muted);font-weight:400">it${r.iteration}</span></span><span class="badge ${r.verdict}">${ICON[r.verdict]} ${r.verdict}</span></h3>
  <div class="desc">${r.description}</div><svg id="s-${css(r.id)}" viewBox="0 0 360 150" role="img" aria-label="Equity curve for ${r.id}"></svg>
  <div class="m"><div>OOS PF<b>${f2(o.profit_factor)}</b></div><div>OOS win<b>${pct(o.win_rate)}</b></div><div>OOS net<b>${fmt$(o.net)}</b></div><div>OOS DD<b>${fmt$(o.max_dd)}</b></div>
  <div>Full PF<b>${f2(f.profit_factor)}</b></div><div>Full win<b>${pct(f.win_rate)}</b></div><div>Trades/wk<b>${f2(f.trades_per_week)}</b></div><div>Lucid OOS<b>${pct(o.lucid_pass_pct)}</b></div></div>
  <details><summary>Parameters & details</summary><pre>${JSON.stringify(r.params,null,1)}
data ${r.start} → ${r.end}, OOS from ${r.split}
configs tested: ${r.configs}
full: net ${fmt$(f.net)}, max DD ${fmt$(f.max_dd)}, sharpe ${f.sharpe}, avg R ${f.avg_r}, best day ${fmt$(f.best_day)}, worst day ${fmt$(f.worst_day)}
lucid (OOS): ${o.lucid_attempts} sims, pass ${pct(o.lucid_pass_pct)}, fail ${pct(o.lucid_fail_pct)}
lucid (full, includes in-sample): ${f.lucid_attempts} sims, pass ${pct(f.lucid_pass_pct)}, fail ${pct(f.lucid_fail_pct)}, median days ${f.lucid_median_days??'–'}${r.notes?'\nnotes: '+r.notes:''}</pre></details></div>`;
}
function chart(r){
  const svg = document.getElementById('s-'+css(r.id)); if(!svg) return;
  const pts = [[r.start+'T00:00:00',0]].concat(r.equity); const W=360,H=150,L=44,R=6,T=8,B=20;
  const t = pts.map(p=>Date.parse(p[0])), y = pts.map(p=>p[1]);
  const t0=Date.parse(r.start), t1=Date.parse(r.end)+864e5, ts=Date.parse(r.split);
  let lo=Math.min(0,...y), hi=Math.max(0,...y); if(hi-lo<100){hi+=50;lo-=50}
  const X=v=>L+(v-t0)/(t1-t0)*(W-L-R), Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  const ticks=[lo,0,hi].filter((v,i,a)=>a.indexOf(v)===i);
  let s = `<rect x="${X(ts)}" y="${T}" width="${W-R-X(ts)}" height="${H-T-B}" fill="var(--oos)"/>`;
  s += ticks.map(v=>`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v===0?'var(--muted)':'var(--grid)'}" stroke-width="1"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end" font-size="10" fill="var(--text2)">${fmt$(v)}</text>`).join('');
  s += `<text x="${X(ts)+4}" y="${T+11}" font-size="10" fill="var(--text2)">OOS</text>`;
  s += `<text x="${L}" y="${H-4}" font-size="10" fill="var(--text2)">${r.start}</text><text x="${W-R}" y="${H-4}" font-size="10" fill="var(--text2)" text-anchor="end">${r.end}</text>`;
  const d = pts.map((p,i)=>`${i?'L':'M'}${X(t[i]).toFixed(1)},${Y(y[i]).toFixed(1)}`).join('');
  s += `<path d="${d}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<line class="xh" y1="${T}" y2="${H-B}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/><circle class="dot" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>`;
  s += `<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>`;
  svg.innerHTML = s;
  const xh=svg.querySelector('.xh'), dot=svg.querySelector('.dot'), tip=$('#tip');
  svg.onmousemove = e => {
    const b=svg.getBoundingClientRect(), mx=(e.clientX-b.left)/b.width*W;
    let k=0, bd=1e9; for(let i=0;i<t.length;i++){const dd=Math.abs(X(t[i])-mx); if(dd<bd){bd=dd;k=i}}
    xh.setAttribute('x1',X(t[k]));xh.setAttribute('x2',X(t[k]));xh.setAttribute('visibility','visible');
    dot.setAttribute('cx',X(t[k]));dot.setAttribute('cy',Y(y[k]));dot.setAttribute('visibility','visible');
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
    tip.innerHTML=`<b>${fmt$(y[k])}</b> cumulative<br><span style="color:var(--text2)">${pts[k][0].slice(0,16).replace('T',' ')}${t[k]>=ts?' · OOS':' · IS'}</span>`;
  };
  svg.onmouseleave = () => {xh.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');tip.style.display='none'};
}
[...new Set(DATA.map(r=>r.tf))].sort().forEach(tf=>$('#ftf').insertAdjacentHTML('beforeend',`<option>${tf}</option>`));
['#ftf','#fv','#fq'].forEach(s=>$(s).addEventListener('input',render));
$('#tbl thead').addEventListener('click',e=>{const i=+e.target.dataset.i; if(isNaN(i))return; sortDir = i===sortI?-sortDir:-1; sortI=i; render();});
$('#log').innerHTML = ITER.slice().reverse().map(it=>`<li><b>Iteration ${it.iteration}</b> · ${it.finished} · ${it.n} configs × ${it.families.length} families × [${it.tfs}] (${Math.round(it.seconds/60)} min)${it.note?' — '+it.note:''}</li>`).join('');
kpis(); render();
</script></body></html>"""


def build():
    reg = load_registry()
    data = [slim(r) for r in reg["runs"]]
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=str)).replace(
        "__ITER__", json.dumps(reg["iterations"], default=str))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    return OUT


if __name__ == "__main__":
    print(build())
