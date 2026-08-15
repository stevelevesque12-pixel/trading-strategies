# Failed 2s (TheStrat) — intraday automation for Tradovate

Codifies the "Failed 2s" multi-timeframe reversal system (Failed-2 bias +
Market Structure Shift/FVG entry trigger) popularized in TheStrat community,
with a backtester and a Tradovate live-execution layer.

**This was built from research, not from watching the source video** — the
X/Twitter video this was requested from couldn't be accessed from this
environment (egress to x.com is blocked here). The rules below are a
best-effort reconstruction from public TheStrat/Trader-Mike material (see
research summary earlier in this conversation). Validate against your own
understanding of the setup before trusting it with money.

**A second, higher-frequency strategy also lives in this repo**:
`structure_scalp/` -- 1-minute structure direction as bias, then on the 5s
chart: wait for a pullback against that bias, wait for the 5s to break back
in the bias direction (the reversal leg), anchor a VWAP to that leg's start,
and enter on the retest touch of that VWAP. Stop = the leg's low/high,
target = a fixed R multiple. See `structure_scalp/strategy.py` and
`tradingview/structure_scalp.pine` (works for MES or MNQ -- same tick size,
just set the point-value input per symbol). It reuses the swing/MSS
primitives below but is a different strategy from Failed-2s, built because
Failed-2s trades too infrequently to be practical for a prop-firm payout
timeline. An earlier version of this strategy entered immediately on every
5s structure break while 15m+1m aligned; that traded more often but mostly
on 5s noise, so it was replaced with the pullback+VWAP-retest sequence
above for better trade quality. Not backtested against real data -- no free
source available provides sub-1-minute history, so it's logic-tested
against synthetic bars only (`tests/test_structure_scalp.py`); forward-test
carefully before trusting it.

**A third strategy also lives in this repo**: `lab_model/` -- Trader
Kane's NQ "Lab Model" (SMT divergence + inverse-FVG entries around
premium/discount zones, NQ traded with ES as a correlation reference).
See the **Lab Model** section further down for the full writeup.

## Strategy logic (Failed-2s)

1. **Bias timeframe** — a Failed-2 (`F2U`/`F2D`) completes: a directional (2)
   candle that reverses and closes back against its own break (a failed
   liquidity sweep). This sets a directional bias and the swept level.
2. **Entry timeframe** — while that bias is active and we're inside the
   intraday entry window, wait for a Market Structure Shift (a strong-bodied
   close through the last confirmed opposite swing point) in the bias
   direction, confirmed by a Fair Value Gap. That's the entry trigger.
3. **Stop** — beyond the tighter of the broken swing point or the bias
   timeframe's swept level, plus a small tick buffer.
4. **Target** — `target_r * risk` (default 1:1).
5. **Intraday only** — all state (bias, swing structure) resets every session
   day. No new entries after `no_entry_after` (default 15:45 ET); any open
   position is force-flattened at `flatten_at` (default 15:55 ET).

Timeframe pairs implemented (entry timeframe, bias timeframe):

| Pair | Entry TF | Bias TF |
|---|---|---|
| `1m-15m` | 1 minute | 15 minute |
| `5m-1h` | 5 minute | 1 hour |
| `15m-4h` | 15 minute | 4 hour |

Code layout:
- `failed2s/` — pure strategy logic (bar classification, Failed-2, swing/MSS/FVG, the signal engine, risk manager, instrument specs). Shared by backtest, live, and the webhook path.
- `backtest/` — event-driven backtester over OHLCV CSV data.
- `live/` — Tradovate REST/WebSocket client + a Python live runner (polls/streams Tradovate directly).
- `tradingview/` — the current recommended way to go live: a Pine Script port runs on TradingView (which has the market data and computes risk-based position size), fires a webhook formatted for **TradersPost** (a hosted bridge that connects to Tradovate with a regular login — no paid API Access Add-On needed, which matters on a prop-firm sim account). See **TRADINGVIEW_WEBHOOK.md** for setup.
- `webhook/` — a self-hosted alternative to TradersPost (`webhook/server.py` talks to Tradovate directly), kept for if/when direct Tradovate API credentials become available. See the "Alternative" section at the bottom of TRADINGVIEW_WEBHOOK.md.
- `tests/` — unit tests for the strategy logic, backtest smoke tests, and webhook sizing/routing tests.

## Setup

```bash
pip install -r requirements.txt
```

## Backtesting

Data format: a CSV with `timestamp,open,high,low,close[,volume]` at
**1-minute** resolution (all three pairs are resampled from this single base
resolution).

**Real historical data.** `sample_data/fetch_real_data.py` pulls genuine
historical 1-minute bars (OANDA S&P 500 / Nasdaq-100 index CFD data,
republished under GPL-3.0 by the `FutureSharks/financial-data` GitHub repo)
and converts them to this repo's CSV schema:

```bash
python sample_data/fetch_real_data.py --instrument SPX500_USD --start-year 2018 --end-year 2019 --out sample_data/real_spx500_2018_2019.csv
```

Read the caveats at the top of that script before trusting results: this is
an index CFD proxy (not literal CME ES/MES tick data), coverage only goes
through mid-2020, and volume is OANDA's tick count, not real exchange
volume. It's genuine market data though — good enough to see whether the
pattern has any real edge, not a substitute for validating against recent
CME futures data before going live. For that, see Databento or FirstRate
Data (both require a paid/API-key account not set up in this environment).

A synthetic-data generator is also included, but only for smoke-testing the
pipeline (pure random walk, no real edge signal to find):

```bash
python sample_data/generate_sample.py --days 20 --out sample_data/sample_1min.csv
```

**Single pair:**

```bash
python -m backtest.run --data sample_data/sample_1min.csv --pair 5m-1h --symbol MES
```

**Compare all three pairs on the same data** (this is what you want to
figure out which pair performs best):

```bash
python -m backtest.compare --data sample_data/sample_1min.csv --symbol MES
```

Prints a side-by-side metrics table (num trades, win rate, total P&L,
profit factor, avg R, max drawdown, expectancy) and writes:
- `trades_<pair>.csv` — a full trade log per pair
- `pair_comparison.csv` — the summary table

Both CLIs support `--symbol` (MES, ES, MNQ, NQ, MCL, CL, MGC, GC),
`--contracts`, `--daily-loss-limit`, `--max-daily-trades`, `--target-r`.

**Backtest assumptions/limitations** (so you know what you're looking at):
fills are simulated at the triggering bar's close and stop/target hits are
checked on subsequent entry-timeframe bars' high/low (one bar of latency, no
slippage/commission modeled); if a bar's range hits both stop and target the
stop is assumed to fill first (conservative). 4-hour resampling floors to
UTC-clock-hour boundaries via pandas, which may not perfectly match how your
data provider aligns 4H candles. The engine force-closes any open position
the instant it sees a bar dated after the entry day (in addition to the
normal same-day flatten-at cutoff), so a position can never silently ride
across multiple sessions -- but if the data feed itself has no bars between
the cutoff and a holiday reopen (Thanksgiving, Christmas, etc.), the
position closes on the first bar that exists, which can be a day or two
later purely because there's no earlier price to close it at. This showed
up in the real SPX500 backtest (2 of 1330 trades, both on US holiday
weekends) -- worth checking your trade log for exit_reason=session_flatten
rows with a multi-day gap before relying on this unattended over holidays.

## Live trading on Tradovate

Three ways to go live:

- **TradingView + TradersPost** (`tradingview/failed2s_mes.pine`) — the
  current recommended path. TradingView runs the signal logic and computes
  risk-based position size directly in Pine, fires a webhook formatted for
  TradersPost, which connects to Tradovate with a regular login (no paid
  API Access Add-On needed). See **TRADINGVIEW_WEBHOOK.md**.
- **TradingView + self-hosted webhook** (`webhook/`) — same idea, but you
  host the bridge yourself instead of paying for TradersPost. Needs direct
  Tradovate API credentials (CID/Secret), which require the paid API
  Access Add-On on a live funded account — not available on most prop-firm
  sim accounts. See the "Alternative" section of TRADINGVIEW_WEBHOOK.md.
- **Python runner** (`live/runner.py`, below) — runs the same
  `failed2s/strategy.py` logic directly against Tradovate's own market data
  feed, no TradingView involved, fixed 1-contract sizing. Same API access
  requirement as the self-hosted webhook path.

**Read this before pointing any of these at a funded account.** The Tradovate client
(`live/tradovate_client.py`) follows Tradovate's public API docs but has
**not been tested end-to-end against a real Tradovate account** from this
environment (no credentials, no network access to tradovateapi.com here).
Validate every call against a **demo** account first.

Environment variables:

```bash
export TRADOVATE_USERNAME=...
export TRADOVATE_PASSWORD=...
export TRADOVATE_APP_ID=...
export TRADOVATE_CID=...        # from Tradovate API settings
export TRADOVATE_SEC=...        # from Tradovate API settings
export TRADOVATE_APP_VERSION=1.0   # optional
export TRADOVATE_DEVICE_ID=failed2s-bot  # optional
```

Dry run (default — logs signals, sends no orders):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username>
```

Live (demo environment, explicit opt-in):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env demo
```

Only after demo validation, live/real money:

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env live
```

Known gap: the runner logs each signal and places a bracket order but does
not yet wire Tradovate's order-fill/user-sync WebSocket back into the risk
manager, so the daily-loss-limit lockout won't see live fills until you add
that callback (`LiveRunner._on_finished_bar` has a note where to hook it
in). Don't rely on the daily loss limit unattended until that's wired up.

## Lab Model (Trader Kane's NQ strategy)

**A third strategy lives in this repo**: `lab_model/` -- Trader Kane's NQ
"Lab Model," built from a source deck (`Trader Kane's NQ Trading
Strategy - The Lab Model.pdf`, FX Replay). It trades NQ using ES purely as
an SMT (correlation-divergence) reference, around premium/discount zones
drawn on 4h/1h/5m, with entries on a 1/3/5m execution timeframe confirmed
by SMT divergence + an inverse Fair Value Gap (iFVG).

**This is a deterministic reconstruction of a discretionary, visual
system, not a transcription.** The source deck describes things like
"identify zones that need to re-balance" and "the first high or low
through the premium/discount midline" -- concrete enough to read, but not
literally executable without judgment calls. Here's exactly what was
decided, so you can evaluate whether it matches your own read of the
material before trusting it:

- **A "recent price leg"** = the two most recently confirmed swing points
  of opposite kind, on whichever timeframe is being drawn (reusing
  `failed2s.structure`'s fractal `SwingTracker`, extended in
  `lab_model/zones.py` to keep full swing history instead of just the
  latest one). Premium/discount split at that leg's midpoint.
- **Balanced** = price has traded back through the leg's midpoint since
  the leg's second point confirmed. Tracked per-leg, so a freshly-formed
  leg starts unbalanced again even if the prior leg had balanced.
- **LLT (Logical Liquidity Target)** = read literally from the deck's own
  wording ("the first high or low through the midline") as the nearest
  pre-existing swing point on the far side of the midpoint -- not the
  extreme of the whole visible range. Used as the take-profit target for
  both entry triggers, computed on the *execution* timeframe's own leg
  (matching the deck's examples, which draw the LLT on the same chart as
  the entry, not on the 4h/1h chart).
- **Entry Trigger #1 (Reversal)**: each of NQ and ES sweeping "the 10am 4hr
  candle H/L" is checked against *that symbol's own* 4h candle (NQ vs its
  reference, ES vs its own) -- not the same price level, since NQ and ES
  trade on entirely different price scales. Whichever side (or both) sweeps
  arms a pending setup; the recorded stop is always NQ's own swept level
  (the tradable instrument). SMT divergence and the iFVG confirmation can
  arrive in either order, matching the deck ("it doesn't matter the order,
  but both are required"), and only divergences forming *after* the sweep
  count (a stale, pre-existing divergence at arm time doesn't retroactively
  qualify).
- **Entry Trigger #2 (Continuation)**: fires when the 5m zone transitions
  from unbalanced to balanced while the 1h or 4h zone is still unbalanced.
  Direction is whichever way continues price toward filling that unfilled
  HTF zone (e.g. an unbalanced HTF up-leg implies further downside).
  Stop uses the execution timeframe's current leg extreme, frozen at the
  moment the setup arms (same "recent H/L" stop concept as the reversal
  trigger, just without an explicit sweep event to anchor it to).
- **4h candle anchoring**: resampled with an explicit origin
  (`lab_model.zones.four_hour_origin`) so 4h bars close at
  02:00/06:00/10:00/14:00/18:00/22:00 local time -- matching the deck's
  "10am 4hr candle" -- instead of pandas' default UTC-midnight-aligned
  bins. Same DST-drift caveat as the plain 4H resampling already noted
  above for failed2s: bins are anchored by fixed elapsed time, not local
  wall clock, so a bin straddling a DST transition day can drift an hour.
- **Execution timeframe is fixed per backtest run** (`--execution-tf`,
  default 1m), not dynamically chosen bar-by-bar the way a discretionary
  trader would pick "whichever presents a good potential iFVG." Use
  `backtest.compare_lab_model` to run the same data through 1m/3m/5m and
  compare -- see **Execution timeframe results** below.
- **Break-even management** ("go b/e once half way to TP") is applied
  mechanically in the backtest engine (`--breakeven-at-r`, default 0.5):
  once price reaches that fraction of the way from entry to target, the
  stop moves to entry. R-multiples in the trade log are still computed
  against the *original* stop, not the moved one.

Code layout:
- `lab_model/zones.py` -- premium/discount leg tracking, balanced/imbalanced state, LLT.
- `lab_model/smt.py` -- SMT (correlation-divergence) detection between two swing histories.
- `lab_model/fvg.py` -- Fair Value Gap detection and inverse-FVG (inversion) tracking.
- `lab_model/strategy.py` -- ties the above into the Reversal and Continuation entry triggers.
- `backtest/lab_model_engine.py` -- two-instrument (NQ+ES), multi-timeframe backtest engine.

### Backtesting the Lab Model

Needs two **synchronized** 1-minute OHLCV CSVs (same schema as above): one
for NQ, one for ES. The real-data fetcher already covers a correlated pair:

```bash
python sample_data/fetch_real_data.py --instrument NAS100_USD --start-year 2019 --end-year 2019 --out sample_data/real_nq_2019.csv
python sample_data/fetch_real_data.py --instrument SPX500_USD --start-year 2019 --end-year 2019 --out sample_data/real_es_2019.csv

python -m backtest.run_lab_model --nq-data sample_data/real_nq_2019.csv --es-data sample_data/real_es_2019.csv
```

Same coverage/caveats as noted above for the OANDA CFD data (2005-mid 2020,
not literal CME tick data, OANDA tick-count volume). Options:
`--execution-tf` (1min/3min/5min), `--stop-buffer-ticks` (default 4),
`--exec-swing-strength` (default 2), `--breakeven-at-r` (default 0.25,
negative to disable), `--commission-per-contract` (default $4.60
round-turn), `--slippage-ticks` (default 1.0), `--contracts`, `--out`.

### Commission and slippage matter more than any parameter tuned above

**All of the profit factors quoted in this README up through the previous
section were computed with zero trading costs.** Once realistic costs are
turned on (the defaults: $4.60 round-turn commission/contract, 1 tick of
adverse slippage on entries, stop exits, and forced market closes -- not
target exits, which are modeled as limit orders), the "optimized" 1m,
stop=4, exec-swing=2, breakeven=0.25 configuration flips from profitable to
a net loser in **both** windows:

| Window | Trades | Win % | Profit Factor | Expectancy/trade |
|---|---|---|---|---|
| 2019, zero cost | 71 | 43.7% | 3.51 | +$13.30 |
| 2019, realistic cost | 71 | 32.4% | **0.84** | **-$2.78** |
| 2018-2020, zero cost | 138 | 44.9% | 2.01 | +$11.49 |
| 2018-2020, realistic cost | 138 | 34.8% | **0.87** | **-$2.68** |

The win rate drop (not just smaller wins) is a real, non-obvious mechanism,
not a bug: entry slippage shifts the fill price on *every* trade, which
shifts exactly where the breakeven-move threshold sits and where a
breakeven-triggered stop actually exits (itself subject to further
slippage). Since the engine only ever holds one position at a time, an
exit landing on a different bar changes whether/when the *next* signal
gets to fire, cascading into a different downstream sequence of trades --
small per-trade friction compounds into a materially different trade
history, not just a linear tax on the same trades.

**Practical takeaway: at $4.60 + 1 tick of round-turn cost and ~40-70
trades/year on a single contract, this configuration doesn't clear its own
costs on this data.** That's a real result, not a caveat to skip past --
before trusting any of the earlier optimization sections, re-run them with
costs on, e.g.:

```bash
python -m backtest.run_lab_model --nq-data sample_data/real_nq_2019.csv --es-data sample_data/real_es_2019.csv
python -m backtest.run_lab_model --nq-data sample_data/real_nq_2019.csv --es-data sample_data/real_es_2019.csv --commission-per-contract 0 --slippage-ticks 0  # to reproduce the earlier zero-cost numbers
```

The stop/target and execution-timeframe sweeps above (`optimize_lab_model`,
`compare_lab_model`) both default to realistic costs now too, but their
*rankings* were produced before this was added -- the best zero-cost
combination is not guaranteed to be the best cost-adjusted one (a
lower-frequency, higher-conviction parameter region could plausibly survive
costs better than the highest-frequency one that won on paper). Re-running
those sweeps with costs on, and specifically looking for configurations
that trade less often but at higher expectancy per trade, is the natural
next step before concluding the strategy has no edge at all.

### Execution timeframe results

Compare 1m/3m/5m execution head-to-head on the same data:

```bash
python -m backtest.compare_lab_model --nq-data sample_data/real_nq_2019.csv --es-data sample_data/real_es_2019.csv
```

Run against the OANDA proxy data on two overlapping windows (2019 only, and
2018 through mid-2020, which also covers the 2018 selloff and the 2020
COVID crash):

| Execution TF | Window | Trades | Win % | Profit Factor | Expectancy/trade |
|---|---|---|---|---|---|
| **1m** | 2018-2020 | 138 | 52.2% | **1.32** | **+$7.61** |
| 3m | 2018-2020 | 77 | 39.0% | 0.43 | -$53.56 |
| 5m | 2018-2020 | 65 | 43.1% | 0.77 | -$15.20 |
| **1m** | 2019 only | 71 | 52.1% | **1.86** | **+$11.97** |
| 3m | 2019 only | 26 | 38.5% | 0.92 | -$4.23 |
| 5m | 2019 only | 25 | 48.0% | 1.75 | +$22.80 |

**1-minute execution is the only timeframe that's consistently profitable
across both windows** -- it's the default for that reason. 3-minute is a
net loser in both windows tested. 5-minute is inconsistent (loses money
2018-2020, profitable in the 2019-only slice, on a sample as thin as 25
trades) -- more data would be needed to trust it either way.

Caveats on top of the general OANDA-proxy caveats already noted: this
compares a *fixed* execution timeframe per run against the deck's actual
discretionary per-setup chart choice, and the higher timeframes'
trade counts here (25-77 over ~2.5 years) are thin enough that these
profit-factor gaps could partly reflect noise rather than a durable edge.
Re-run `compare_lab_model` yourself before trusting this on a longer or
more recent dataset.

### Stop-loss and target parameter results

There's no free "target R multiple" to tune here -- the target is always
the LLT, a structural price level, not a multiplier. The parameters that
actually shape stop/target outcomes are `--stop-buffer-ticks` (room beyond
the swept/recent H/L before the stop sits), `--exec-swing-strength` (the
fractal window that confirms swings on the execution timeframe, which
directly drives which LLT gets picked as the target), and `--breakeven-at-r`
(trade management, swept alongside the other two since it changes realized
R on winners). Grid-search all three at once:

```bash
python -m backtest.optimize_lab_model --nq-data sample_data/real_nq_2019.csv --es-data sample_data/real_es_2019.csv
```

**A bug was caught and fixed during this optimization**: the sweep
originally reused one `LabModelStrategy` instance across the
`--breakeven-at-r` inner loop. `LabModelStrategy` is stateful (zone/swing
history, pending setups, reference candles), so every run after the first
in that loop executed against a strategy still carrying leftover state from
the previous run -- silently corrupting every breakeven comparison. It was
caught by a smell test: `breakeven_at_r=1.0` is mathematically guaranteed
to behave identically to disabling breakeven entirely (its trigger
threshold coincides exactly with the target-hit check, which is evaluated
first), but the buggy sweep showed them producing different trade counts.
Fixed by building a fresh strategy per combination (`optimize_lab_model.py`
now does this); the numbers below are from the corrected tool. Worth
internalizing before trusting *any* parameter sweep against a stateful
strategy object -- this class of bug produces plausible-looking, wrong
numbers with no error or crash.

Swept `stop_buffer_ticks` in {0,2,4,8,12,16}, `exec_swing_strength` in
{1,2,3}, `breakeven_at_r` in {none,0.25,0.5,0.75,1.0} -- 90 combinations --
on the same two windows as the execution-timeframe comparison above. With
`stop_buffer_ticks=4, exec_swing_strength=2` (both already the defaults --
confirmed rather than changed) held fixed, varying breakeven:

| Breakeven at | Window | Trades | Win % | Profit Factor | Expectancy/trade | Max DD |
|---|---|---|---|---|---|---|
| **0.25** | 2019 | 71 | 43.7% | **3.51** | **+$13.30** | **$120** |
| 0.5 (old default) | 2019 | 71 | 52.1% | 1.86 | +$11.97 | $260 |
| 0.75 | 2019 | 70 | 61.4% | 0.92 | -$2.89 | $990 |
| 1.0 / none | 2019 | 70 | 68.6% | 1.10 | +$3.37 | $776 |
| **0.25** | 2018-2020 | 138 | 44.9% | **2.01** | **+$11.49** | **$704** |
| 0.5 (old default) | 2018-2020 | 138 | 52.2% | 1.32 | +$7.61 | $1038 |
| 0.75 | 2018-2020 | 137 | 59.9% | 1.12 | +$4.32 | $1008 |
| 1.0 / none | 2018-2020 | 137 | 68.6% | 0.90 | -$5.27 | $2010 |

**`breakeven_at_r=0.25` wins on profit factor, expectancy, and max
drawdown, monotonically, in both windows** -- moving the stop to breakeven
sooner (a quarter of the way to target instead of halfway) consistently
locks in more edge than it gives up in prematurely-stopped winners. That's
now the default. `stop_buffer_ticks=4` and `exec_swing_strength=2` were
independently confirmed as the best choice among the values tested (also
already the defaults) -- see the full sweep CSV for the underlying grid if
you want to check other combinations.

Same OANDA-proxy caveats as above, plus: this is still only two windows
over one data source, and the win-rate/profit-factor tradeoff at
`breakeven_at_r=0.25` (fewer winners, each one worth more on average, per
`avg_win`/`avg_loss` in the sweep CSV) is a real behavioral shift, not just
a bigger number -- make sure that trade-off matches your own risk
tolerance before adopting it.

## Running the tests

```bash
pytest
```
