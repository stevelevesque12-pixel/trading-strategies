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
`structure_scalp/` -- 15m + 1m market-structure alignment with entries on a
5-second chart. See `structure_scalp/strategy.py` and
`tradingview/structure_scalp_mes.pine`. It reuses the swing/MSS primitives
below but is a different strategy, built because Failed-2s trades too
infrequently to be practical for a prop-firm payout timeline. It has NOT
been backtested against real data -- no free source available provides
sub-1-minute history, so it's logic-tested against synthetic bars only
(`tests/test_structure_scalp.py`); forward-test carefully before trusting it.

**A third strategy, `volume_scalp/`, is a volume-confirmed scalp for
MES/MNQ** -- unlike structure_scalp's 5-second entries, it runs on a single
timeframe (1-minute by default) and so it CAN be backtested end-to-end
against real historical data, not just synthetic bars. See "Strategy logic
(Volume Scalp)" below.

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
- `volume_scalp/` — the third strategy (indicators + signal logic), described below. Its own backtest engine/CLIs live in `backtest/volume_engine.py`, `backtest/run_volume_scalp.py`, `backtest/compare_volume_scalp.py`.
- `tests/` — unit tests for the strategy logic, backtest smoke tests, and webhook sizing/routing tests.

## Strategy logic (Volume Scalp)

`volume_scalp/` — a volume-confirmed scalp for MES/MNQ, built for higher
trade frequency than Failed-2s while still being backtestable against real
data (unlike `structure_scalp/`, which needs sub-1-minute bars nobody
provides for free). Single timeframe, default 1-minute. Two independent
entry setups, checked in order, one position at a time:

1. **Volume breakout** (momentum/continuation) — price closes through a
   recent N-bar high/low channel on a bar where:
   - **RVOL** (relative volume: this bar's volume vs. the trailing
     average, baseline excludes the bar itself) is above threshold
     (default 1.5x) — filters out breakouts on thin participation that
     tend to snap back.
   - **Volume delta** — a close-location-value (CLV) proxy for net
     buying/selling volume (`((close-low)-(high-close))/(high-low) *
     volume`, the same idea behind the classic Accumulation/Distribution
     Line — this repo has no tick/bid-ask data, so this is the standard
     OHLCV-only approximation of order flow), summed over a trailing
     window, agrees with the breakout direction.
   - Optionally, price is on the correct side of session **VWAP** (a
     trend filter — default on).
2. **Volume climax fade** (exhaustion/reversal) — a bar with RVOL far
   above normal (default 3x) that pushed hard in one direction but closed
   back near the middle/opposite side (a rejection wick ≥50% of the bar's
   range by default) fades the move — a volume spike with no
   follow-through close reads as absorption/exhaustion (classic Wyckoff
   upthrust/spring behavior), not real continuation. Disable with
   `enable_climax_fade=False` to run breakout-only.

**Stop** — breakout: beyond the tighter of the signal bar's own high/low or
the broken channel level, plus a tick buffer. Climax fade: beyond the
signal bar's extreme, plus a tick buffer.
**Target** — `target_r * risk` (default 1.5:1).
**Intraday only** — all rolling state (VWAP, RVOL baseline, breakout
channel, volume-delta window) resets every session day; same entry-window /
flatten-cutoff behavior as Failed-2s (`SessionConfig`, shared code).

See `volume_scalp/indicators.py` and `volume_scalp/strategy.py` for the
exact math, and `tests/test_volume_indicators.py` /
`tests/test_volume_scalp_strategy.py` for hand-verified worked examples of
every rule (breakout fire/block, VWAP gating, both climax-fade directions,
session reset).

**Backtesting:**

```bash
python -m backtest.run_volume_scalp --data sample_data/sample_1min.csv --symbol MES
python -m backtest.run_volume_scalp --data sample_data/sample_1min.csv --symbol MNQ --timeframe 5min

# Compare bar timeframes (1min/3min/5min by default) on the same data:
python -m backtest.compare_volume_scalp --data sample_data/sample_1min.csv --symbol MES
```

Both support `--contracts`, `--daily-loss-limit`, `--max-daily-trades`,
`--target-r`, `--stop-buffer-ticks`, `--volume-window`,
`--breakout-window`, `--delta-window`, `--breakout-rvol`, `--climax-rvol`,
`--climax-wick-pct`, `--no-vwap-filter`, `--no-breakout`,
`--no-climax-fade` — run with `--help` for the full list. Same fill/exit
assumptions and holiday-gap caveat as the Failed-2s backtester (see
"Backtest assumptions/limitations" above).

**Going live:** `tradingview/volume_scalp_mes.pine` — same
TradingView + TradersPost path as the other two strategies (see
"Live trading on Tradovate" below and **TRADINGVIEW_WEBHOOK.md**). Unlike
`structure_scalp_mes.pine`, this runs on a normal (non-seconds) chart
timeframe, so it's fully usable on TradingView's Strategy Tester with
ordinary historical depth. There is currently no Python live runner
(`live/runner.py` is Failed-2s-specific) for this strategy — TradingView is
the only supported live path for now.

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

## Running the tests

```bash
pytest
```
