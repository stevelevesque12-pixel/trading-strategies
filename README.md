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

## Strategy logic

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
- `tradingview/` + `webhook/` — the other way to go live: a Pine Script port runs on TradingView (which has the market data) and fires a webhook with the signal; `webhook/server.py` sizes the position from a risk budget and places the order on Tradovate. See **TRADINGVIEW_WEBHOOK.md** for setup.
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

Two ways to go live, both ending at the same `live/tradovate_client.py`:

- **TradingView + webhook** (`tradingview/` + `webhook/`) — TradingView runs
  the signal logic (it has live market data), fires a webhook on each
  signal, and `webhook/server.py` sizes the position from a dollar risk
  budget and places the order. This is the path with risk-based contract
  sizing built in. See **TRADINGVIEW_WEBHOOK.md**.
- **Python runner** (`live/runner.py`, below) — runs the same
  `failed2s/strategy.py` logic directly against Tradovate's own market data
  feed, no TradingView involved, fixed 1-contract sizing.

**Read this before pointing either one at a funded account.** The Tradovate client
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
