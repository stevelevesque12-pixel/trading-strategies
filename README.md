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

**A third strategy: `tr_breakout/`** -- a daily volatility breakout on
MNQ. Stop-entry orders at RTH open +/- 0.25 x the prior session's true range
(TR1), active 09:30-13:00 ET; 1R stop = 0.25 x TR1; stop to break-even from
the bar after a completed bar reaches +2R; exit on stop/BE or at 15:55 ET;
re-entry after a stop once price crosses back through the level (max 3
entries/day); fixed $1,000 risk per trade (1% of $100k, not compounded --
`--compound` sizes off current equity instead); $0.95
RT commission + 1 tick slippage per side. Assumptions the spec leaves open
(RTH vs full-Globex TR1, intrabar ordering on OHLC bars, 15:55 exit on 15m
bars) are documented at the top of `tr_breakout/strategy.py`.

```bash
python -m tr_breakout.run --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet
```

Result on 10 years of NQ 15m bars (sized as MNQ at $2/pt, fixed $1,000
risk): 3,731 trades, 32% win rate, +0.15R avg, PF 1.27, +$550k net
(~$55k/yr on $100k), max drawdown -$58k (2016-05 to 2017-05, right at the
start -- equity bottomed at ~$42k). Losing years were 2016 (-$15k) and 2017
(-$4k), when TR was small in points and fixed costs were ~0.1R per trade;
2026 YTD is +$13k with May-Aug -$14k. Bucketing by TR1 as a % of price
does *not* show "higher volatility = better": the top quintile was the
weakest (+0.03R), so the weak early years look driven more by cost per R
than by volatility. On the overlapping periods, 15m, 5m and 1m data give
nearly identical results, so the intrabar-path assumption isn't driving the
numbers.

**Prop-firm check:** `python -m tr_breakout.prop_sim --data <parquet> --from 2022-01-01`
replays the trade stream from every start day through a generic 50K
evaluation (+$3,000 target, $2,000 EOD trailing max loss, 50 MNQ cap -- set
the flags to your firm's rules) at several fixed $-risk levels. Since 2022:
~71% pass at $200/trade, ~63% at $250, ~49% at $400 (a zero-edge coin flip
passes ~40% with a 3k/2k target/drawdown). Below ~$200 risk, 1 MNQ is often
too big for the stop at current NQ prices and most days get skipped.

**Monte Carlo:** `python -m tr_breakout.monte_carlo --data <parquet> --from 2022-01-01 --block 20`
resamples whole trading days (20-day blocks keep streaks/regimes clustered)
into 20k random paths: 1-year P&L/drawdown for a $100k account at fixed
$1,000 risk, and prop-eval pass odds per $ risk level. `--haircut 0.5`
stress-tests with half the historical edge.

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

**`sample_data/real_multi_instrument/`** — real CME/COMEX/CBOT/NYMEX
futures data supplied directly by the user (not via `fetch_real_data.py`/
`fetch_yfinance.py`), stored as parquet rather than CSV specifically so it
sidesteps `.gitignore`'s `sample_data/real_*.csv` rule (that rule exists
on purpose - real data is normally meant to be fetched on demand, not
committed - see the git history for the discussion before adding more
here). `--data` now accepts these directly; `backtest/data.py`'s
`load_1m_csv` dispatches on file extension.

- Micro contracts (MES, MNQ, MYM, MGC, MCL, SIL, M2K) at 1m/5m/15m -
  depth is per-interval, shared across all seven symbols: 1m covers
  2026-08-02 to 2026-08-25 (~3.5 weeks), 5m covers 2026-05-10 to
  2026-08-25 (~3.5 months), 15m covers 2025-09-30 to 2026-08-25
  (~11 months).
- Full-size contracts (ES, NQ, GC, SI) at 15m only, 2016-05 to 2026-08
  (~10 years, ~240k bars each).

Only the 1m files are genuine 1-minute data - the `1m-15m` pair needs
that resolution specifically. The 5m/15m files load and resample fine
for `5m-1h`/`15m-4h` (their OHLC is just built from coarser source bars
than a true 1-minute feed would give you), but don't point them at
`1m-15m` - the resample can't invent finer bars than the file has.

Provenance beyond "the user supplied these directly" wasn't stated and
hasn't been independently verified (which vendor, whether `volume` is
real exchange volume) - treat results from this data as "real market
data, source TBD" rather than production-grade until that's confirmed,
same caveat as the OANDA data above.

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
