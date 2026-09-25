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

**A third, unrelated strategy lives in `turtle_trading/`**: Richard
Dennis and William Eckhardt's 1983 Turtle Trading rules, a daily-bar
trend-following system. See "Turtle Trading" below.

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

## Turtle Trading (Richard Dennis)

`turtle_trading/strategy.py` implements the original Turtle rules (per
Curtis Faith's *The Original Turtle Trading Rules*) on **daily** bars:

- **N** = 20-day Wilder ATR. **Unit** = `floor(equity * 1% / (N * point value))`,
  so a 1N move against one unit costs 1% of equity.
- **System 1**: enter on a 20-day breakout, *skipped* if the previous
  20-day breakout would have been a winner (tracked as a shadow trade
  whether or not it was taken); a 55-day breakout is the failsafe entry
  when skipped. Exit on a 10-day opposite breakout.
- **System 2**: enter on every 55-day breakout, exit on a 20-day opposite
  breakout.
- **Stop** 2N from entry. **Pyramid** one more unit every +0.5N from the
  previous fill, max 4 units; every add moves the stop for all units to 2N
  from the newest fill.

Backtest (`turtle_trading/backtest.py`) runs any number of markets as one
portfolio sharing a single realized-equity account. Intraday data is
rolled up into daily bars on the CME 18:00 ET session boundary. The
`SYMBOL=` part only picks the contract spec, so the 10-year full-size
datasets can be traded as micros (needed for 1%-risk sizing on a small
account):

```bash
D=sample_data/real_multi_instrument
python -m turtle_trading.backtest \
  --market MES=$D/real_es_15m_2016-05-29_2026-08-25.parquet \
  --market MNQ=$D/real_nq_15m_2016-05-29_2026-08-25.parquet \
  --market MGC=$D/real_gc_15m_2016-05-26_2026-08-25.parquet \
  --market SIL=$D/real_si_15m_2016-05-26_2026-08-25.parquet \
  --system both --equity 100000
```

Options: `--risk-pct` (default 1.0), `--max-units`, `--no-skip-filter`,
`--slippage-ticks` (default 1 per fill), `--commission` (default $2.50
round trip per contract), `--out` (trade log CSV).

Results on that 4-market basket, 2016-05 to 2026-08, $100k, default costs:

| Risk/unit | System | Trades | Win % | Profit factor | CAGR | Max DD (closed) |
|---|---|---|---|---|---|---|
| 1% | S1 (20/10) | 339 | 23.6 | 0.93 | -4.6% | 62% |
| 1% | S2 (55/20) | 242 | 19.4 | 1.03 | +1.6% | 81% |
| 0.5% | S1 (20/10) | 317 | 24.3 | 1.06 | +1.7% | 33% |
| 0.5% | S2 (55/20) | 219 | 20.6 | 1.26 | +5.9% | 48% |

Gold was the only consistently profitable market (S2 alone on MGC: PF
1.72, +7.9% CAGR); silver lost the most. Turning off System 1's
skip-after-winner filter roughly doubled its losses. Four equity-index/metal
markets is a far narrower basket than the ~20 diversified futures (bonds,
currencies, grains, energies) the Turtles traded, and diversification
across uncorrelated trends is what the system depends on -- read these
numbers as "this basket doesn't give the rules enough to work with," not
as a verdict on the method.

Daily-bar fill assumptions: stop orders fill at the level (or the open on
a gap) plus slippage; open positions check exits before adding units; after
an entry or add the stop/exit channel is re-checked on the same bar and
assumed hit if the bar's range reaches it (conservative); a bar breaking
both channels is ignored for entries. Not modeled: the Turtles'
portfolio-wide unit caps (12 per direction, 6 per correlated group), the
10%-drawdown equity haircut, and roll costs on continuous-contract data.

## Running the tests

```bash
pytest
```
