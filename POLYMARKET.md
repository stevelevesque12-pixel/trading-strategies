# Polymarket paper market maker (`polymarket_mm/`)

Step 1 of 3 toward a Polymarket market-making bot: **paper trading only**.
It reads Polymarket's live public order books and trades, decides two-sided
quotes, and simulates fills against what actually prints. There are no API
keys, no wallet, and no real orders anywhere in this code.

The plan:
1. **Paper market maker** (this). Collect fills and estimated rewards on live markets.
2. **Markout and P&L report** (`polymarket_mm/report.py`, included). Decide whether spread plus rewards beats adverse selection.
3. **Live trading at small size**, only if step 2 says the edge is real. This means adding order signing via `py-clob-client` with hard limits and a kill switch.

## Quick start

```bash
pip install -r requirements.txt

# 1. See which reward markets look attractive for the default quote right now
python -m polymarket_mm.run_paper --scan-only

# 2. Paper-trade the top 5 scanner picks for 2 hours (Ctrl-C stops early and still prints the report)
python -m polymarket_mm.run_paper --markets 5 --minutes 120

# ...or choose markets yourself (slug from the polymarket.com URL, or a 0x condition id)
python -m polymarket_mm.run_paper --market will-benjamin-netanyahu-be-the-next-prime-minister-of-israel

# 3. Re-read any run's log later, optionally dumping fills with markouts to CSV
python -m polymarket_mm.report runs/paper_20261018_170311.jsonl --fills-csv fills.csv
```

Run it for days, not minutes. Most quiet reward markets get no fills at all
in a short session, and a few hours of data can't tell you about the news
events that make or break this strategy.

## What it does

**Market selection** (`api.py`): it pulls every market currently paying
liquidity rewards (about 10k as of Oct 2026), then drops:
- markets resolving within 2 days,
- markets with a game start time (live sports, where makers get run over),
- markets priced outside 0.10–0.90.

It then ranks the rest by the reward our default quote would earn at
today's competition.

**Quoting** (`quoting.py`): it quotes the YES token at
`band_fraction × max_spread` from the midpoint (default: halfway out to the
reward band). Defaults:
- Quotes join existing price levels and never step ahead of the best bid or ask.
- Quotes skew against inventory: long → lower prices.
- Size is the market's reward `min_size` (smaller orders earn nothing).
- No quotes if the book is wider than 5¢.

A negative YES position is economically the same as holding NO, which is
how a real "sell without inventory" fills on Polymarket.

**Risk** (`engine.py`):
- If the mid moves ≥ 2¢ within 10 s, all quotes are pulled for a 30 s cooldown.
- A market stops quoting once its P&L, including estimated rewards, is below −$50.
- Requotes happen at most once a second.

**Fill model** (`paper.py`):
- A new order joins the back of the queue at its price.
- Real trades at our price eat the queue ahead of us first, then us.
- A trade through our price, or the book crossing it, fills us in full.
- Orders and cancels take 250 ms to land, so a stale quote can get picked off while its cancel is in flight.
- Cancels by others ahead of us are **not** credited, so passive fills are under-counted. Trade-through fills (the toxic ones) are counted in full.

This leans toward showing adverse selection rather than hiding it.

**Rewards** (`rewards.py`): Polymarket's published scoring,
`((max_spread − distance)/max_spread)² × size`, with the two-sided
`Q_min` rule. Competing makers are approximated from aggregate book
levels, which overstates them, so the estimate is conservative.
Treat `est_rewards` as a rough estimate, not a payout.

## Reading the report

| column | meaning |
|---|---|
| `trading_pnl` | cash from fills plus the open position marked at the last mid (fees included if `--maker-fee-bps`) |
| `est_rewards` | estimated liquidity rewards earned while quoting |
| `mo_Xs_c` | size-weighted markout X seconds after each fill, cents/share. Buy: mid(t+X) − price; sell: price − mid(t+X). Positive is good. |

How to read it:
- **`mo_0s_c`** is the edge captured against the mid at fill time.
- **If markouts fall from `mo_0s_c` to negative by `mo_300s_c`**, the people
  hitting you know something. The strategy only works if
  `est_rewards + spread captured` stays clearly above that loss across many
  days and markets.
- **Where losses concentrate:** look at which markets, and which times
  (around news), the bad markouts cluster in.

## What we learned from the first live runs (Oct 18 2026)

- **Busy markets:** on big, liquid markets (e.g. the Brazil election, Israel
  PM), there are about 70k shares of competing quotes within 3¢ of the mid.
  A 100–150 share quote earns roughly 0.02–0.05% of the reward pool, i.e.
  cents per day. Those markets are tight (1¢), so a quote 3¢ out rarely fills.
- **Quiet markets:** small or quiet reward markets can show $10–80/day
  estimated for ~$190 of capital because almost nobody else is quoting. The
  catch is that the competition can show up at any time. Thin markets can
  also gap on news, and the fill model only partly captures the risk of
  being the only liquidity there.

## Not modelled / known gaps

- Market resolution: we avoid markets resolving within 2 days but don't
  simulate payout.
- Reward details beyond the published formula: the adjusted midpoint,
  per-epoch sampling, the per-order `min_size` of other makers.
- Taker fees: we never take, but check fees per market before going live.
  `fee_rate_bps` on each trade print shows what the venue charged.
- Eligibility: Polymarket's international exchange restricts US persons;
  US users go through Polymarket's US entity, which has its own market list
  and rules. Don't use a VPN to get around restrictions.
