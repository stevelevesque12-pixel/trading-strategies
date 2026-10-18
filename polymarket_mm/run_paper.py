"""
Paper-trade the market maker against Polymarket's live books.

    # list the best reward markets for the default quote, then exit
    python -m polymarket_mm.run_paper --scan-only

    # paper-trade the top 5 scanner picks for 2 hours
    python -m polymarket_mm.run_paper --markets 5 --minutes 120

    # or pick markets yourself (condition id or slug)
    python -m polymarket_mm.run_paper --market some-market-slug --market 0xabc...

Nothing here signs or sends orders; there are no keys involved. Output is a
JSONL event log in --out (default runs/) and a summary at the end, which you
can regenerate any time with `python -m polymarket_mm.report <log>`.
"""

import argparse
import os
import threading
import time
from datetime import datetime

from .api import fetch_market, fetch_reward_markets, market_info, scan, summarize
from .engine import EventLog, MarketMaker, RiskParams
from .feed import MarketFeed
from .paper import PaperBroker
from .quoting import QuoteParams
from .report import format_summary, load, summary


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", action="append", default=[], help="condition id or slug (repeatable)")
    ap.add_argument("--markets", type=int, default=5, help="number of scanner picks when --market not given")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--include-sports", action="store_true", help="allow markets with a game start time")
    ap.add_argument("--min-days-to-end", type=float, default=2.0)
    ap.add_argument("--max-capital", type=float, default=500.0, help="per-market capital cap for scanner picks")
    g = ap.add_argument_group("quoting")
    g.add_argument("--size", type=float, default=0.0, help="shares per side (min: market reward min_size)")
    g.add_argument("--band-fraction", type=float, default=0.5)
    g.add_argument("--min-edge-ticks", type=int, default=1, help="min distance from mid in ticks (0 = may join the touch)")
    g.add_argument("--skew-per-share", type=float, default=0.00005)
    g.add_argument("--max-position", type=float, default=1000.0)
    g.add_argument("--no-join-only", action="store_true", help="allow improving the best bid/ask")
    g = ap.add_argument_group("risk / simulation")
    g.add_argument("--pull-move", type=float, default=0.02)
    g.add_argument("--cooldown", type=float, default=30.0)
    g.add_argument("--max-loss", type=float, default=50.0, help="per-market loss limit, USDC")
    g.add_argument("--latency", type=float, default=0.25, help="order/cancel latency, seconds")
    g.add_argument("--maker-fee-bps", type=float, default=0.0)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    qp = QuoteParams(size=a.size, band_fraction=a.band_fraction,
                     min_edge_ticks=a.min_edge_ticks, skew_per_share=a.skew_per_share,
                     max_position=a.max_position, join_only=not a.no_join_only)
    risk = RiskParams(pull_move=a.pull_move, cooldown=a.cooldown, max_loss=a.max_loss)

    if a.market:
        infos = [market_info(fetch_market(ref)) for ref in a.market]
    else:
        print("scanning reward markets...")
        picks = scan(fetch_reward_markets(), qp, top_n=a.markets, min_days_to_end=a.min_days_to_end,
                     include_sports=a.include_sports, max_capital=a.max_capital)
        print(summarize(picks))
        infos = [p[0] for p in picks]
    if a.scan_only or not infos:
        return

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"paper_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    log = EventLog(path)
    makers, by_token = [], {}
    for info in infos:
        log.write(type="meta", ts=time.time(), market=info.condition_id, question=info.question,
                  slug=info.slug, daily_rate=info.daily_rate, max_spread=info.max_spread,
                  min_size=info.min_size, tick=info.tick)
        mm = MarketMaker(info, qp, risk, log, PaperBroker(a.latency, a.maker_fee_bps))
        makers.append(mm)
        by_token[info.yes_token] = by_token[info.no_token] = mm

    lock = threading.Lock()

    def on_event(ev):
        now = time.time()
        et = ev.get("event_type")
        with lock:
            if et == "book":
                mm = by_token.get(ev.get("asset_id"))
                if mm:
                    mm.on_book(ev["asset_id"], ev.get("bids", []), ev.get("asks", []), now)
            elif et == "price_change":
                touched = {by_token[c["asset_id"]] for c in ev.get("price_changes", []) if c.get("asset_id") in by_token}
                for mm in touched:
                    mm.on_price_changes(ev["price_changes"], now)
            elif et == "last_trade_price":
                mm = by_token.get(ev.get("asset_id"))
                if mm:
                    mm.on_trade(ev, now)
            elif et == "tick_size_change":
                mm = by_token.get(ev.get("asset_id"))
                if mm and ev.get("new_tick_size"):
                    mm.on_tick_size(ev["new_tick_size"], now)

    feed = MarketFeed(list(by_token), on_event)
    feed.start()
    print(f"paper trading {len(makers)} market(s) for {a.minutes:g} min; log: {path}")
    end, next_status = time.time() + a.minutes * 60, time.time() + 60
    try:
        while time.time() < end:
            time.sleep(2)
            now = time.time()
            with lock:
                for mm in makers:
                    mm.on_timer(now)
                if now >= next_status:
                    next_status = now + 60
                    tot = sum(mm.pnl() for mm in makers)
                    rw = sum(mm.rewards for mm in makers)
                    pos = ", ".join(f"{mm.broker.position:+.0f}" for mm in makers)
                    print(f"[{datetime.now():%H:%M:%S}] trading P&L {tot:+.2f}  est rewards {rw:.2f}  positions [{pos}]")
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        feed.stop()
        with lock:
            for mm in makers:
                mm.flush(time.time())
        log.close()
    print()
    print(format_summary(summary(load(path))))
    print(f"\nlog: {path}  (re-run: python -m polymarket_mm.report {path})")


if __name__ == "__main__":
    main()
