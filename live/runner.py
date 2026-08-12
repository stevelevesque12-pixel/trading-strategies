"""
Live runner: wires the Failed-2s strategy + risk manager to Tradovate.

Safety defaults: dry-run (log signals, place no orders) and the demo
environment, both must be explicitly overridden to trade real money.

Usage:
    python -m live.runner --pair 5m-1h --symbol MES --account-spec myuser
    python -m live.runner --pair 5m-1h --symbol MES --account-spec myuser --live --env live

This has not been validated end-to-end against Tradovate from this
environment -- see live/tradovate_client.py for details. Test thoroughly on
demo before pointing this at a funded/live account.
"""

import argparse
import sys
from datetime import datetime, date as date_cls

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy, SessionConfig

from .tradovate_client import TradovateChartFeed, TradovateClient

# Ratio of bias_tf minutes to entry_tf minutes, used to aggregate incoming
# entry-tf bars into bias-tf bars locally (avoids a second WS subscription).
_TF_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240}


class LiveRunner:
    def __init__(self, pair_name: str, symbol: str, account_spec: str, dry_run: bool = True, env: str = "demo"):
        self.pair = PAIRS[pair_name]
        self.instrument = INSTRUMENTS[symbol]
        self.symbol = symbol
        self.account_spec = account_spec
        self.dry_run = dry_run

        self.session = SessionConfig()
        self.strategy = Failed2sStrategy(tick_size=self.instrument.tick_size, session=self.session)
        self.risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=3)

        self.client = TradovateClient(env=env)
        self.account_id = None

        self._entry_agg_start = None
        self._entry_agg = None  # dict o/h/l/c/v being built from finer WS bars
        self._bias_agg_start = None
        self._bias_agg = None
        self._position_open = False

    def start(self) -> None:
        if not self.dry_run:
            self.client.authenticate()
            accounts = self.client.list_accounts()
            if not accounts:
                raise RuntimeError("No Tradovate accounts returned for this login")
            self.account_id = accounts[0]["id"]
            print(f"Authenticated. Using account_id={self.account_id} env={self.client.env}")
        else:
            print("DRY RUN: no orders will be sent. Pass --live --env live to trade for real.")

        entry_minutes = _TF_MINUTES[self.pair.entry_tf]
        feed = TradovateChartFeed(self.client, on_bar=lambda raw: self._on_raw_bar(raw, entry_minutes))
        feed.subscribe_chart(self.symbol, element_size_minutes=entry_minutes)

    def _on_raw_bar(self, raw: dict, entry_minutes: int) -> None:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        bar = Bar(ts, raw["open"], raw["high"], raw["low"], raw["close"], raw.get("upVolume", 0) + raw.get("downVolume", 0))

        bias_minutes = _TF_MINUTES[self.pair.bias_tf]
        self._aggregate(bar, entry_minutes, is_bias=False)
        self._aggregate(bar, bias_minutes, is_bias=True)

    def _aggregate(self, bar: Bar, period_minutes: int, is_bias: bool) -> None:
        bucket = bar.timestamp.replace(
            minute=(bar.timestamp.minute // period_minutes) * period_minutes, second=0, microsecond=0
        )
        agg_start_attr = "_bias_agg_start" if is_bias else "_entry_agg_start"
        agg_attr = "_bias_agg" if is_bias else "_entry_agg"
        agg_start = getattr(self, agg_start_attr)
        agg = getattr(self, agg_attr)

        if agg_start is not None and bucket != agg_start:
            finished = Bar(agg_start, agg["open"], agg["high"], agg["low"], agg["close"], agg["volume"])
            self._on_finished_bar(finished, is_bias)
            agg = None

        if agg is None:
            agg = {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume}
        else:
            agg["high"] = max(agg["high"], bar.high)
            agg["low"] = min(agg["low"], bar.low)
            agg["close"] = bar.close
            agg["volume"] += bar.volume

        setattr(self, agg_start_attr, bucket)
        setattr(self, agg_attr, agg)

    def _on_finished_bar(self, bar: Bar, is_bias: bool) -> None:
        if is_bias:
            self.strategy.on_bias_bar(bar)
            return

        today: date_cls = bar.timestamp.date()

        if self._position_open and bar.timestamp.time() >= self.session.flatten_at:
            print(f"[{bar.timestamp}] Session flatten cutoff reached -- liquidating.")
            if not self.dry_run:
                self.client.liquidate_position(self.account_id, self.symbol)
            self._position_open = False

        signal = self.strategy.on_entry_bar(bar)
        if signal is None:
            return

        if self._position_open or not self.risk.can_enter(today):
            print(f"[{bar.timestamp}] Signal {signal.direction} ignored (position open or risk-locked).")
            return

        print(
            f"[{signal.timestamp}] SIGNAL {signal.direction} entry={signal.entry_price} "
            f"stop={signal.stop_price} target={signal.target_price} ({signal.reason})"
        )
        if self.dry_run:
            return

        action = "Buy" if signal.direction == "long" else "Sell"
        self.client.place_bracket_order(
            account_id=self.account_id,
            account_spec=self.account_spec,
            symbol=self.symbol,
            action=action,
            qty=1,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
        )
        self._position_open = True
        # Note: PnL isn't recorded into self.risk here since fills/exits are
        # reported asynchronously by Tradovate; wire a fill/exit callback
        # from the user-data WS to call self.risk.record_trade(...) before
        # relying on the daily-loss-limit lockout in live trading.


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Failed-2s strategy live against Tradovate")
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--account-spec", required=True, help="Tradovate account username/spec")
    parser.add_argument("--live", action="store_true", help="Actually send orders (default: dry-run/log-only)")
    parser.add_argument("--env", default="demo", choices=["demo", "live"])
    args = parser.parse_args()

    if args.env == "live" and not args.live:
        print("Refusing to connect to the live Tradovate environment without --live.", file=sys.stderr)
        sys.exit(1)

    runner = LiveRunner(
        pair_name=args.pair,
        symbol=args.symbol,
        account_spec=args.account_spec,
        dry_run=not args.live,
        env=args.env,
    )
    runner.start()


if __name__ == "__main__":
    main()
