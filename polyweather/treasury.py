"""Turn positions back into cash automatically: merge matched pairs, redeem resolved markets.

* Merge: a bucket where we hold both YES and NO is worth exactly $1 per pair
  whatever happens. Merging returns that $1 now instead of after resolution,
  freeing capital for quoting. The pairs leave both the inventory and the
  ledger, booked as cash at $1 each (realized gain = $1 - what the pair cost).
* Redeem: after a market resolves, winning tokens are worth $1 each but stay
  tokens until redeemed. Without this the bot slowly runs out of cash.

Both go through Polymarket's relayer (gasless) via exchange.ExchangeBroker.
The exchange's own `mergeable` / `redeemable` flags decide what to touch.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .alerts import Alerter
from .ledger import Ledger

TREASURY_SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_treasury (
    ts REAL, action TEXT, condition_id TEXT, slug TEXT, bucket INTEGER, pairs REAL,
    cash REAL, realized REAL, tx TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS pilot_events (ts REAL, kind TEXT, detail TEXT);
"""


@dataclass
class TreasuryConfig:
    merge_every_s: float = 900.0
    redeem_every_s: float = 1800.0
    min_merge_pairs: float = 5.0  # don't spend a relayer call on dust
    max_tx_per_run: int = 10
    give_up_after: int = 3  # failures per condition before alerting and skipping it


@dataclass
class Treasury:
    broker: object
    ledger: Ledger
    alerter: Alerter
    cfg: TreasuryConfig = field(default_factory=TreasuryConfig)
    inventories: Optional[dict] = None  # slug -> maker.Inventory, kept in step with merges
    _last_merge: float = 0.0
    _last_redeem: float = 0.0
    _fails: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ledger.db.executescript(TREASURY_SCHEMA)

    def _log(self, action, cid, slug, bucket, pairs, cash, realized, tx, status):
        self.ledger.db.execute("INSERT INTO pilot_treasury VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (time.time(), action, cid, slug, bucket, pairs, cash, realized, tx, status))
        self.ledger.commit()

    def _failed(self, cid: str, action: str, exc: Exception) -> None:
        n = self._fails.get((action, cid), 0) + 1
        self._fails[(action, cid)] = n
        self._log(action, cid, None, None, 0, 0, 0, None, f"error: {str(exc)[:200]}")
        if n == self.cfg.give_up_after:
            self.alerter.send(f"{action} keeps failing", f"condition {cid}: {exc}", level="urgent")

    def _blocked(self, cid: str, action: str) -> bool:
        return self._fails.get((action, cid), 0) >= self.cfg.give_up_after

    def run(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        do_merge = now - self._last_merge >= self.cfg.merge_every_s
        do_redeem = now - self._last_redeem >= self.cfg.redeem_every_s
        if not (do_merge or do_redeem):
            return {}
        positions = self.broker.positions()
        out = {"merged_pairs": 0.0, "merge_cash": 0.0, "redeemed": 0}
        n_tx = 0
        if do_merge:
            self._last_merge = now
            by_cid = defaultdict(list)
            for p in positions:
                if p.mergeable and not p.archived:
                    by_cid[str(p.condition_id)].append(p)
            for cid, ps in by_cid.items():
                if n_tx >= self.cfg.max_tx_per_run or self._blocked(cid, "merge"):
                    continue
                if len(ps) < 2:
                    continue
                pairs = float(min(p.current_size for p in ps))
                if pairs < self.cfg.min_merge_pairs:
                    continue
                try:
                    outcome = self.broker.merge(cid).wait()
                except Exception as exc:
                    self._failed(cid, "merge", exc)
                    continue
                n_tx += 1
                realized, slug, bucket = self._book_merge(ps, pairs)
                self._log("merge", cid, slug, bucket, pairs, pairs, realized,
                          getattr(outcome, "transaction_hash", None), "ok")
                out["merged_pairs"] += pairs
                out["merge_cash"] += pairs
        if do_redeem:
            self._last_redeem = now
            done = set()
            for p in positions:
                cid = str(p.condition_id)
                if not p.redeemable or cid in done or self._blocked(cid, "redeem") or n_tx >= self.cfg.max_tx_per_run:
                    continue
                try:
                    outcome = self.broker.redeem(cid).wait()
                except Exception as exc:
                    self._failed(cid, "redeem", exc)
                    continue
                n_tx += 1
                done.add(cid)
                value = sum(float(q.current_value) for q in positions if str(q.condition_id) == cid)
                self._log("redeem", cid, p.event_slug, None, 0, value, 0,
                          getattr(outcome, "transaction_hash", None), "ok")
                out["redeemed"] += 1
        if out["merged_pairs"] or out["redeemed"]:
            self.alerter.send("cash returned",
                              f"merged {out['merged_pairs']:.0f} pairs (${out['merge_cash']:.2f}), "
                              f"redeemed {out['redeemed']} resolved markets", level="info")
        return out

    def _book_merge(self, ps, pairs: float):
        """Remove merged pairs from ledger + inventory at average cost; return (realized, slug, bucket)."""
        tokens = {str(p.asset_id) for p in ps}
        rows = [r for r in self.ledger.positions() if r.token_id in tokens]
        if len(rows) < 2:
            return 0.0, None, None
        removed = 0.0
        for r in rows:
            k = min(pairs, r.shares)
            cost = r.cost * k / r.shares if r.shares else 0.0
            removed += cost
            self.ledger.add_fill(r.slug, r.bucket, r.side, r.token_id, r.city, r.kind, r.target_date, r.label,
                                 -k, -cost)
            if self.inventories is not None and r.slug in self.inventories:
                self.inventories[r.slug].add(r.side, r.bucket, -k, -cost)
        self.ledger.db.execute("""INSERT INTO pilot_events VALUES (?,?,?)""",
                               (time.time(), "merge", json.dumps({"slug": rows[0].slug, "pairs": pairs})))
        self.ledger.commit()
        return pairs - removed, rows[0].slug, rows[0].bucket
