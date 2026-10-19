"""SQLite ledger: every forecast, every model price, every order, every position.

The forecast log is not just bookkeeping -- it is the calibration dataset.
Live ensemble spreads aren't archived anywhere public, so the only way to
calibrate them honestly is to record what we saw and score it once the
market resolves (`python -m polyweather calibrate --from-ledger`).
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    ts REAL, slug TEXT, city TEXT, kind TEXT, target_date TEXT, lead INTEGER, source TEXT,
    center REAL, spread REAL, kernel_sd REAL, observed_extreme REAL, n_obs INTEGER
);
CREATE TABLE IF NOT EXISTS model_probs (
    ts REAL, slug TEXT, bucket INTEGER, label TEXT, model_p REAL, blended_p REAL,
    yes_bid REAL, yes_ask REAL, no_ask REAL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, mode TEXT, slug TEXT, city TEXT, kind TEXT,
    target_date TEXT, bucket INTEGER, label TEXT, side TEXT, token_id TEXT, limit_price REAL,
    shares REAL, filled_shares REAL, cost REAL, fee REAL, model_p REAL, edge REAL,
    status TEXT, exchange_id TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    slug TEXT, bucket INTEGER, side TEXT, token_id TEXT, city TEXT, kind TEXT, target_date TEXT,
    label TEXT, shares REAL, cost REAL, settled INTEGER DEFAULT 0, payout REAL DEFAULT 0,
    PRIMARY KEY (slug, bucket, side)
);
CREATE TABLE IF NOT EXISTS resolutions (
    slug TEXT PRIMARY KEY, city TEXT, kind TEXT, target_date TEXT, winner INTEGER, value REAL, ts REAL
);
"""


@dataclass
class Position:
    slug: str
    bucket: int
    side: str  # "YES" / "NO"
    token_id: str
    city: str
    kind: str
    target_date: str
    label: str
    shares: float
    cost: float
    settled: bool
    payout: float


class Ledger:
    def __init__(self, path: str = "polyweather.db"):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- logging
    def log_forecast(self, slug, city, kind, target_date, lead, source, center, spread, kernel_sd,
                     observed_extreme, n_obs):
        self.db.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), slug, city, kind, str(target_date), lead, source, center, spread,
                         kernel_sd, observed_extreme, n_obs))

    def log_probs(self, slug, rows):
        now = time.time()
        self.db.executemany("INSERT INTO model_probs VALUES (?,?,?,?,?,?,?,?,?)",
                            [(now, slug, *r) for r in rows])

    def record_order(self, **kw) -> int:
        kw.setdefault("ts", time.time())
        cols = ",".join(kw)
        cur = self.db.execute(f"INSERT INTO orders ({cols}) VALUES ({','.join('?' * len(kw))})",
                              list(kw.values()))
        self.db.commit()
        return int(cur.lastrowid)

    def add_fill(self, slug, bucket, side, token_id, city, kind, target_date, label, shares, cost):
        self.db.execute(
            """INSERT INTO positions (slug,bucket,side,token_id,city,kind,target_date,label,shares,cost)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(slug,bucket,side) DO UPDATE SET shares=shares+excluded.shares,
               cost=cost+excluded.cost""",
            (slug, bucket, side, token_id, city, kind, str(target_date), label, shares, cost))
        self.db.commit()

    def record_resolution(self, slug, city, kind, target_date, winner, value=None):
        self.db.execute("INSERT OR REPLACE INTO resolutions VALUES (?,?,?,?,?,?,?)",
                        (slug, city, kind, str(target_date), winner, value, time.time()))
        self.db.commit()

    def commit(self):
        self.db.commit()

    # -- queries
    def positions(self, open_only: bool = True) -> list[Position]:
        q = "SELECT slug,bucket,side,token_id,city,kind,target_date,label,shares,cost,settled,payout FROM positions"
        if open_only:
            q += " WHERE settled=0"
        return [Position(*r[:10], bool(r[10]), r[11]) for r in self.db.execute(q)]

    def settle(self, slug: str, winner: int) -> float:
        pnl = 0.0
        for p in [p for p in self.positions() if p.slug == slug]:
            won = (p.bucket == winner) if p.side == "YES" else (p.bucket != winner)
            payout = p.shares if won else 0.0
            pnl += payout - p.cost
            self.db.execute("UPDATE positions SET settled=1, payout=? WHERE slug=? AND bucket=? AND side=?",
                            (payout, p.slug, p.bucket, p.side))
        self.db.commit()
        return pnl

    def realized_pnl(self, since_ts: Optional[float] = None) -> float:
        q = "SELECT COALESCE(SUM(payout-cost),0) FROM positions WHERE settled=1"
        return float(self.db.execute(q).fetchone()[0])

    def forecast_samples(self, max_age_days: int = 120):
        """(city, kind, source, lead, center, spread, observed) for resolved events.

        Uses the last pre-observation forecast per (event, source, lead) so a
        source isn't over-weighted by how often the bot happened to poll.
        """
        q = """
        SELECT f.city, f.kind, f.source, f.lead, f.center, f.spread, r.value
        FROM forecasts f JOIN resolutions r ON r.slug = f.slug
        WHERE r.value IS NOT NULL AND f.n_obs = 0 AND f.ts > ?
          AND f.ts = (SELECT MAX(f2.ts) FROM forecasts f2 WHERE f2.slug=f.slug AND f2.source=f.source
                      AND f2.lead=f.lead AND f2.n_obs=0)
        """
        return list(self.db.execute(q, (time.time() - max_age_days * 86400,)))
