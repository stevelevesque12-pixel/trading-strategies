"""Replay the reverse-engineered ORB entry on the 5-minute bars in
sample_data/bars and compare exit rules across MNQ, MES, MCL and MGC.

Usage:  python -m orb_lab.run [--cost 1.5]
"""
import argparse

import numpy as np

from orb_lab.engine import BARS, Market, load_bars, signals, simulate

MARKETS = [
    Market("MNQ", BARS / "MNQ1_5m_2026-05-10_08-25.csv", 2.0, 0.25),
    Market("MES", BARS / "MES1_5m_2026-05-10_08-25.csv", 5.0, 0.25),
    Market("MCL", BARS / "MCL1_5m_2026-05-10_08-25.csv", 100.0, 0.01),
    Market("MCL@09:00", BARS / "MCL1_5m_2026-05-10_08-25.csv", 100.0, 0.01, open_hm="09:00", last_entry="11:30"),
    Market("MGC", BARS / "MGC1_5m_2026-05-10_08-25.csv", 10.0, 0.1),
    Market("MGC@08:20", BARS / "MGC1_5m_2026-05-10_08-25.csv", 10.0, 0.1, open_hm="08:20", last_entry="10:50"),
]
RULES = {  # (stop, target) as multiples of the opening-range height
    "stop .75xOR / tgt .15xOR": (0.75, 0.15),
    "stop .5xOR / tgt .15xOR": (0.5, 0.15),
    "stop .35xOR / tgt .15xOR": (0.35, 0.15),
    "stop .5xOR / tgt .1xOR": (0.5, 0.1),
    "stop .5xOR / tgt .25xOR": (0.5, 0.25),
    "stop .5xOR / tgt .5xOR (1:1)": (0.5, 0.5),
    "stop .5xOR / tgt 1xOR (1:2)": (0.5, 1.0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost", type=float, default=1.5, help="$ per micro round trip")
    a = ap.parse_args()
    for m in MARKETS:
        b = load_bars(m)
        sg = signals(b, m)
        cost = a.cost / m.point_value
        rng = np.median([s[3] - s[4] for s in sg])
        print(f"== {m.name}: {len(sg)} signals, median opening range {rng:.2f} pts (${rng * m.point_value:.0f}/micro)")
        for name, (ks, kt) in RULES.items():
            t = simulate(b, m, lambda c, ks=ks, kt=kt: dict(stop=ks * c["rng"], target=kt * c["rng"]), sigs=sg)
            usd = (t.pts - cost) * m.point_value
            print(f"   {name:30s} win {np.mean(usd > 0) * 100:3.0f}%  avgW ${usd[usd > 0].mean():6.1f}"
                  f"  avgL ${usd[usd < 0].mean():7.1f}  exp ${usd.mean():6.2f}/trade  total ${usd.sum():7.0f}")


if __name__ == "__main__":
    main()
