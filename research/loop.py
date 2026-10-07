"""Run the strategy search for a fixed wall-clock budget, regenerating the dashboard after every result.

    python -m research.loop --minutes 100 --workers 4

Untested families are run first (shuffled); once every family on a track has a
result, families are re-run with fresh seeds (a different random search path),
which is how the leaderboard keeps improving over a long loop.
"""

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from research import dashboard, validate
from research.search import all_families, append, done_families, optimise_family


def _job(args):
    fam, track, seed = args
    try:
        return optimise_family(fam, track=track, seed=seed)
    except Exception as e:  # keep the loop alive; log the failure
        return {"error": repr(e), "family": fam, "track": track}


def promising_families(track):
    """Families with at least one robust result on the current engine version."""
    import json
    from research.search import RESULTS, VERSION
    out = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("v") == VERSION and d["track"] == track and d["robust"]:
                s = d["spec"]
                out.add((s["trend"], s["conf1"], s["conf2"], s["regime"]))
    return sorted(out)


FOCUS = []


def plan(track, n, rng):
    done = done_families(track)
    fams = all_families()
    if FOCUS:  # only families that use at least one focus component
        fams = [f for f in fams if any(c in f for c in FOCUS)]
    fresh = [f for f in fams if f not in done]
    rng.shuffle(fresh)
    # a quarter of each batch re-runs promising families from new seeds: robustness across
    # independent re-optimisations is the best evidence that a family isn't a fluke
    prom = promising_families(track)
    if prom:
        k = max(1, n // 4)
        fresh = [rng.choice(prom) for _ in range(k)] + fresh
    if len(fresh) < n:
        extra = fams[:]
        rng.shuffle(extra)
        fresh += extra
    return fresh[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--five-min-every", type=int, default=6, help="every Nth job runs on the 5m MES track")
    ap.add_argument("--focus", default="", help="comma-separated component names to restrict new families to")
    args = ap.parse_args()
    FOCUS[:] = [c for c in args.focus.split(",") if c]

    rng = random.Random()
    deadline = time.time() + args.minutes * 60
    jobs_done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        while time.time() < deadline:
            batch = []
            for track, k in (("5m_scalp", args.workers * 2), ("15m_fast", max(1, args.workers // 2))):
                batch += [(f, track, rng.randrange(10**9)) for f in plan(track, k, rng)]
            futs = [ex.submit(_job, b) for b in batch]
            for fu in as_completed(futs):
                rec = fu.result()
                if rec is None:
                    continue
                if "error" in rec:
                    print("ERROR", rec, flush=True)
                    continue
                append(rec)
                jobs_done += 1
                if rec["robust"] and rec["track"] in ("15m_full", "15m_joint", "15m_fast"):
                    try:
                        validate.save(rec)  # auto stress-test every robust 10-year result
                    except Exception as e:
                        print("validate failed", rec["id"], repr(e), flush=True)
                print(f"[{time.strftime('%H:%M:%S')}] {rec['track']:<10} {rec['name']:<45} "
                      f"IS PF {rec['is']['profit_factor']:.2f} OOS PF {rec['oos']['profit_factor']:.2f} "
                      f"OOS pass {rec['oos']['lucid']['pass_rate']:.0%} robust={rec['robust']}", flush=True)
                dashboard.build()
    print(f"done: {jobs_done} strategies logged", flush=True)


if __name__ == "__main__":
    main()
