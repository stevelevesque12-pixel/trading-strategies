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


def plan(track, n, rng):
    done = done_families(track)
    fams = all_families()
    fresh = [f for f in fams if f not in done]
    rng.shuffle(fresh)
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
    args = ap.parse_args()

    rng = random.Random()
    deadline = time.time() + args.minutes * 60
    jobs_done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        while time.time() < deadline:
            batch = []
            for track, k in (("15m_full", args.workers * 2), ("5m_recent", max(1, args.workers * 2 // args.five_min_every))):
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
                if rec["robust"] and rec["track"] == "15m_full":
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
