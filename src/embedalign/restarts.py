"""All-or-nothing: is the failure in the structure or in the search?

`scale.py` turns up something Part 1 never saw. On the Catalan pair the
unsupervised method scores 98.9% at 4,000 tokens, 2.5% at 8,000 and 0.0% at
16,000 -- while supervised Procrustes sits at 98% throughout. A structure
that a single orthogonal matrix maps almost perfectly is being missed by the
search that is supposed to find that matrix.

Self-learning is stochastic, so the obvious test is to run it many times.
Two things matter:

  * is the outcome **bimodal** -- near-100% or near-0%, with nothing between?
  * does the method's **own unsupervised objective** tell the good runs from
    the bad ones?  If it does, restarts are a fix and not a cheat, because
    picking the best run needs no supervision.  If it does not, the method is
    unusable no matter how many times you run it.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "wordalign"))

import align  # noqa: E402
import experiment  # noqa: E402
import models  # noqa: E402
from scale import retrieval_chunked  # noqa: E402

OBJ_ROWS = 6000  # rows used to score a run's own objective


def unsupervised_objective(x, y, q, m=OBJ_ROWS):
    """Mean CSLS score of each row's best match -- what the method optimises.

    Computed over a fixed prefix so that runs at different vocabulary sizes
    stay comparable and no 44k x 44k matrix has to exist.
    """
    d = min(x.shape[1], y.shape[1])
    m = min(m, x.shape[0], y.shape[0])
    sim = align.csls_scores(np.asarray(x[:m, :d] @ q, dtype=np.float32),
                            np.asarray(y[:m, :d], dtype=np.float32))
    return float(sim.max(axis=1).mean())


def sweep_restarts(a, b, sizes, tries=8, limit=60000, cut=6000, verbose=True):
    rows = []
    for n in sizes:
        pair = experiment.build_pair(a, b, limit=limit, matched=True, n_max=n)
        got = pair.meta["n_x"]
        if rows and got == rows[-1]["n"]:
            break
        runs = []
        t0 = time.time()
        for s in range(tries):
            q = align.vecmap_unsupervised(pair.x, pair.y,
                                          cut=min(cut, got), seed=s)
            r, _ = retrieval_chunked(pair.x, pair.y, q, pair.gold, ks=(1,))
            runs.append({"seed": s, "top1": r[1],
                         "objective": unsupervised_objective(pair.x, pair.y, q)})
        best = max(runs, key=lambda r: r["objective"])
        row = {"n": got, "runs": runs,
               "best_by_objective_top1": best["top1"],
               "best_by_objective_seed": best["seed"],
               "mean_top1": float(np.mean([r["top1"] for r in runs])),
               "max_top1": float(max(r["top1"] for r in runs)),
               "n_locked_on": int(sum(r["top1"] > 0.5 for r in runs)),
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        if verbose:
            scores = " ".join(f"{r['top1'] * 100:5.1f}" for r in runs)
            print(f"  n={got:6d}  top-1 per restart: [{scores} ]", flush=True)
            print(f"           {row['n_locked_on']}/{tries} locked on; "
                  f"picking the best by the method's own objective gives "
                  f"{row['best_by_objective_top1'] * 100:.1f}%", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/embedalign-results/restarts.json")
    ap.add_argument("--sizes", default="1000,2000,3000,4000,6000")
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--pairs", default="ca-plantl|ca-aina")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    ms = {n: models.load(n) for n in models.MODELS}
    out = {"tries": args.tries}
    for spec in args.pairs.split(","):
        a, b = spec.split("|")
        print(f"\n#### {a} / {b}: {args.tries} restarts at each size\n",
              flush=True)
        out[f"{a}|{b}"] = sweep_restarts(ms[a], ms[b], sizes,
                                         tries=args.tries)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
