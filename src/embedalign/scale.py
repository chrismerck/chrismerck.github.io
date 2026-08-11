"""How far does the unsupervised search get if you give it the whole vocabulary?

Part 1's headline number came from fitting the rotation over a 20,000-word
space, and Part 1's sharpest finding was that everything below about 2,000
items fails regardless of how much text the vectors came from.  The main
comparison run here is capped at 2,000 rows so that Gromov-Wasserstein can
be included at all, which puts it exactly at that cliff edge.

This script removes the cap.  Two models with 44,000 shared token forms can
be aligned over all of them, so the question "is this a search failure or a
structure failure?" has a direct answer: hand the search more constraints and
see whether it locks on.

Retrieval is computed in chunks -- a 40,000 x 40,000 similarity matrix is
6 GB in float32 and there is no reason to hold one.
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


def retrieval_chunked(x, y, q, gold, ks=(1, 5, 10), chunk=2048):
    """Top-k accuracy of x @ q against y, without materialising the full sim."""
    d = min(x.shape[1], y.shape[1])
    xq = np.ascontiguousarray((x[:, :d] @ q), dtype=np.float32)
    yy = np.ascontiguousarray(y[:, :d], dtype=np.float32)
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    n_scored = 0
    for lo in range(0, xq.shape[0], chunk):
        hi = min(lo + chunk, xq.shape[0])
        g = gold[lo:hi]
        ok = g >= 0
        if not ok.any():
            continue
        sim = xq[lo:hi][ok] @ yy.T
        top = np.argpartition(-sim, kmax, axis=1)[:, :kmax]
        order = np.take_along_axis(sim, top, 1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        truth = g[ok][:, None]
        for k in ks:
            hits[k] += int((top[:, :k] == truth).any(axis=1).sum())
        n_scored += int(ok.sum())
    return {k: hits[k] / max(n_scored, 1) for k in ks}, n_scored


def run_scale(a, b, sizes, limit=60000, cut=6000, seed=0, verbose=True,
              shuffle=False):
    rows = []
    for n in sizes:
        pair = experiment.build_pair(a, b, limit=limit, matched=True,
                                     n_max=n, seed=seed, shuffle=shuffle)
        got = pair.meta["n_x"]
        if rows and got == rows[-1]["n"]:
            break  # ran out of shared tokens
        st = experiment.structure(pair)

        t0 = time.time()
        q = align.vecmap_unsupervised(pair.x, pair.y,
                                      cut=got if cut is None else min(cut, got),
                                      seed=seed)
        uns, _ = retrieval_chunked(pair.x, pair.y, q, pair.gold)
        t_uns = time.time() - t0

        idx = np.flatnonzero(pair.gold >= 0)
        rng = np.random.default_rng(seed)
        sel = rng.permutation(idx)[: len(idx) // 2]
        seeds = [(int(i), int(pair.gold[i])) for i in sel]
        d = min(pair.x.shape[1], pair.y.shape[1])
        qs = align.orthogonal_procrustes(
            pair.x[[p[0] for p in seeds], :d],
            pair.y[[p[1] for p in seeds], :d])
        held = pair.gold.copy()
        held[sel] = -1
        sup, n_held = retrieval_chunked(pair.x, pair.y, qs, held)

        row = {"n": got, "pearson": st["pearson"],
               "neighbourhood_overlap": st["neighbourhood_overlap"],
               "unsupervised": {str(k): v for k, v in uns.items()},
               "supervised": {str(k): v for k, v in sup.items()},
               "n_seeds": len(seeds), "n_held": n_held,
               "seconds": round(t_uns, 1)}
        rows.append(row)
        if verbose:
            print(f"  n={got:6d}  r={st['pearson']:.3f}   "
                  f"unsupervised P@1 {uns[1] * 100:5.1f}%  P@10 "
                  f"{uns[10] * 100:5.1f}%   supervised P@1 "
                  f"{sup[1] * 100:5.1f}%   ({t_uns:.0f}s)", flush=True)
    return rows


def restarts(a, b, n, tries=8, limit=60000, cut=6000, verbose=True):
    """Is the unsupervised failure a search failure? Try more seeds.

    The self-learning objective is unsupervised, so picking the best run by
    its own objective smuggles in nothing.  If some restarts succeed and
    others do not, the problem is the search; if they all agree on a bad
    answer, the problem is the structure.
    """
    pair = experiment.build_pair(a, b, limit=limit, matched=True, n_max=n)
    out = []
    for s in range(tries):
        q = align.vecmap_unsupervised(pair.x, pair.y,
                                      cut=min(cut, pair.meta["n_x"]), seed=s)
        r, _ = retrieval_chunked(pair.x, pair.y, q, pair.gold, ks=(1,))
        d = min(pair.x.shape[1], pair.y.shape[1])
        # the objective is only used to rank restarts against each other, so
        # a fixed subset keeps a 44k x 44k float32 matrix off the heap
        m = min(pair.meta["n_x"], 8000)
        sim = align.csls_scores(
            np.asarray(pair.x[:m, :d] @ q, dtype=np.float32),
            np.asarray(pair.y[:m, :d], dtype=np.float32))
        obj = float(sim.max(axis=1).mean())
        out.append({"seed": s, "top1": r[1], "objective": obj})
        if verbose:
            print(f"    seed {s}  objective {obj:8.4f}  top-1 "
                  f"{r[1] * 100:5.1f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/embedalign-results/scale.json")
    ap.add_argument("--sizes", default="1000,2000,4000,8000,16000,32000,60000")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    ms = {n: models.load(n) for n in models.MODELS}
    out = {}
    for a, b in [("ca-plantl", "ca-aina"), ("da-botxo", "da-danskbert"),
                 ("en-roberta", "fr-camembert")]:
        print(f"\n#### {a} / {b}: unsupervised vs supervised against size\n",
              flush=True)
        out[f"{a}|{b}"] = run_scale(ms[a], ms[b], sizes)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)

    # Control: shuffle one side so that no method can see frequency order at
    # all, and fit on the whole vocabulary rather than a frequent prefix.
    # If the headline survives this, the frequency prior is not doing the work.
    print("\n#### control: order destroyed, rotation fit on everything\n",
          flush=True)
    out["ca-plantl|ca-aina|shuffled"] = run_scale(
        ms["ca-plantl"], ms["ca-aina"], [2000, 4000, 8000], cut=None,
        shuffle=True)

    print("\n#### restarts at the largest Catalan size\n", flush=True)
    best = max(out["ca-plantl|ca-aina"], key=lambda r: r["n"])
    out["restarts"] = {"pair": "ca-plantl|ca-aina", "n": best["n"],
                       "runs": restarts(ms["ca-plantl"], ms["ca-aina"],
                                        best["n"])}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
