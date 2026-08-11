"""Two follow-up checks that the main run does not answer.

1.  **Is the shared structure real, or is it punctuation?**  Every tokenizer
    on earth contains the digits, the ASCII punctuation and the Latin
    letters, and those pieces sit in a distinctive, largely universal corner
    of any embedding space.  A cross-script pair whose only shared tokens are
    `,`, `7` and `%` can post a high correlation while sharing nothing about
    language at all.  Split the shared tokens by script and re-measure.

2.  **How much of the result is the postprocessing?**  Transformer embedding
    tables are anisotropic in a way count vectors are not, so the
    unit / centre / unit step could plausibly be doing the work.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "wordalign"))

import align  # noqa: E402
import experiment  # noqa: E402
import isomorphism  # noqa: E402
import models  # noqa: E402


def is_wordy(piece, kind):
    form = models.normalise_piece(piece, kind)
    if form is None:
        return None
    s = form[0]
    return any(ch.isalpha() for ch in s)


def script_split(a, b, limit=10000, max_points=1200, seed=0):
    """Relational correlation over alphabetic vs non-alphabetic tokens."""
    pairs = models.shared_tokens(a, b, limit=limit)
    ma = experiment.postprocess(a.matrix("input"))
    mb = experiment.postprocess(b.matrix("input"))
    groups = {"alphabetic": [], "symbols/digits": []}
    for i, j in pairs:
        w = is_wordy(a.vocab[i], a.kind)
        if w is None:
            continue
        groups["alphabetic" if w else "symbols/digits"].append((i, j))
    out = {}
    for name, ps in groups.items():
        if len(ps) < 60:
            out[name] = {"n": len(ps), "pearson": None}
            continue
        rc = isomorphism.relational_correlation(ma, mb, ps,
                                                max_points=max_points, seed=seed)
        no = isomorphism.neighbourhood_overlap(ma, mb, ps, k=10,
                                               max_points=max_points, seed=seed)
        out[name] = {"n": len(ps), "pearson": rc["pearson"],
                     "neighbourhood_overlap": no["mean_overlap"]}
    return out


def postprocessing_ablation(a, b, n=4000, limit=10000, seed=0):
    """Does the unit / centre / unit step matter, and how much?"""
    rows = []
    for label in ("raw", "unit only", "unit + centre + unit"):
        pair = experiment.build_pair(a, b, limit=limit, matched=True,
                                     n_max=n, post=False, seed=seed)
        x, y = pair.x, pair.y
        if label == "unit only":
            x, y = experiment._unit(x), experiment._unit(y)
        elif label == "unit + centre + unit":
            x, y = experiment.postprocess(x), experiment.postprocess(y)
        p = experiment.Pair(np.ascontiguousarray(x), np.ascontiguousarray(y),
                            pair.gold, pair.meta)
        st = experiment.structure(p)
        res = experiment.run_methods(p, methods=["vecmap",
                                                 "procrustes_supervised"],
                                     verbose=False)
        rows.append({"postprocessing": label, "pearson": st["pearson"],
                     "vecmap": res["vecmap"]["accuracy"],
                     "vecmap_top1": res["vecmap"]["retrieval"]["1"],
                     "supervised": res["procrustes_supervised"]["accuracy"]})
        print(f"  {label:24s} r={rows[-1]['pearson']:.3f}  "
              f"vecmap {rows[-1]['vecmap'] * 100:5.1f}%  "
              f"supervised {rows[-1]['supervised'] * 100:5.1f}%", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/embedalign-results/extras.json")
    args = ap.parse_args()
    ms = {n: models.load(n) for n in models.MODELS}
    out = {}

    print("\n#### is it language, or is it punctuation?\n", flush=True)
    pairs = [("ca-plantl", "ca-aina"), ("da-botxo", "da-danskbert"),
             ("en-roberta", "de-bert"), ("en-roberta", "fr-camembert"),
             ("en-roberta", "zh-bert"), ("de-bert", "zh-bert"),
             ("zh-bert", "ja-bert")]
    rows = []
    print(f"{'pair':30s} {'alpha n':>8} {'alpha r':>8} "
          f"{'sym n':>7} {'sym r':>7}")
    for a, b in pairs:
        s = script_split(ms[a], ms[b])
        al, sy = s["alphabetic"], s["symbols/digits"]
        rows.append({"a": a, "b": b, **{f"alpha_{k}": v for k, v in al.items()},
                     **{f"sym_{k}": v for k, v in sy.items()}})
        fa = f"{al['pearson']:8.3f}" if al["pearson"] is not None else "       -"
        fs = f"{sy['pearson']:7.3f}" if sy["pearson"] is not None else "      -"
        print(f"{a + ' / ' + b:30s} {al['n']:8d} {fa} {sy['n']:7d} {fs}",
              flush=True)
    out["script_split"] = rows

    print("\n#### postprocessing ablation (ca-plantl / ca-aina, n=4000)\n",
          flush=True)
    out["postprocessing"] = postprocessing_ablation(ms["ca-plantl"],
                                                    ms["ca-aina"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
