"""Is it the *amount* of distortion, or its *shape*?

The calibration in `distortion.py` adds isotropic noise, and the
unsupervised methods shrug it off far below the correlation that German and
French actually achieve.  So a scalar "how alike are these spaces" number
cannot be the whole explanation for the failure.

The obvious suspect is that every method here assumes the map between the
two languages is a *rotation*.  If the true German-to-French map stretches
some directions more than others, an orthogonal-only search is looking for
something that isn't there -- and the relational correlation can stay high
while recoverability collapses.

Two tests:

  1. distort German vectors by a deliberately non-orthogonal linear map of
     controlled condition number, and watch correlation and recoverability
     come apart;
  2. on the real pair, fit both an orthogonal map and an unconstrained
     linear map from the gold dictionary.  If the unconstrained map is much
     better, the true relationship is not a rotation.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align
import evaluate
import gold as goldmod
import isomorphism
from distortion import injective_pairs
from run_embeddings import load_topn
from run_experiment import OMW, postprocess


def anisotropic(x, cond, seed=0):
    """Rotate, then stretch the axes so the map is no longer orthogonal."""
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    q = align.random_orthogonal(d, rng)
    # singular values spread geometrically over the requested condition number
    s = np.exp(np.linspace(0, np.log(cond), d))
    rng.shuffle(s)
    q2 = align.random_orthogonal(d, rng)
    a = q @ np.diag(s) @ q2
    perm = rng.permutation(x.shape[0])
    y = align.unit_rows(x[perm] @ a)
    return y, perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--conds", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default="/tmp/results/structure.json")
    a = ap.parse_args()

    report = {}

    print("loading vectors ...", flush=True)
    de_words, de_raw = load_topn("de_core_news_lg", a.vocab)
    fr_words, fr_raw = load_topn("fr_core_news_lg", a.vocab)
    x = postprocess(de_raw)
    y = postprocess(fr_raw)
    gold = goldmod.build_gold(OMW)
    ev = evaluate.evaluable_set(de_words, fr_words, gold)
    pairs = injective_pairs(de_words, fr_words, ev, 4000)
    print(f"  {len(pairs)} gold pairs, {len(ev)} scoreable words")

    # ---- test 2 first: is the real map a rotation at all?
    half = len(pairs) // 2
    train, _ = pairs[:half], pairs[half:]
    xi = np.array([p[0] for p in train])
    yi = np.array([p[1] for p in train])
    d = x.shape[1]

    q_orth = align.orthogonal_procrustes(x[xi], y[yi])
    w_lin, *_ = np.linalg.lstsq(x[xi], y[yi], rcond=None)

    for name, mapping in (("orthogonal", q_orth), ("unconstrained", w_lin)):
        mapped = align.unit_rows(x @ mapping)
        ret = evaluate.retrieval_at_k(
            align.csls_scores(mapped, y), ev, fr_words)
        rel = isomorphism.relational_correlation(mapped, y, pairs)
        report[f"supervised_{name}"] = {
            "p1": ret[1], "p5": ret[5], "p10": ret[10],
            "relational_correlation_after_mapping": rel["pearson"],
        }
        print(f"  supervised {name:14s} P@1={ret[1]*100:5.2f}% "
              f"P@5={ret[5]*100:5.2f}%")

    sv = np.linalg.svd(w_lin, compute_uv=False)
    report["unconstrained_map_spectrum"] = {
        "condition_number": float(sv[0] / sv[-1]),
        "singular_values_head": [float(v) for v in sv[:10]],
        "singular_values_tail": [float(v) for v in sv[-10:]],
    }
    print(f"  best-fit linear map has condition number "
          f"{sv[0]/sv[-1]:.1f}")

    # ---- test 1: correlation vs recoverability under anisotropy
    print("anisotropic calibration ...", flush=True)
    base = x[:a.n]
    rows = []
    for cond in [float(c) for c in a.conds.split(",")]:
        yn, perm = anisotropic(base, cond, seed=1)
        inv = np.argsort(perm)
        truth = [(i, int(inv[i])) for i in range(base.shape[0])]
        rel = isomorphism.relational_correlation(base, yn, truth)
        ov = isomorphism.neighbourhood_overlap(base, yn, truth, k=10)
        q = align.vecmap_unsupervised(base, yn, cut=min(4000, a.n))
        _, c = align.hard_match(align.cosine_sim(base @ q, yn))
        acc = float(np.mean(np.asarray(c) == inv))
        rows.append({"condition_number": cond, "pearson": rel["pearson"],
                     "overlap_10": ov["mean_overlap"], "vecmap": acc})
        print(f"  cond={cond:<5} r={rel['pearson']:.3f} "
              f"overlap@10={ov['mean_overlap']:.3f} "
              f"vecmap={acc*100:6.2f}%", flush=True)
    report["anisotropic_axis"] = rows

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
