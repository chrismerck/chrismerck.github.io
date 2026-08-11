"""The difficulty ladder, rebuilt for embedding matrices.

Part 1 shipped two confident, clean, entirely wrong negative results before
this ladder caught them.  The rule it enforces: no negative result counts
until every method has been shown to recover an answer that provably exists.

Rungs, each a bit more like the real thing:

  0. the pair identity check -- x against itself, unrotated
  1. a real embedding matrix, randomly rotated and permuted (exact
     isomorphism, real geometry)
  2. the same, with noise at 5% and 15% of the typical coordinate
  3. the same, but with the two spaces given different *effective* ranks,
     which is what a width mismatch between a 15M and a 110M model would
     look like after projection

A method that clears rung 1 at ~100% and then sits at 0% on two real models
is telling you about the models.  A method that fails rung 1 is telling you
about your code.
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

import align  # noqa: E402  (from src/wordalign)
import experiment  # noqa: E402
import models  # noqa: E402


def rotate_permute(x, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    q = align.random_orthogonal(x.shape[1], rng)
    perm = rng.permutation(x.shape[0])
    y = x[perm] @ q
    if noise:
        y = y + noise * rng.standard_normal(y.shape) * np.abs(y).mean()
    y = align.unit_rows(y)
    # gold[i] = row of y holding x's row i
    gold = np.argsort(perm)
    return y, gold


def synthetic_pair(x, noise=0.0, seed=0):
    y, gold = rotate_permute(x, noise=noise, seed=seed)
    return experiment.Pair(x, y, gold, {"synthetic": True})


def real_matrix(name="de-bert", n=2000, which="input"):
    m = models.load(name)
    return experiment.postprocess(m.matrix(which)[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default="ca-aina")
    ap.add_argument("--out", default="/tmp/embedalign-results/sanity.json")
    a = ap.parse_args()

    x = real_matrix(a.model, a.n)
    out = {}

    print(f"\n=== rung 0: {a.model} against itself (n={a.n}) ===", flush=True)
    p = experiment.Pair(x, x, np.arange(x.shape[0]), {})
    print("  structure:", experiment.structure(p))
    out["0. identity"] = experiment.run_methods(
        p, methods=["hungarian_direct", "vecmap"])

    for label, noise in [("1. rotated + permuted", 0.0),
                         ("2. + 5% noise", 0.05),
                         ("2. + 15% noise", 0.15)]:
        print(f"\n=== rung {label} ({a.model}, n={a.n}) ===", flush=True)
        p = synthetic_pair(x, noise=noise)
        st = experiment.structure(p)
        print(f"  relational correlation {st['pearson']:.3f}, "
              f"neighbourhood overlap {st['neighbourhood_overlap']:.3f}")
        res = experiment.run_methods(p)
        res["_structure"] = st
        out[label] = res

    print(f"\n=== rung 3: rank mismatch (768 vs 256) ===", flush=True)
    mc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(mc, full_matrices=False)
    x_low = align.unit_rows(mc @ vt[:256].T @ vt[:256])
    y, gold = rotate_permute(x_low, seed=1)
    p = experiment.Pair(x, y, gold, {})
    st = experiment.structure(p)
    print(f"  relational correlation {st['pearson']:.3f}")
    res = experiment.run_methods(p)
    res["_structure"] = st
    out["3. rank mismatch 768 vs 256"] = res

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
