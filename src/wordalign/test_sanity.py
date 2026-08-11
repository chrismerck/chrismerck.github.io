"""A difficulty ladder, so a negative result means something.

Each rung makes the problem slightly more like the real thing:

  1. isotropic gaussian cloud, rotated and permuted   (exact isomorphism,
     but no structure to grab hold of)
  2. clustered cloud, rotated and permuted            (exact isomorphism,
     with structure -- much more like a semantic space)
  3. real German vectors, rotated and permuted        (real geometry,
     exact isomorphism)
  4. real German vectors + noise                      (near isomorphism)

If a method clears rung 3 and fails on German-to-French, the failure is
about the two languages, not about the code.
"""

import argparse
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import align


def score(cols, perm):
    """y[j] == x[perm[j]] @ Q, so x-row r belongs with y-row inv[r]."""
    inv = np.argsort(perm)
    return float(np.mean(np.asarray(cols) == inv))


def rotate_permute(x, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    q = align.random_orthogonal(x.shape[1], rng)
    perm = rng.permutation(x.shape[0])
    y = x[perm] @ q
    if noise:
        y = y + noise * rng.standard_normal(y.shape) * np.abs(y).mean()
    y = align.unit_rows(y)
    return y, perm


def isotropic(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return align.unit_rows(rng.standard_normal((n, d)))


def clustered(n, d, k=40, spread=0.35, seed=0):
    """A mixture of gaussians -- a crude stand-in for semantic clusters."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, d))
    which = rng.integers(0, k, size=n)
    pts = centres[which] + spread * rng.standard_normal((n, d))
    return align.unit_rows(pts)


def real_vectors(n, model="de_core_news_lg"):
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_embeddings import load_topn
    from run_experiment import postprocess

    words, raw = load_topn(model, n)
    return postprocess(raw)


def evaluate_rung(name, x, y, perm, gw_vocab=None, verbose=True):
    print(f"\n=== {name}  (n={x.shape[0]}, d={x.shape[1]}) ===", flush=True)
    out = {}

    r, c = align.hungarian_direct(x, y)
    out["hungarian-direct"] = score(c, perm)

    r, c = align.hungarian_profile(x, y, k=min(200, x.shape[0] // 4))
    out["hungarian-profile"] = score(c, perm)

    t0 = time.time()
    r, c, _, obj = align.gw_match(x, y, n_init=2, outer=80)
    out["gromov-wasserstein"] = score(c, perm)
    gw_time = time.time() - t0
    d = min(x.shape[1], y.shape[1])
    q_gw = align.orthogonal_procrustes(x[r, :d], y[c, :d])

    r, c, q, o, which = align.wp_restarts(x, y, n_restarts=4)
    out["wasserstein-procrustes (random init)"] = score(c, perm)

    r, c, q, o, which = align.wp_restarts(x, y, n_restarts=0, gw_init=q_gw)
    out["wasserstein-procrustes (GW init)"] = score(c, perm)

    q = align.vecmap_unsupervised(x, y, cut=min(6000, x.shape[0]))
    r, c = align.hard_match(align.cosine_sim(x @ q, y))
    out["vecmap-unsupervised"] = score(c, perm)

    inv = np.argsort(perm)
    step = max(x.shape[0] // 100, 1)
    pairs = [(i, int(inv[i])) for i in range(0, x.shape[0], step)]
    r, c, _ = align.procrustes_supervised(x, y, pairs)
    out[f"procrustes-supervised ({len(pairs)} seeds)"] = score(c, perm)

    if verbose:
        for k, v in out.items():
            print(f"  {k:40s} {v*100:6.2f}%")
        print(f"  (GW took {gw_time:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--real-n", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/results/sanity.json")
    a = ap.parse_args()

    all_out = {}

    x = isotropic(a.n, 300)
    y, perm = rotate_permute(x)
    all_out["1. isotropic, exact"] = evaluate_rung("1. isotropic, exact", x, y, perm)

    x = clustered(a.n, 300)
    y, perm = rotate_permute(x)
    all_out["2. clustered, exact"] = evaluate_rung("2. clustered, exact", x, y, perm)

    xr = real_vectors(a.real_n)
    y, perm = rotate_permute(xr)
    all_out["3. real German vectors, exact"] = evaluate_rung(
        "3. real German vectors, rotated+permuted", xr, y, perm)

    for noise in (0.05, 0.15):
        y, perm = rotate_permute(xr, noise=noise)
        all_out[f"4. real German vectors, noise={noise}"] = evaluate_rung(
            f"4. real German vectors, noise={noise}", xr, y, perm)

    import json, os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(all_out, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
