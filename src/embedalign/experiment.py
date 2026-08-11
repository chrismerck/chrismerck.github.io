"""Running the Part 1 machinery against transformer embedding matrices.

Nothing in `wordalign.align` knows or cares that its input used to be a PPMI
count matrix, so the methods are imported unchanged.  What changes is the
data and, crucially, the ground truth: with two vocabularies that share
surface forms, the answer is a *known permutation*, not a fuzzy translation
lexicon.  A method either recovers it or it does not.

Two settings:

  matched   keep only the rows that have a counterpart, shuffle one side.
            n x n, ground truth is one permutation, no distractors.  This is
            the transformer analogue of the shared-tokenizer experiment.
  open      keep the top-N ids of each model as they come.  Most rows have
            no counterpart at all; scoring happens on the subset that does.
            Harder, and closer to what you would actually face.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "wordalign"))

import align  # noqa: E402  (from src/wordalign)
import isomorphism  # noqa: E402

import models  # noqa: E402


# ---------------------------------------------------------------- data prep


def postprocess(vecs):
    """unit-norm -> mean-centre -> unit-norm, exactly as in Part 1.

    Transformer embedding tables are famously anisotropic -- a large common
    mean vector plus a couple of rogue coordinates -- so the centring step
    matters more here than it did for count vectors.
    """
    v = _unit(np.asarray(vecs, dtype=np.float64))
    v = v - v.mean(axis=0, keepdims=True)
    return _unit(v)


def _unit(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def reduce_dim(x, y, seed=0):
    """Project both spaces to the smaller dimension, if they differ.

    The orthogonal methods need equal dimension.  Truncating one side's
    coordinates would be arbitrary, so instead each side is rotated into its
    own principal subspace and the smaller rank is kept.  For the models here
    this is a no-op -- everything is 768-dimensional -- but a 15M/110M pair
    would need it, and saying what we would do is part of the specification.
    """
    d = min(x.shape[1], y.shape[1])
    if x.shape[1] == y.shape[1] == d:
        return x, y
    out = []
    for m in (x, y):
        if m.shape[1] == d:
            out.append(m)
            continue
        mc = m - m.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(mc, full_matrices=False)
        out.append(mc @ vt[:d].T)
    return out[0], out[1]


class Pair:
    """Two embedding matrices plus the permutation that relates them."""

    def __init__(self, x, y, gold, meta):
        self.x = x
        self.y = y
        self.gold = gold  # gold[i] = index into y matching row i of x, or -1
        self.meta = meta

    @property
    def scorable(self):
        return np.flatnonzero(self.gold >= 0)


def build_pair(a, b, which="input", limit=10000, matched=True, n_max=None,
               casefold=False, initial_only=False, seed=0, post=True,
               shuffle=False):
    """Assemble the two matrices and the ground-truth permutation.

    Each side keeps *its own* vocabulary's frequency order, exactly as Part 1
    kept the German and French vocabularies in theirs.  This matters more
    than it looks: VecMap's `cut` fits the rotation on the first `cut` rows,
    meaning the most frequent tokens, and that is a legitimate unsupervised
    choice because frequency order is a property of each model on its own.

    The first version of this function shuffled one side to make the ground
    truth a random permutation.  That quietly destroyed the `cut`: at 16,000
    tokens the two working sets then had only a couple of thousand tokens in
    common and self-learning could not form a dictionary at all.  It looked
    like a clean structural failure.  It was a harness bug.  `shuffle=True`
    is kept as a control -- with `cut` raised to the full vocabulary it is
    the stricter experiment, since then no frequency information reaches any
    method at all.
    """
    pairs = models.shared_tokens(a, b, which=which, limit=limit,
                                 casefold=casefold, initial_only=initial_only)
    rng = np.random.default_rng(seed)
    ma, mb = a.matrix(which), b.matrix(which)

    if matched:
        if n_max and len(pairs) > n_max:
            pairs = pairs[:n_max]
        ia = np.array([p[0] for p in pairs])
        ib = np.array([p[1] for p in pairs])
        x = ma[ia]
        if shuffle:
            order = rng.permutation(len(pairs))
        else:
            order = np.argsort(ib, kind="stable")  # B's own frequency order
        y = mb[ib][order]
        gold = np.argsort(order)  # row i of x lives at gold[i] in y
        tokens = [a.vocab[i] for i in ia]
    else:
        na = min(limit or ma.shape[0], ma.shape[0], n_max or 10 ** 9)
        nb = min(limit or mb.shape[0], mb.shape[0], n_max or 10 ** 9)
        x, y = ma[:na], mb[:nb]
        gold = np.full(na, -1, dtype=int)
        for i, j in pairs:
            if i < na and j < nb:
                gold[i] = j
        tokens = [a.vocab[i] for i in range(na)]

    if post:
        x, y = postprocess(x), postprocess(y)
    x, y = reduce_dim(x, y)
    meta = {"a": a.name, "b": b.name, "which": which, "matched": matched,
            "n_x": int(x.shape[0]), "n_y": int(y.shape[0]),
            "n_gold": int((gold >= 0).sum()), "limit": limit,
            "dim": int(x.shape[1]), "tokens": tokens}
    return Pair(np.ascontiguousarray(x), np.ascontiguousarray(y), gold, meta)


# ---------------------------------------------------------------- scoring


def score_matching(rows, cols, gold):
    """Fraction of scorable rows sent to their true partner."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    ok = gold[rows] >= 0
    if not ok.any():
        return 0.0, 0
    return float(np.mean(cols[ok] == gold[rows][ok])), int(ok.sum())


def score_retrieval(sim, gold, ks=(1, 5, 10)):
    """Nearest-neighbour accuracy, no one-to-one constraint."""
    idx = np.flatnonzero(gold >= 0)
    if len(idx) == 0:
        return {k: 0.0 for k in ks}
    top = np.argsort(-sim[idx], axis=1)[:, : max(ks)]
    truth = gold[idx][:, None]
    return {k: float(np.mean((top[:, :k] == truth).any(axis=1))) for k in ks}


# ---------------------------------------------------------------- methods

# Gromov-Wasserstein and the profile Hungarian are O(n^2) in memory and
# O(n^3) per iteration; above a few thousand rows they stop being runnable
# at all, which is itself part of the story.
GW_CAP = 2000
PROFILE_CAP = 12000


def run_methods(pair, methods=None, gw_init=2, wp_restarts=4, seed=0,
                verbose=True, seed_frac=0.5):
    x, y, gold = pair.x, pair.y, pair.gold
    n, m = x.shape[0], y.shape[0]
    out = {}
    all_methods = ["frequency_rank", "hungarian_direct", "hungarian_profile",
                   "gw_match", "wp_restarts", "vecmap", "procrustes_supervised"]
    methods = methods or all_methods

    def record(name, value, secs, extra=None):
        out[name] = {"accuracy": value, "seconds": round(secs, 1)}
        if extra:
            out[name].update(extra)
        if verbose:
            print(f"    {name:24s} {value * 100:6.2f}%   ({secs:.0f}s)",
                  flush=True)

    if "frequency_rank" in methods:
        t0 = time.time()
        r, c = align.frequency_rank(n, m)
        record("frequency_rank", score_matching(r, c, gold)[0], time.time() - t0)

    if "hungarian_direct" in methods:
        t0 = time.time()
        r, c = align.hungarian_direct(x, y)
        record("hungarian_direct", score_matching(r, c, gold)[0],
               time.time() - t0)

    if "hungarian_profile" in methods and n <= PROFILE_CAP:
        t0 = time.time()
        r, c = align.hungarian_profile(x, y, k=min(200, n // 4))
        record("hungarian_profile", score_matching(r, c, gold)[0],
               time.time() - t0)

    q_gw = None
    if "gw_match" in methods and n <= GW_CAP:
        t0 = time.time()
        r, c, _, obj = align.gw_match(x, y, n_init=gw_init, outer=80, seed=seed)
        d = min(x.shape[1], y.shape[1])
        q_gw = align.orthogonal_procrustes(x[r, :d], y[c, :d])
        record("gw_match", score_matching(r, c, gold)[0], time.time() - t0,
               {"gw_cost": obj})

    if "wp_restarts" in methods and n <= GW_CAP * 3:
        t0 = time.time()
        r, c, q, o, which = align.wp_restarts(x, y, n_restarts=wp_restarts,
                                              seed=seed)
        record("wp_restarts", score_matching(r, c, gold)[0], time.time() - t0,
               {"init": which})
        if q_gw is not None:
            t0 = time.time()
            r, c, q, o, which = align.wp_restarts(x, y, n_restarts=0,
                                                  gw_init=q_gw, seed=seed)
            record("wp_gw_init", score_matching(r, c, gold)[0],
                   time.time() - t0)

    if "vecmap" in methods:
        t0 = time.time()
        q = align.vecmap_unsupervised(x, y, cut=min(6000, n, m), seed=seed)
        sim = align.cosine_sim(x @ q, y)
        ret = score_retrieval(sim, gold)
        if n <= 20000:
            r, c = align.hard_match(sim)
            acc = score_matching(r, c, gold)[0]
        else:
            acc = ret[1]
        record("vecmap", acc, time.time() - t0,
               {"retrieval": {str(k): v for k, v in ret.items()}})

    if "procrustes_supervised" in methods:
        t0 = time.time()
        idx = np.flatnonzero(gold >= 0)
        rng = np.random.default_rng(seed)
        sel = rng.permutation(idx)[: max(int(len(idx) * seed_frac), 2)]
        seeds = [(int(i), int(gold[i])) for i in sel]
        r, c, q = align.procrustes_supervised(x, y, seeds)
        sim = align.cosine_sim(x @ q, y)
        held = gold.copy()
        held[sel] = -1  # score on the half we did not fit on
        acc = score_matching(r, c, held)[0]
        record("procrustes_supervised", acc, time.time() - t0,
               {"n_seeds": len(seeds),
                "retrieval": {str(k): v
                              for k, v in score_retrieval(sim, held).items()}})
    return out


# ---------------------------------------------------------------- structure


def structure(pair, k=10, max_points=1500, seed=0):
    """Alignment-free: is there anything here to find?"""
    idx = pair.scorable
    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    pairs = [(int(i), int(pair.gold[i])) for i in idx]
    rc = isomorphism.relational_correlation(pair.x, pair.y, pairs,
                                            max_points=max_points, seed=seed)
    no = isomorphism.neighbourhood_overlap(pair.x, pair.y, pairs, k=k,
                                           max_points=max_points, seed=seed)
    return {"pearson": rc["pearson"], "spearman": rc["spearman"],
            "n_points": rc["n_points"], "neighbourhood_overlap":
            no["mean_overlap"], "k": k}
