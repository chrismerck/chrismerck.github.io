"""Ways of matching one language's word cloud onto another's.

The problem: we have German vectors X (n x d) and French vectors Y (m x e),
built from unrelated corpora.  The coordinate systems are unrelated -- SVD
dimension 7 in German has nothing to do with SVD dimension 7 in French --
so we can only use things that are invariant to how each space happens to
be rotated.

Methods here, roughly in order of how much they respect that:

  frequency_rank        match by frequency order.  no vectors at all.
  hungarian_direct      pretend the coordinates line up.  they don't.
  hungarian_profile     compare *sorted* within-language similarity profiles.
  gromov_wasserstein    compare within-language distance matrices properly.
  wasserstein_procrustes  alternate Hungarian matching and SVD rotation.
  procrustes_supervised   fit the rotation on a seed dictionary (cheating,
                          used as a ceiling).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd as dense_svd


# --------------------------------------------------------------------------
# helpers


def cosine_sim(a, b):
    return a @ b.T


def cosine_dist_matrix(x):
    d = 1.0 - (x @ x.T)
    np.fill_diagonal(d, 0.0)
    np.maximum(d, 0.0, out=d)
    return d


def orthogonal_procrustes(x, y):
    """Q minimising ||x Q - y||_F, restricted to orthogonal Q."""
    u, _, vt = dense_svd(x.T @ y, full_matrices=False)
    return u @ vt


def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def hard_match(score):
    """Hungarian algorithm: one-to-one matching maximising total score."""
    rows, cols = linear_sum_assignment(-score)
    return rows, cols


# --------------------------------------------------------------------------
# the methods


def frequency_rank(n, m):
    """The zero-information baseline: i'th most frequent to i'th most frequent."""
    k = min(n, m)
    return np.arange(k), np.arange(k)


def hungarian_direct(x, y):
    """Chris's original proposal, taken literally.

    Treat SVD dimension i in German as comparable to SVD dimension i in
    French and run the Hungarian algorithm on cosine distance.  Included
    because it is the obvious thing to try and it is instructive to watch
    it fail.
    """
    d = min(x.shape[1], y.shape[1])
    return hard_match(cosine_sim(x[:, :d], y[:, :d]))


def similarity_profile(x, k=200, ref=None):
    """A rotation-invariant fingerprint of each word.

    Take each word's cosine similarity to every other word in its own
    language, sort it, and keep the top k.  Two words that sit in similarly
    shaped neighbourhoods get similar profiles, whatever the coordinates.
    """
    sims = x @ (x if ref is None else ref).T
    if ref is None:
        np.fill_diagonal(sims, -np.inf)
    part = np.sort(sims, axis=1)[:, ::-1][:, :k]
    return part


def hungarian_profile(x, y, k=200):
    px = similarity_profile(x, k)
    py = similarity_profile(y, k)
    # negative euclidean distance as the score
    score = -(
        (px ** 2).sum(1)[:, None] + (py ** 2).sum(1)[None, :] - 2 * px @ py.T
    )
    return hard_match(score)


def _sinkhorn(cost, p, q, eps, iters=100):
    kmat = np.exp(-(cost - cost.min()) / eps)
    u = np.ones_like(p)
    v = np.ones_like(q)
    for _ in range(iters):
        u = p / np.maximum(kmat @ v, 1e-300)
        v = q / np.maximum(kmat.T @ u, 1e-300)
    return u[:, None] * kmat * v[None, :]


def gromov_wasserstein(c1, c2, eps=5e-3, outer=100, inner=50, init=None,
                       tol=1e-9, verbose=False):
    """Entropic Gromov-Wasserstein (Peyre et al. 2016), square loss.

    Finds a soft coupling T between the two point clouds that makes the
    within-language distance matrices agree as well as possible.  Never
    compares a German coordinate to a French one -- only German distances
    to French distances.  This is the formal version of "do the two fuzzy
    structures have the same shape?"
    """
    n, m = c1.shape[0], c2.shape[0]
    p = np.full(n, 1.0 / n)
    q = np.full(m, 1.0 / m)

    # cost-tensor constant term for the square loss
    const = (c1 ** 2) @ np.outer(p, np.ones(m)) + np.outer(np.ones(n), q) @ (c2 ** 2).T

    t = np.outer(p, q) if init is None else init
    prev = np.inf
    for it in range(outer):
        tens = const - c1 @ t @ (2.0 * c2).T
        t = _sinkhorn(tens, p, q, eps, iters=inner)
        obj = float((tens * t).sum())
        if verbose and it % 10 == 0:
            print(f"      gw iter {it:3d}  obj {obj:.6f}")
        if abs(prev - obj) < tol:
            break
        prev = obj
    return t, obj


def _random_coupling(n, m, rng, sink=20):
    """A random doubly-stochastic-ish starting point.

    The uniform coupling p q^T is itself a stationary point of the GW
    iteration, so starting there gets you nowhere at all.  This was the
    single most important bug to fix in this experiment.
    """
    p = np.full(n, 1.0 / n)
    q = np.full(m, 1.0 / m)
    t = rng.random((n, m)) + 1e-3
    for _ in range(sink):
        t *= (p / t.sum(axis=1))[:, None]
        t *= (q / t.sum(axis=0))[None, :]
    return t


def gw_match(x, y, eps=5e-3, outer=100, inner=50, n_init=5, anneal=True,
             seed=0, verbose=False):
    """Gromov-Wasserstein with random restarts and entropy annealing.

    Keeps whichever run reaches the lowest GW cost -- an unsupervised
    criterion, so no dictionary sneaks in.
    """
    c1 = cosine_dist_matrix(x)
    c2 = cosine_dist_matrix(y)
    # put both on the same scale so a single eps works for either
    c1 = c1 / c1.mean()
    c2 = c2 / c2.mean()

    rng = np.random.default_rng(seed)
    n, m = c1.shape[0], c2.shape[0]
    schedule = [eps * 8, eps * 4, eps * 2, eps] if anneal else [eps]

    best = None
    for run in range(n_init):
        t = None if run == 0 else _random_coupling(n, m, rng)
        obj = np.inf
        for e in schedule:
            t, obj = gromov_wasserstein(
                c1, c2, eps=e, outer=max(outer // len(schedule), 10),
                inner=inner, init=t)
        if verbose:
            tag = "uniform" if run == 0 else f"random{run}"
            print(f"    gw init {tag:8s} cost {obj:.6f}")
        if best is None or obj < best[1]:
            best = (t, obj)

    t, obj = best
    rows, cols = hard_match(t)
    return rows, cols, t, obj


def wasserstein_procrustes(x, y, q_init=None, iters=30, seed=0, verbose=False):
    """Alternate between the two things we know how to solve exactly.

    Given a rotation, the best one-to-one matching is a linear assignment
    problem -> Hungarian.  Given a matching, the best rotation is orthogonal
    Procrustes -> SVD.  Iterate.  Same two hammers as the Humpty Dumpty
    puzzle, pointed at a different nail.
    """
    rng = np.random.default_rng(seed)
    d = min(x.shape[1], y.shape[1])
    xs, ys = x[:, :d], y[:, :d]

    q = random_orthogonal(d, rng) if q_init is None else q_init
    prev_cols = None
    for it in range(iters):
        rows, cols = hard_match(cosine_sim(xs @ q, ys))
        q = orthogonal_procrustes(xs[rows], ys[cols])
        obj = float((cosine_sim(xs @ q, ys)[rows, cols]).sum())
        if verbose:
            print(f"      wp iter {it:3d}  obj {obj:.4f}")
        if prev_cols is not None and np.array_equal(prev_cols, cols):
            break
        prev_cols = cols
    return rows, cols, q, obj


def wp_restarts(x, y, n_restarts=5, iters=30, seed=0, gw_init=None,
                verbose=False):
    """Run Wasserstein-Procrustes from several starts, keep the best.

    "Best" is measured by the matching objective itself, not by any
    dictionary -- otherwise we would be smuggling supervision in through
    the back door.
    """
    best = None
    inits = []
    if gw_init is not None:
        inits.append(("gw", gw_init))
    for r in range(n_restarts):
        inits.append((f"rand{r}", None))

    for name, init in inits:
        rows, cols, q, obj = wasserstein_procrustes(
            x, y, q_init=init, iters=iters, seed=seed + hash(name) % 1000,
            verbose=False)
        if verbose:
            print(f"    restart {name:6s} objective {obj:.4f}")
        if best is None or obj > best[3]:
            best = (rows, cols, q, obj, name)
    return best


def self_learning(x, y, q_init, iters=50, k_csls=10, keep=None, seed=0,
                  verbose=False):
    """VecMap-style self-learning (Artetxe et al. 2018).

    Unlike the Hungarian methods this induces a *many-to-one* dictionary by
    nearest neighbour each round, which is a much weaker constraint and
    therefore much easier to escape a bad local optimum with.  Included to
    show what the assignment formulation is up against.
    """
    n, m = x.shape[0], y.shape[0]
    d = min(x.shape[1], y.shape[1])
    xs, ys = x[:, :d], y[:, :d]
    keep = keep or min(n, m)
    q = q_init
    prev = None
    for it in range(iters):
        score = csls_scores(xs @ q, ys, k=k_csls)
        fwd = np.argmax(score, axis=1)
        bwd = np.argmax(score, axis=0)
        # mutual nearest neighbours only -- the usual way to keep the
        # induced dictionary from degenerating
        pairs = [(i, j) for i, j in enumerate(fwd) if bwd[j] == i]
        if len(pairs) < 10:
            break
        xi = np.array([p[0] for p in pairs])
        yi = np.array([p[1] for p in pairs])
        q = orthogonal_procrustes(xs[xi], ys[yi])
        if verbose and it % 10 == 0:
            print(f"      sl iter {it:3d}  {len(pairs)} mutual pairs")
        if prev is not None and len(pairs) == prev:
            break
        prev = len(pairs)
    rows, cols = hard_match(cosine_sim(xs @ q, ys))
    return rows, cols, q


def sorted_profile_init(x, y, cut=4000):
    """Artetxe et al.'s unsupervised initialisation.

    The intuition is exactly the one that motivates this whole experiment:
    a word's *sorted* vector of similarities to its own language is a
    fingerprint of the shape of its neighbourhood, and that shape should
    survive translation even though the coordinates do not.
    """
    xc, yc = x[:cut], y[:cut]
    mx = np.sort(xc @ xc.T, axis=1)
    my = np.sort(yc @ yc.T, axis=1)
    mx = unit_rows(mx - mx.mean(axis=0, keepdims=True))
    my = unit_rows(my - my.mean(axis=0, keepdims=True))
    sim = mx @ my.T
    fwd = np.argmax(sim, axis=1)
    return np.arange(cut), fwd


def unit_rows(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def vecmap_unsupervised(x, y, cut=6000, keep_prob0=0.1, threshold=1e-6,
                        max_iters=400, k_csls=10, seed=0, verbose=False):
    """Robust self-learning with stochastic dictionary induction.

    This is the method that actually cracked unsupervised bilingual lexicon
    induction (Artetxe, Labaka & Agirre, ACL 2018).  The trick that makes
    it work is the *stochasticity*: early on, most of the similarity matrix
    is randomly masked out, so the induced dictionary is deliberately bad
    and the search cannot settle into the nearest bad local optimum.  The
    mask is relaxed only once progress stalls.

    Contrast with the Hungarian methods above, which commit to a globally
    consistent one-to-one matching at every single step and therefore have
    nowhere to wander.
    """
    rng = np.random.default_rng(seed)
    d = min(x.shape[1], y.shape[1])
    xc = np.ascontiguousarray(x[:cut, :d], dtype=np.float32)
    yc = np.ascontiguousarray(y[:cut, :d], dtype=np.float32)
    n, m = xc.shape[0], yc.shape[0]

    src, trg = sorted_profile_init(xc, yc, cut=min(cut, n, m))
    keep_prob = keep_prob0
    best_obj = -np.inf
    q = np.eye(d, dtype=np.float32)

    for it in range(max_iters):
        q = orthogonal_procrustes(xc[src], yc[trg]).astype(np.float32)
        sim = csls_scores(xc @ q, yc, k=k_csls)

        if keep_prob < 1.0:
            mask = rng.random(sim.shape) < keep_prob
            sim = np.where(mask, sim, -np.inf)

        fwd = np.argmax(sim, axis=1)
        bwd = np.argmax(sim, axis=0)
        obj = float(
            np.mean(sim[np.arange(n), fwd][np.isfinite(sim[np.arange(n), fwd])])
        )
        # symmetric dictionary: both directions, as in the paper
        src = np.concatenate([np.arange(n), bwd])
        trg = np.concatenate([fwd, np.arange(m)])

        if obj - best_obj < threshold:
            if keep_prob >= 1.0:
                if verbose:
                    print(f"      vecmap converged at iter {it}")
                break
            keep_prob = min(1.0, keep_prob * 2.0)
            if verbose:
                print(f"      vecmap iter {it:3d}: relaxing mask to "
                      f"keep_prob={keep_prob:.3f}")
        best_obj = max(best_obj, obj)

    q = orthogonal_procrustes(xc[src], yc[trg]).astype(np.float32)
    return q


def procrustes_supervised(x, y, pairs):
    """Ceiling: fit the rotation on known translation pairs."""
    xi = np.array([p[0] for p in pairs])
    yi = np.array([p[1] for p in pairs])
    d = min(x.shape[1], y.shape[1])
    q = orthogonal_procrustes(x[xi, :d], y[yi, :d])
    rows, cols = hard_match(cosine_sim(x[:, :d] @ q, y[:, :d]))
    return rows, cols, q


# --------------------------------------------------------------------------
# retrieval (for comparison with the bilingual-lexicon-induction literature,
# which reports nearest-neighbour accuracy rather than a one-to-one matching)


def csls_scores(x, y, k=10):
    """Cross-domain similarity local scaling (Conneau et al. 2018).

    Uses float32 and a partial selection rather than a full sort; at a
    20k x 20k vocabulary the naive version allocates several gigabytes per
    intermediate and spends most of its time sorting values it discards.
    """
    sim = np.asarray(x, dtype=np.float32) @ np.asarray(y, dtype=np.float32).T
    rx = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)
    ry = np.partition(sim, -k, axis=0)[-k:, :].mean(axis=0)
    sim *= 2.0
    sim -= rx[:, None]
    sim -= ry[None, :]
    return sim
