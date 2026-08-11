"""How similar are the two shapes, really?

Chris's intuition was that the two languages build "fuzzy crystalline
structures" that might have the same form.  That is a testable claim and it
does not need any alignment algorithm to test: take a set of known
translation pairs, and ask whether the German similarity between a pair of
German words predicts the French similarity between their translations.

  relational correlation = corr( sim_de(i, j),  sim_fr(t(i), t(j)) )

1.0 means the two structures are the same shape.  0.0 means knowing how two
German words relate tells you nothing about how their French translations
relate.  Everything interesting happens in between.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def relational_correlation(x, y, pairs, max_points=1500, seed=0):
    """Correlation between within-language similarities over matched pairs."""
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    if len(pairs) > max_points:
        idx = rng.choice(len(pairs), max_points, replace=False)
        pairs = [pairs[i] for i in idx]
    xi = np.array([p[0] for p in pairs])
    yi = np.array([p[1] for p in pairs])

    sx = x[xi] @ x[xi].T
    sy = y[yi] @ y[yi].T
    iu = np.triu_indices(len(pairs), k=1)
    a, b = sx[iu], sy[iu]
    return {
        "pearson": float(pearsonr(a, b)[0]),
        "spearman": float(spearmanr(a, b)[0]),
        "n_points": len(pairs),
    }


def neighbourhood_overlap(x, y, pairs, k=10, max_points=1500, seed=0):
    """Fraction of a word's k nearest neighbours that survive translation.

    A more intuitive companion to the correlation: if `Hund`'s ten nearest
    German neighbours translate to ten words that are among `chien`'s ten
    nearest French neighbours, the local structure is preserved.
    """
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    if len(pairs) > max_points:
        idx = rng.choice(len(pairs), max_points, replace=False)
        pairs = [pairs[i] for i in idx]
    xi = np.array([p[0] for p in pairs])
    yi = np.array([p[1] for p in pairs])

    sx = x[xi] @ x[xi].T
    sy = y[yi] @ y[yi].T
    np.fill_diagonal(sx, -np.inf)
    np.fill_diagonal(sy, -np.inf)
    nx = np.argsort(-sx, axis=1)[:, :k]
    ny = np.argsort(-sy, axis=1)[:, :k]

    overlaps = [len(set(nx[i]) & set(ny[i])) / k for i in range(len(pairs))]
    return {"mean_overlap": float(np.mean(overlaps)), "k": k,
            "n_points": len(pairs)}


def eigenvalue_profile(x, k=100):
    """Spectrum of the within-language Gram matrix, normalised.

    Two spaces with genuinely the same shape should have similar spectra --
    a cheap, alignment-free structural fingerprint.
    """
    g = x @ x.T
    vals = np.linalg.eigvalsh(g)[::-1][:k]
    return vals / vals.sum()
