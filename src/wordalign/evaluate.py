"""Scoring a proposed German-French matching."""

import numpy as np
from scipy.stats import spearmanr


def evaluable_set(de_words, fr_words, gold):
    """German indices that have at least one gold translation we could hit.

    A word whose only correct answer isn't in the French vocabulary cannot
    be got right by any method, so scoring it would just deflate every
    number equally and hide the differences we care about.
    """
    fr_set = set(fr_words)
    out = {}
    for i, w in enumerate(de_words):
        targets = gold.get(w)
        if not targets:
            continue
        hits = targets & fr_set
        if hits:
            out[i] = hits
    return out


def matching_accuracy(rows, cols, de_words, fr_words, evaluable):
    """Fraction of scoreable matched pairs that are a real translation."""
    n_scored = 0
    n_right = 0
    hits = []
    for r, c in zip(rows, cols):
        if r not in evaluable:
            continue
        n_scored += 1
        if fr_words[c] in evaluable[r]:
            n_right += 1
            hits.append((de_words[r], fr_words[c]))
    return (n_right / n_scored if n_scored else 0.0), n_scored, hits


def random_baseline(de_words, fr_words, evaluable, trials=20, seed=0):
    """Empirical chance level, given that words have several valid answers."""
    rng = np.random.default_rng(seed)
    m = len(fr_words)
    scores = []
    for _ in range(trials):
        perm = rng.permutation(m)
        acc, _, _ = matching_accuracy(
            list(evaluable.keys()),
            [perm[i % m] for i in evaluable.keys()],
            de_words, fr_words, evaluable)
        scores.append(acc)
    return float(np.mean(scores))


def pos_agreement(rows, cols, de_pos, fr_pos, de_words, fr_words):
    """Does the matching at least respect part of speech?

    A method can be useless as a dictionary and still have found real
    structure, if nouns land on nouns and verbs on verbs.
    """
    n, agree = 0, 0
    for r, c in zip(rows, cols):
        p1 = de_pos.get(de_words[r])
        p2 = fr_pos.get(fr_words[c])
        if p1 is None or p2 is None:
            continue
        n += 1
        agree += int(p1 == p2)
    return (agree / n if n else 0.0), n


def pos_agreement_chance(rows, cols, de_pos, fr_pos, de_words, fr_words):
    """Chance POS agreement given the two marginal tag distributions."""
    import collections

    d = collections.Counter()
    f = collections.Counter()
    for r in rows:
        p = de_pos.get(de_words[r])
        if p:
            d[p] += 1
    for c in cols:
        p = fr_pos.get(fr_words[c])
        if p:
            f[p] += 1
    dn, fn = sum(d.values()), sum(f.values())
    if not dn or not fn:
        return 0.0
    return sum((d[t] / dn) * (f[t] / fn) for t in set(d) | set(f))


def frequency_correlation(rows, cols):
    """Rank correlation between matched positions.

    Vocabularies are frequency-ordered, so this asks whether the matching
    sends common words to common words.
    """
    if len(rows) < 3:
        return 0.0
    rho, _ = spearmanr(rows, cols)
    return float(rho)


def retrieval_at_k(score, evaluable, fr_words, ks=(1, 5, 10)):
    """Nearest-neighbour accuracy from a scoring matrix (not one-to-one)."""
    idx = sorted(evaluable.keys())
    if not idx:
        return {k: 0.0 for k in ks}
    top = np.argsort(-score[idx], axis=1)[:, : max(ks)]
    out = {}
    for k in ks:
        right = 0
        for row, i in enumerate(idx):
            cand = {fr_words[j] for j in top[row, :k]}
            if cand & evaluable[i]:
                right += 1
        out[k] = right / len(idx)
    return out
