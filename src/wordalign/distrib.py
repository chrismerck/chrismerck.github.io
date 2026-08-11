"""Distributional vectors, built the way Manning & Schutze describe.

Count the company a word keeps within a +/-w window, weight the counts with
positive pointwise mutual information, then reduce with a truncated SVD.
Nothing neural, nothing autoregressive -- just the co-occurrence structure.
"""

import collections

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def build_vocab(sentences, size, min_count=5):
    counts = collections.Counter()
    for sent in sentences:
        counts.update(w for w, _ in sent)
    items = [(w, c) for w, c in counts.most_common() if c >= min_count]
    words = [w for w, _ in items[:size]]
    return words, {w: i for i, w in enumerate(words)}, counts


def _flatten(sentences, index):
    """Long token-id array plus a parallel sentence-id array."""
    toks, sids = [], []
    for si, sent in enumerate(sentences):
        for w, _ in sent:
            toks.append(index.get(w, -1))
            sids.append(si)
    return np.asarray(toks, dtype=np.int32), np.asarray(sids, dtype=np.int32)


def cooccurrence(sentences, target_index, context_index, window=5):
    """Symmetric windowed co-occurrence counts, shape (|target|, |context|).

    Pairs that would straddle a sentence boundary are dropped, which is why
    we carry the sentence ids around.
    """
    t_tok, sids = _flatten(sentences, target_index)
    c_tok, _ = _flatten(sentences, context_index)

    n_t, n_c = len(target_index), len(context_index)
    total = sp.csr_matrix((n_t, n_c), dtype=np.float64)

    for k in range(1, window + 1):
        same = sids[:-k] == sids[k:]
        # word at i is the target, word at i+k is the context
        a, b = t_tok[:-k], c_tok[k:]
        m = same & (a >= 0) & (b >= 0)
        if m.any():
            total = total + sp.coo_matrix(
                (np.ones(m.sum()), (a[m], b[m])), shape=(n_t, n_c)
            ).tocsr()
        # and the mirror image, so the window is genuinely +/-k
        a, b = t_tok[k:], c_tok[:-k]
        m = same & (a >= 0) & (b >= 0)
        if m.any():
            total = total + sp.coo_matrix(
                (np.ones(m.sum()), (a[m], b[m])), shape=(n_t, n_c)
            ).tocsr()
    return total


def ppmi(counts, shift=0.0, cds=0.75):
    """Positive PMI with context distribution smoothing (Levy & Goldberg)."""
    counts = counts.tocsr().astype(np.float64)
    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    col_sum = np.asarray(counts.sum(axis=0)).ravel()
    col_sum = np.power(col_sum, cds)
    total_r = row_sum.sum()
    total_c = col_sum.sum()

    row_sum[row_sum == 0] = 1.0
    col_sum[col_sum == 0] = 1.0

    out = counts.tocoo()
    # log( p(w,c) / (p(w) p(c)) ) with the two marginals normalised separately
    vals = np.log(
        (out.data / total_r) / ((row_sum[out.row] / total_r) * (col_sum[out.col] / total_c))
    )
    vals -= shift
    np.maximum(vals, 0.0, out=vals)
    keep = vals > 0
    return sp.coo_matrix(
        (vals[keep], (out.row[keep], out.col[keep])), shape=counts.shape
    ).tocsr()


def svd_embed(matrix, dim=300, eig_weight=0.5, seed=0):
    """Truncated SVD of the PPMI matrix -> dense vectors."""
    dim = min(dim, min(matrix.shape) - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(matrix.shape))
    u, s, _ = spl.svds(matrix, k=dim, v0=v0)
    order = np.argsort(-s)
    u, s = u[:, order], s[order]
    return u * (s ** eig_weight)


def unit(x, axis=1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def make_vectors(sentences, vocab_size=5000, context_size=10000, window=5,
                 dim=300, min_count=5, seed=0):
    """Full pipeline: sentences -> (words, unit-norm vectors, counts)."""
    words, index, counts = build_vocab(sentences, vocab_size, min_count)
    ctx_words, ctx_index, _ = build_vocab(sentences, context_size, min_count)
    co = cooccurrence(sentences, index, ctx_index, window=window)
    weighted = ppmi(co)
    vecs = svd_embed(weighted, dim=dim, seed=seed)
    return words, unit(vecs), counts
