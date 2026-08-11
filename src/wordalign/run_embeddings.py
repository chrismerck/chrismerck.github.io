"""Large-vocabulary experiment on off-the-shelf monolingual vectors.

`run_experiment.py` builds its own count vectors from small treebanks.
This script instead takes spaCy's German and French vector tables --
trained separately, on large monolingual web corpora, with no bilingual
signal of any kind -- and asks the same question at a realistic vocabulary
size.  Vocabulary is the embedding table's own frequency order, so it does
not depend on the treebanks at all.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align
import corpora
import distrib
import evaluate
import gold as goldmod
from run_experiment import DATA, UD, OMW, DE_UD, FR_UD, spacy_path, postprocess


def load_topn(model, n, min_len=2):
    """Top-n frequency-ranked alphabetic word vectors from a spaCy model."""
    import spacy

    nlp = spacy.load(spacy_path(model),
                     exclude=["parser", "ner", "tagger", "morphologizer",
                              "attribute_ruler", "lemmatizer"])
    vecs = nlp.vocab.vectors
    row2key = {r: k for k, r in vecs.key2row.items()}
    data = np.asarray(vecs.data)

    words, rows, seen = [], [], set()
    for r in range(data.shape[0]):
        key = row2key.get(r)
        if key is None:
            continue
        w = nlp.vocab.strings[key].lower()
        if len(w) < min_len or not w.isalpha() or w in seen:
            continue
        seen.add(w)
        words.append(w)
        rows.append(r)
        if len(words) >= n:
            break
    return words, data[rows].astype(np.float64)


def cached_pos(lang, files, path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    sents = corpora.load_conllu([os.path.join(UD, f) for f in files])
    pm = corpora.pos_map(sents)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pm, fh, ensure_ascii=False)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000,
                    help="vocabulary for retrieval-based evaluation")
    ap.add_argument("--match-vocab", type=int, default=5000,
                    help="vocabulary for the one-to-one Hungarian matching")
    ap.add_argument("--gw-vocab", type=int, default=2000,
                    help="Gromov-Wasserstein is O(n^3); keep it small")
    ap.add_argument("--sl-cut", type=int, default=6000)
    ap.add_argument("--out", default="/tmp/results/embeddings.json")
    a = ap.parse_args()

    t0 = time.time()
    print("loading vectors ...", flush=True)
    de_words, de_raw = load_topn("de_core_news_lg", a.vocab)
    fr_words, fr_raw = load_topn("fr_core_news_lg", a.vocab)
    x = postprocess(de_raw)
    y = postprocess(fr_raw)
    print(f"  X {x.shape}  Y {y.shape}  ({time.time()-t0:.0f}s)")

    gold = goldmod.build_gold(OMW)
    ev = evaluate.evaluable_set(de_words, fr_words, gold)
    print(f"  {len(ev)} of {len(de_words)} source words scoreable")

    de_pos = cached_pos("de", DE_UD, "/tmp/results/de_pos.json")
    fr_pos = cached_pos("fr", FR_UD, "/tmp/results/fr_pos.json")

    results, qualitative = {}, {}
    results["_meta"] = {"vocab": a.vocab, "match_vocab": a.match_vocab,
                        "gw_vocab": a.gw_vocab, "n_evaluable": len(ev),
                        "source": "spaCy de/fr core_news_lg vectors"}

    def record(name, rows, cols, q=None, sub=None):
        words_d = de_words if sub is None else de_words[:sub]
        words_f = fr_words if sub is None else fr_words[:sub]
        ev_sub = ev if sub is None else {i: t for i, t in ev.items() if i < sub}
        acc, n_scored, hits = evaluate.matching_accuracy(
            rows, cols, words_d, words_f, ev_sub)
        pos_acc, _ = evaluate.pos_agreement(
            rows, cols, de_pos, fr_pos, words_d, words_f)
        pos_chance = evaluate.pos_agreement_chance(
            rows, cols, de_pos, fr_pos, words_d, words_f)
        rho = evaluate.frequency_correlation(rows, cols)
        entry = {"accuracy": acc, "n_scored": n_scored,
                 "pos_agreement": pos_acc, "pos_agreement_chance": pos_chance,
                 "freq_rank_rho": rho, "vocab": sub or a.vocab}
        if q is not None:
            d = min(x.shape[1], y.shape[1])
            score = align.csls_scores(x[:, :d] @ q, y[:, :d])
            entry["retrieval"] = evaluate.retrieval_at_k(score, ev, fr_words)
        results[name] = entry
        qualitative[name] = {
            "correct_samples": hits[:60],
            "first_pairs": [(words_d[r], words_f[c])
                            for r, c in list(zip(rows, cols))[:60]],
        }
        line = (f"  {name:26s} acc={acc*100:5.2f}%  pos={pos_acc*100:5.1f}%"
                f" (chance {pos_chance*100:4.1f}%)  rho={rho:+.3f}")
        if "retrieval" in entry:
            r = entry["retrieval"]
            line += f"  P@1={r[1]*100:5.2f}% P@5={r[5]*100:5.2f}%"
        print(line, flush=True)

    nm = a.match_vocab
    xm, ym = x[:nm], y[:nm]

    # ---- 1. the naive proposal
    r, c = align.hungarian_direct(xm, ym)
    record("hungarian-direct", r, c, sub=nm)

    # ---- 2. sorted similarity profiles + Hungarian
    r, c = align.hungarian_profile(xm, ym, k=500)
    record("hungarian-profile", r, c, sub=nm)

    # ---- 3. Gromov-Wasserstein on the small end
    ng = a.gw_vocab
    print("  Gromov-Wasserstein ...", flush=True)
    tg = time.time()
    r, c, _, obj = align.gw_match(x[:ng], y[:ng], eps=5e-3, outer=80,
                                  n_init=3, verbose=True)
    print(f"    ({time.time()-tg:.0f}s cost {obj:.6f})")
    record("gromov-wasserstein", r, c, sub=ng)
    d = min(x.shape[1], y.shape[1])
    q_gw = align.orthogonal_procrustes(x[:ng][r, :d], y[:ng][c, :d])
    results["gromov-wasserstein"]["gw_cost"] = obj

    # ---- 4. Wasserstein-Procrustes (Hungarian + SVD, alternating)
    print("  Wasserstein-Procrustes ...", flush=True)
    tw = time.time()
    r, c, q_wp, o, which = align.wp_restarts(
        xm, ym, n_restarts=4, gw_init=q_gw, verbose=True)
    print(f"    ({time.time()-tw:.0f}s best={which})")
    record("wasserstein-procrustes", r, c, q=q_wp, sub=nm)

    # ---- 5. stochastic self-learning
    print("  VecMap-style stochastic self-learning ...", flush=True)
    tv = time.time()
    q_vm = align.vecmap_unsupervised(x, y, cut=a.sl_cut, verbose=True)
    print(f"    ({time.time()-tv:.0f}s)")
    r, c = align.hard_match(align.cosine_sim(xm[:, :d] @ q_vm, ym[:, :d]))
    record("vecmap-unsupervised", r, c, q=q_vm, sub=nm)

    # ---- 6. supervised ceiling
    fr_index = {w: i for i, w in enumerate(fr_words)}
    pairs = [(i, min(fr_index[t] for t in ts)) for i, ts in sorted(ev.items())]
    half = len(pairs) // 2
    r, c, q_sup = align.procrustes_supervised(xm, ym, [
        p for p in pairs[:half] if p[0] < nm and p[1] < nm])
    record("procrustes-supervised", r, c, q=q_sup, sub=nm)
    results["procrustes-supervised"]["n_seed"] = half

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "qualitative": qualitative}, fh,
                  ensure_ascii=False, indent=1)
    print(f"wrote {a.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
