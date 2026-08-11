"""The variable that turned out to matter: how many words you look at.

At a 2,000-word vocabulary every unsupervised method here fails.  At
20,000 it works.  That is a phase transition, not a gradient, and it is
worth locating properly -- both on off-the-shelf vectors and on the count
vectors built from treebanks, where the vocabulary ceiling is set by how
much text there is.

Only two methods are swept: stochastic self-learning (which succeeds) and
supervised Procrustes (the ceiling).  The Gromov-Wasserstein and
Wasserstein-Procrustes matchings are O(n^3) and cannot be run at these
sizes -- which is itself part of the finding.
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
from run_embeddings import load_topn
from run_experiment import UD, OMW, DE_UD, FR_UD, postprocess


def score_pair(x, y, de_words, fr_words, gold, sl_cut, seed=0):
    ev = evaluate.evaluable_set(de_words, fr_words, gold)
    if len(ev) < 30:
        return None
    d = min(x.shape[1], y.shape[1])

    q_u = align.vecmap_unsupervised(x, y, cut=min(sl_cut, x.shape[0], y.shape[0]),
                                    seed=seed)
    uns = evaluate.retrieval_at_k(
        align.csls_scores(x[:, :d] @ q_u, y[:, :d]), ev, fr_words)

    fr_index = {w: i for i, w in enumerate(fr_words)}
    pairs = [(i, min(fr_index[t] for t in ts)) for i, ts in sorted(ev.items())]
    half = max(len(pairs) // 2, 10)
    q_s = align.orthogonal_procrustes(
        x[[p[0] for p in pairs[:half]], :d],
        y[[p[1] for p in pairs[:half]], :d])
    sup = evaluate.retrieval_at_k(
        align.csls_scores(x[:, :d] @ q_s, y[:, :d]), ev, fr_words)

    return {"n_evaluable": len(ev), "n_seed": half,
            "unsupervised_p1": uns[1], "unsupervised_p5": uns[5],
            "supervised_p1": sup[1], "supervised_p5": sup[5]}


def sweep_embeddings(sizes, out):
    print("=== off-the-shelf vectors, varying vocabulary ===", flush=True)
    gold = goldmod.build_gold(OMW)
    biggest = max(sizes)
    de_words_all, de_raw = load_topn("de_core_news_lg", biggest)
    fr_words_all, fr_raw = load_topn("fr_core_news_lg", biggest)

    rows = []
    for n in sizes:
        t0 = time.time()
        # re-normalise within each vocabulary: mean-centring depends on it
        x = postprocess(de_raw[:n])
        y = postprocess(fr_raw[:n])
        r = score_pair(x, y, de_words_all[:n], fr_words_all[:n], gold,
                       sl_cut=n)
        if r is None:
            continue
        r["vocab"] = n
        rows.append(r)
        print(f"  vocab {n:>6,}  unsupervised P@1={r['unsupervised_p1']*100:6.2f}%"
              f"  supervised P@1={r['supervised_p1']*100:6.2f}%"
              f"  ({r['n_evaluable']} scoreable, {time.time()-t0:.0f}s)",
              flush=True)
        # write after every size: the large vocabularies take a long time and
        # a partial curve is still a usable curve
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print(f"wrote {out}")
    return rows


def sweep_counts(sizes, out, condition="ud-indep"):
    """Same sweep on vectors we build ourselves, where text is the limit."""
    print(f"=== count vectors ({condition}), varying vocabulary ===", flush=True)
    gold = goldmod.build_gold(OMW)

    if condition == "de-de-split":
        a_files = ["de_hdt-ud-train-a-1.conllu", "de_hdt-ud-train-a-2.conllu"]
        b_files = ["de_hdt-ud-train-b-1.conllu", "de_hdt-ud-train-b-2.conllu"]
        de_s = corpora.load_conllu([os.path.join(UD, f) for f in a_files])
        fr_s = corpora.load_conllu([os.path.join(UD, f) for f in b_files])
    else:
        de_s = corpora.load_conllu([os.path.join(UD, f) for f in DE_UD])
        fr_s = corpora.load_conllu([os.path.join(UD, f) for f in FR_UD])

    rows = []
    for n in sizes:
        t0 = time.time()
        dw, dv, _ = distrib.make_vectors(de_s, vocab_size=n, dim=300,
                                         min_count=3)
        fw, fv, _ = distrib.make_vectors(fr_s, vocab_size=n, dim=300,
                                         min_count=3)
        if len(dw) < n * 0.6 or len(fw) < n * 0.6:
            print(f"  vocab {n:>6,}: corpus only supports "
                  f"{len(dw)}/{len(fw)} words, stopping")
            break
        g = {w: {w} for w in dw} if condition == "de-de-split" else gold
        x, y = postprocess(dv), postprocess(fv)
        r = score_pair(x, y, dw, fw, g, sl_cut=len(dw))
        if r is None:
            continue
        r["vocab"] = n
        r["de_vocab_actual"] = len(dw)
        r["fr_vocab_actual"] = len(fw)
        rows.append(r)
        print(f"  vocab {n:>6,}  unsupervised P@1={r['unsupervised_p1']*100:6.2f}%"
              f"  supervised P@1={r['supervised_p1']*100:6.2f}%"
              f"  ({r['n_evaluable']} scoreable, {time.time()-t0:.0f}s)",
              flush=True)
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,2000,4000,6000,10000,15000,20000")
    ap.add_argument("--count-sizes", default="2000,4000,6000,8000,12000")
    ap.add_argument("--results", default="/tmp/results")
    ap.add_argument("--skip-counts", action="store_true")
    a = ap.parse_args()

    sizes = [int(s) for s in a.sizes.split(",")]
    os.makedirs(a.results, exist_ok=True)
    sweep_embeddings(sizes, os.path.join(a.results, "vocab-sweep.json"))
    if not a.skip_counts:
        cs = [int(s) for s in a.count_sizes.split(",")]
        sweep_counts(cs, os.path.join(a.results, "vocab-sweep-dede.json"),
                     "de-de-split")
        sweep_counts(cs, os.path.join(a.results, "vocab-sweep-udindep.json"),
                     "ud-indep")
