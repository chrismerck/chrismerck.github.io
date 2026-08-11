"""How much text do you need before the two shapes match?

The interesting quantity is not "did unsupervised alignment work" but
"*how isomorphic are these two spaces at all*".  A supervised orthogonal
Procrustes fit answers that: if a single rotation, fitted on known pairs,
maps German onto French well, then the two distributional geometries
really do have the same shape and any failure is a search failure.

This sweeps corpus size and reports that ceiling alongside what the
unsupervised methods manage.
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
from run_experiment import UD, OMW, DE_UD, FR_UD, postprocess


def build(sents, vocab_size, dim, window, context):
    words, vecs, _ = distrib.make_vectors(
        sents, vocab_size=vocab_size, context_size=context, window=window,
        dim=dim, min_count=3)
    return words, postprocess(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="50000,100000,250000,500000,900000")
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--out", default="/tmp/results/sweep.json")
    a = ap.parse_args()

    sizes = [int(s) for s in a.sizes.split(",")]
    gold = goldmod.build_gold(OMW)
    rows = []

    for size in sizes:
        t0 = time.time()
        de = corpora.load_conllu([os.path.join(UD, f) for f in DE_UD],
                                 max_tokens=size)
        fr = corpora.load_conllu([os.path.join(UD, f) for f in FR_UD],
                                 max_tokens=size)
        de_n, fr_n = corpora.token_count(de), corpora.token_count(fr)
        de_words, x = build(de, a.vocab, a.dim, a.window, a.context)
        fr_words, y = build(fr, a.vocab, a.dim, a.window, a.context)
        ev = evaluate.evaluable_set(de_words, fr_words, gold)
        if len(ev) < 30:
            print(f"size {size}: too few scoreable words ({len(ev)}), skipping")
            continue

        fr_index = {w: i for i, w in enumerate(fr_words)}
        pairs = [(i, min(fr_index[t] for t in ts)) for i, ts in sorted(ev.items())]
        half = max(len(pairs) // 2, 10)
        d = min(x.shape[1], y.shape[1])

        # supervised ceiling: fit on half the dictionary, score on all
        q = align.orthogonal_procrustes(
            x[[p[0] for p in pairs[:half]], :d],
            y[[p[1] for p in pairs[:half]], :d])
        sup = evaluate.retrieval_at_k(
            align.csls_scores(x[:, :d] @ q, y[:, :d]), ev, fr_words)

        # unsupervised, same vectors
        q_u = align.vecmap_unsupervised(x, y, cut=min(a.vocab, len(de_words)))
        uns = evaluate.retrieval_at_k(
            align.csls_scores(x[:, :d] @ q_u, y[:, :d]), ev, fr_words)

        row = {"target_tokens": size, "de_tokens": de_n, "fr_tokens": fr_n,
               "n_evaluable": len(ev), "n_seed": half,
               "supervised_p1": sup[1], "supervised_p5": sup[5],
               "unsupervised_p1": uns[1], "unsupervised_p5": uns[5]}
        rows.append(row)
        print(f"  {de_n:>9,} de / {fr_n:>9,} fr tokens  "
              f"supervised P@1={sup[1]*100:5.2f}% P@5={sup[5]*100:5.2f}%  "
              f"unsupervised P@1={uns[1]*100:5.2f}%  "
              f"({len(ev)} scoreable, {time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
