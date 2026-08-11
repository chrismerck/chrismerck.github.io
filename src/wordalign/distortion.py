"""Put German-vs-French on the same axis as a controlled distortion.

The plan:

  * measure how alike two vector spaces are, with a metric that needs no
    alignment algorithm (isomorphism.relational_correlation);
  * calibrate that metric against synthetic distortion, where we also know
    whether the unsupervised methods can still recover the matching;
  * measure the same metric for real German-vs-French, and for the
    same-language control (German from two disjoint corpora);
  * read off where the real cases land.

If German-vs-French sits below the point where recovery collapses, that is
a quantitative answer to "is the shape similar enough?" rather than a
shrug.
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
import isomorphism
from run_embeddings import load_topn
from run_experiment import UD, OMW, DE_UD, FR_UD, postprocess
from test_sanity import rotate_permute


def injective_pairs(de_words, fr_words, ev, limit=1500):
    """A 1-to-1 subset of the gold dictionary, ordered by German frequency."""
    fr_index = {w: i for i, w in enumerate(fr_words)}
    used = set()
    pairs = []
    for i in sorted(ev):
        cands = sorted(ev[i], key=lambda w: fr_index[w])
        for w in cands:
            j = fr_index[w]
            if j not in used:
                used.add(j)
                pairs.append((i, j))
                break
        if len(pairs) >= limit:
            break
    return pairs


def recoverability(x, y, quick=True):
    """Can the unsupervised methods find the matching? Returns accuracies."""
    out = {}
    q = align.vecmap_unsupervised(x, y, cut=min(4000, x.shape[0]))
    r, c = align.hard_match(align.cosine_sim(x @ q, y))
    out["vecmap"] = (r, c)
    if not quick:
        r, c, _, _ = align.gw_match(x, y, n_init=2, outer=60)
        out["gw"] = (r, c)
    r, c = align.hungarian_profile(x, y, k=min(300, x.shape[0] // 4))
    out["profile"] = (r, c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--noise", default="0.0,0.02,0.05,0.1,0.15,0.25,0.4,0.6")
    ap.add_argument("--out", default="/tmp/results/distortion.json")
    a = ap.parse_args()

    report = {}
    gold = goldmod.build_gold(OMW)

    # ---------------------------------------------------------------- real
    print("loading real vectors ...", flush=True)
    de_words, de_raw = load_topn("de_core_news_lg", a.vocab)
    fr_words, fr_raw = load_topn("fr_core_news_lg", a.vocab)
    x = postprocess(de_raw)
    y = postprocess(fr_raw)
    ev = evaluate.evaluable_set(de_words, fr_words, gold)
    pairs = injective_pairs(de_words, fr_words, ev, a.pairs)
    print(f"  {len(pairs)} one-to-one gold pairs")

    report["de_fr_embeddings"] = {
        **isomorphism.relational_correlation(x, y, pairs),
        **{f"overlap_{k}": isomorphism.neighbourhood_overlap(x, y, pairs, k=k)
           ["mean_overlap"] for k in (1, 5, 10)},
        "source": "spaCy de/fr core_news_lg, top %d" % a.vocab,
    }
    print(f"  DE-FR (spaCy vectors): "
          f"r={report['de_fr_embeddings']['pearson']:.3f} "
          f"overlap@10={report['de_fr_embeddings']['overlap_10']:.3f}")

    # -------------------------------------------------- same-language control
    print("building German count vectors from two disjoint corpora ...",
          flush=True)
    a_sents = corpora.load_conllu(
        [os.path.join(UD, f) for f in
         ["de_hdt-ud-train-a-1.conllu", "de_hdt-ud-train-a-2.conllu"]])
    b_sents = corpora.load_conllu(
        [os.path.join(UD, f) for f in
         ["de_hdt-ud-train-b-1.conllu", "de_hdt-ud-train-b-2.conllu"]])
    wa, va, _ = distrib.make_vectors(a_sents, vocab_size=4000, dim=200)
    wb, vb, _ = distrib.make_vectors(b_sents, vocab_size=4000, dim=200)
    xa, xb = postprocess(va), postprocess(vb)
    bi = {w: i for i, w in enumerate(wb)}
    same = [(i, bi[w]) for i, w in enumerate(wa) if w in bi][:a.pairs]
    report["de_de_split_counts"] = {
        **isomorphism.relational_correlation(xa, xb, same),
        **{f"overlap_{k}": isomorphism.neighbourhood_overlap(xa, xb, same, k=k)
           ["mean_overlap"] for k in (1, 5, 10)},
        "n_shared_vocab": len(same),
        "source": "German HDT halves a vs b, PPMI+SVD",
    }
    print(f"  DE-DE (disjoint corpora, count vectors): "
          f"r={report['de_de_split_counts']['pearson']:.3f} "
          f"overlap@10={report['de_de_split_counts']['overlap_10']:.3f}")

    # ------------------------------------------- cross-lingual count vectors
    print("building DE/FR count vectors ...", flush=True)
    de_s = corpora.load_conllu([os.path.join(UD, f) for f in DE_UD])
    fr_s = corpora.load_conllu([os.path.join(UD, f) for f in FR_UD])
    dw, dv, _ = distrib.make_vectors(de_s, vocab_size=4000, dim=200)
    fw, fv, _ = distrib.make_vectors(fr_s, vocab_size=4000, dim=200)
    dx, fy = postprocess(dv), postprocess(fv)
    ev2 = evaluate.evaluable_set(dw, fw, gold)
    p2 = injective_pairs(dw, fw, ev2, a.pairs)
    report["de_fr_counts"] = {
        **isomorphism.relational_correlation(dx, fy, p2),
        **{f"overlap_{k}": isomorphism.neighbourhood_overlap(dx, fy, p2, k=k)
           ["mean_overlap"] for k in (1, 5, 10)},
        "n_pairs": len(p2),
        "source": "UD treebanks, PPMI+SVD",
    }
    print(f"  DE-FR (count vectors): "
          f"r={report['de_fr_counts']['pearson']:.3f} "
          f"overlap@10={report['de_fr_counts']['overlap_10']:.3f}")

    # ------------------------------------------------------ synthetic axis
    print("calibrating against synthetic distortion ...", flush=True)
    base = x[:3000]
    rows = []
    for noise in [float(s) for s in a.noise.split(",")]:
        t0 = time.time()
        yn, perm = rotate_permute(base, noise=noise, seed=1)
        inv = np.argsort(perm)
        truth = [(i, int(inv[i])) for i in range(base.shape[0])]
        rel = isomorphism.relational_correlation(base, yn, truth)
        ov = isomorphism.neighbourhood_overlap(base, yn, truth, k=10)
        rec = recoverability(base, yn)
        accs = {k: float(np.mean(np.asarray(c) == inv))
                for k, (r, c) in rec.items()}
        rows.append({"noise": noise, "pearson": rel["pearson"],
                     "spearman": rel["spearman"],
                     "overlap_10": ov["mean_overlap"], **accs})
        print(f"  noise={noise:<5} r={rel['pearson']:.3f} "
              f"overlap@10={ov['mean_overlap']:.3f}  "
              f"vecmap={accs['vecmap']*100:6.2f}%  "
              f"profile={accs['profile']*100:6.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
    report["synthetic_axis"] = rows

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
