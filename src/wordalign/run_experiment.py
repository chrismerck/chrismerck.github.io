"""Driver: build the vector spaces, try to align them, score the result.

Usage:  python3 run_experiment.py <condition> [--vocab N] [--out FILE]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align
import corpora
import distrib
import evaluate
import gold as goldmod

DATA = os.environ.get("WORDALIGN_DATA", "/tmp/corp")
UD = os.path.join(DATA, "ud")
EP = os.path.join(DATA, "europarl_raw")
OMW = os.path.join(DATA, "extomw", "extended_omw", "wikt")

DE_UD = [
    "de_hdt-ud-train-a-1.conllu", "de_hdt-ud-train-a-2.conllu",
    "de_hdt-ud-train-b-1.conllu", "de_hdt-ud-train-b-2.conllu",
    "de_gsd-ud-train.conllu",
]
FR_UD = [
    "fr_gsd-ud-train.conllu", "fr_ftb-ud-train.conllu",
    "fr_sequoia-ud-train.conllu", "fr_partut-ud-train.conllu",
    "fr_rhapsodie-ud-train.conllu",
]


def _ud(names, max_tokens=None):
    return corpora.load_conllu([os.path.join(UD, n) for n in names],
                               max_tokens=max_tokens)


def load_condition(name, max_tokens=None):
    """Return (de_sentences, fr_sentences, kind)."""
    if name == "ud-indep":
        # German newswire/web treebanks vs French web/news/literary
        # treebanks.  Different domains, different decades, no shared text.
        return _ud(DE_UD, max_tokens), _ud(FR_UD, max_tokens), "cross"

    if name in ("europarl-comp", "europarl-par"):
        # europarl-comp: German from the first half of the sitting days and
        # French from the second half.  Same register, zero shared content.
        # europarl-par: both languages over all days, i.e. genuinely
        # parallel text.  That is cheating, and it is in here precisely so
        # we can see how much the cheating buys.
        de_half = "a" if name == "europarl-comp" else None
        fr_half = "b" if name == "europarl-comp" else None
        de = corpora.load_europarl(
            corpora.europarl_files(os.path.join(EP, "german"), "de", de_half),
            spacy_path("de_core_news_lg"), max_tokens)
        fr = corpora.load_europarl(
            corpora.europarl_files(os.path.join(EP, "french"), "fr", fr_half),
            spacy_path("fr_core_news_lg"), max_tokens)
        return de, fr, "cross"

    if name == "de-de-split":
        # Control: align German to itself across disjoint corpus halves.
        # Tells us how much of the loss is "cross-lingual" and how much is
        # just "small noisy corpora".
        a = _ud(["de_hdt-ud-train-a-1.conllu", "de_hdt-ud-train-a-2.conllu"],
                max_tokens)
        b = _ud(["de_hdt-ud-train-b-1.conllu", "de_hdt-ud-train-b-2.conllu"],
                max_tokens)
        return a, b, "self"

    raise SystemExit(f"unknown condition {name}")


def spacy_path(name):
    """Resolve a spaCy model name to the unpacked directory under DATA.

    The model tarballs are unpacked rather than pip-installed, so we point
    spaCy straight at the directory.
    """
    import glob as _glob

    hits = _glob.glob(os.path.join(DATA, f"{name}-*", name, f"{name}-*"))
    if not hits:
        raise SystemExit(f"could not find unpacked spaCy model {name} in {DATA}")
    return sorted(hits)[-1]


def spacy_vectors(model, sentences, vocab_size, min_count=5):
    """Take the frequency-ranked vocabulary but use off-the-shelf vectors.

    The point of this condition is to separate two very different failure
    modes: "the idea is wrong" and "my 500k-token corpus makes rubbish
    vectors".  These vectors come from large, independent, monolingual web
    corpora -- one per language, never aligned to each other.
    """
    import spacy

    nlp = spacy.load(spacy_path(model),
                     exclude=["parser", "ner", "tagger", "morphologizer",
                              "attribute_ruler", "lemmatizer"])
    ranked, _, counts = distrib.build_vocab(sentences, 10 ** 9, min_count)
    words, vecs = [], []
    for w in ranked:
        v = nlp.vocab.get_vector(w)
        if v is not None and v.any():
            words.append(w)
            vecs.append(v)
        if len(words) >= vocab_size:
            break
    return words, np.asarray(vecs, dtype=np.float64), counts


def postprocess(vecs):
    """unit-norm -> mean-centre -> unit-norm (standard for this literature)."""
    v = distrib.unit(vecs)
    v = v - v.mean(axis=0, keepdims=True)
    return distrib.unit(v)


def run(condition, vocab_size=2000, dim=200, window=5, context=10000,
        gw_eps=5e-3, gw_outer=100, restarts=5, out_path=None,
        max_tokens=None, vectors="count", gw_init=5):
    t0 = time.time()
    print(f"[{condition}] loading corpora ...", flush=True)
    de_sents, fr_sents, kind = load_condition(condition, max_tokens)
    de_tokens = corpora.token_count(de_sents)
    fr_tokens = corpora.token_count(fr_sents)
    print(f"  German {de_tokens:,} tokens / {len(de_sents):,} sentences")
    print(f"  Other  {fr_tokens:,} tokens / {len(fr_sents):,} sentences")

    de_pos = corpora.pos_map(de_sents)
    fr_pos = corpora.pos_map(fr_sents)

    if vectors == "spacy":
        print("  loading off-the-shelf vectors ...", flush=True)
        de_words, de_vecs, de_counts = spacy_vectors(
            "de_core_news_lg", de_sents, vocab_size)
        fr_model = "de_core_news_lg" if kind == "self" else "fr_core_news_lg"
        fr_words, fr_vecs, fr_counts = spacy_vectors(
            fr_model, fr_sents, vocab_size)
    else:
        print("  building distributional vectors ...", flush=True)
        de_words, de_vecs, de_counts = distrib.make_vectors(
            de_sents, vocab_size=vocab_size, context_size=context,
            window=window, dim=dim)
        fr_words, fr_vecs, fr_counts = distrib.make_vectors(
            fr_sents, vocab_size=vocab_size, context_size=context,
            window=window, dim=dim)
    x = postprocess(de_vecs)
    y = postprocess(fr_vecs)
    print(f"  X {x.shape}  Y {y.shape}   ({time.time()-t0:.0f}s)", flush=True)

    if kind == "self":
        # aligning German with German: the gold answer is the identity
        gold = {w: {w} for w in de_words}
    else:
        gold = goldmod.build_gold(OMW)

    ev = evaluate.evaluable_set(de_words, fr_words, gold)
    print(f"  {len(ev)} of {len(de_words)} source words are scoreable")

    results = {}
    qualitative = {}

    def record(name, rows, cols, extra=None):
        acc, n_scored, hits = evaluate.matching_accuracy(
            rows, cols, de_words, fr_words, ev)
        pos_acc, pos_n = evaluate.pos_agreement(
            rows, cols, de_pos, fr_pos, de_words, fr_words)
        pos_chance = evaluate.pos_agreement_chance(
            rows, cols, de_pos, fr_pos, de_words, fr_words)
        rho = evaluate.frequency_correlation(rows, cols)
        results[name] = {
            "accuracy": acc,
            "n_scored": n_scored,
            "pos_agreement": pos_acc,
            "pos_agreement_chance": pos_chance,
            "pos_n": pos_n,
            "freq_rank_rho": rho,
            **(extra or {}),
        }
        qualitative[name] = {
            "correct_samples": hits[:40],
            "first_pairs": [
                (de_words[r], fr_words[c]) for r, c in list(zip(rows, cols))[:40]
            ],
        }
        print(f"  {name:24s} acc={acc*100:5.2f}%  pos={pos_acc*100:5.1f}%"
              f" (chance {pos_chance*100:4.1f}%)  rho={rho:+.3f}"
              f"  [{n_scored} scored]", flush=True)

    chance = evaluate.random_baseline(de_words, fr_words, ev)
    print(f"  chance accuracy ~ {chance*100:.2f}%")
    results["_chance"] = {"accuracy": chance}
    results["_meta"] = {
        "condition": condition, "vocab": vocab_size, "dim": dim,
        "window": window, "de_tokens": de_tokens, "fr_tokens": fr_tokens,
        "n_evaluable": len(ev),
    }

    n = min(len(de_words), len(fr_words))

    r, c = align.frequency_rank(len(de_words), len(fr_words))
    record("frequency-rank", r, c)

    r, c = align.hungarian_direct(x, y)
    record("hungarian-direct", r, c)

    r, c = align.hungarian_profile(x, y, k=min(200, n // 4))
    record("hungarian-profile", r, c)

    print("  running Gromov-Wasserstein ...", flush=True)
    tg = time.time()
    r, c, coupling, gw_obj = align.gw_match(
        x, y, eps=gw_eps, outer=gw_outer, n_init=gw_init, verbose=True)
    print(f"    ({time.time()-tg:.0f}s, objective {gw_obj:.6f})")
    record("gromov-wasserstein", r, c, {"gw_objective": gw_obj})

    # Procrustes rotation implied by the GW matching, used to warm-start WP
    d = min(x.shape[1], y.shape[1])
    q_gw = align.orthogonal_procrustes(x[r, :d], y[c, :d])

    print("  running Wasserstein-Procrustes ...", flush=True)
    tw = time.time()
    r, c, q, obj, which = align.wp_restarts(
        x, y, n_restarts=restarts, gw_init=q_gw, verbose=True)
    print(f"    ({time.time()-tw:.0f}s, best start = {which})")
    record("wasserstein-procrustes", r, c,
           {"objective": obj, "best_init": which})

    score = align.csls_scores(x[:, :d] @ q, y[:, :d])
    ret = evaluate.retrieval_at_k(score, ev, fr_words)
    results["wasserstein-procrustes"]["retrieval"] = ret
    print(f"    retrieval (CSLS) P@1={ret[1]*100:.2f}% P@5={ret[5]*100:.2f}%")

    print("  running self-learning ...", flush=True)
    ts = time.time()
    r, c, q_sl = align.self_learning(x, y, q_gw, verbose=True)
    print(f"    ({time.time()-ts:.0f}s)")
    record("self-learning", r, c)
    score = align.csls_scores(x[:, :d] @ q_sl, y[:, :d])
    ret = evaluate.retrieval_at_k(score, ev, fr_words)
    results["self-learning"]["retrieval"] = ret
    print(f"    retrieval (CSLS) P@1={ret[1]*100:.2f}% P@5={ret[5]*100:.2f}%")

    # ---- supervised ceiling
    seed_pairs = []
    fr_index = {w: i for i, w in enumerate(fr_words)}
    for i, targets in ev.items():
        j = min((fr_index[t] for t in targets), key=lambda k: k)
        seed_pairs.append((i, j))
    if len(seed_pairs) >= 20:
        half = len(seed_pairs) // 2
        r, c, q_sup = align.procrustes_supervised(x, y, seed_pairs[:half])
        record("procrustes-supervised", r, c,
               {"n_seed": half, "note": "ceiling, uses a dictionary"})
        score = align.csls_scores(x[:, :d] @ q_sup, y[:, :d])
        ret = evaluate.retrieval_at_k(score, ev, fr_words)
        results["procrustes-supervised"]["retrieval"] = ret
        print(f"    retrieval (CSLS) P@1={ret[1]*100:.2f}% P@5={ret[5]*100:.2f}%")

    payload = {"results": results, "qualitative": qualitative,
               "de_words": de_words[:200], "fr_words": fr_words[:200]}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"  wrote {out_path}")
    print(f"[{condition}] done in {time.time()-t0:.0f}s")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--gw-eps", type=float, default=5e-3)
    ap.add_argument("--gw-outer", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--vectors", choices=["count", "spacy"], default="count")
    ap.add_argument("--gw-init", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.condition, vocab_size=a.vocab, dim=a.dim, window=a.window,
        context=a.context, gw_eps=a.gw_eps, gw_outer=a.gw_outer,
        restarts=a.restarts, out_path=a.out, max_tokens=a.max_tokens,
        vectors=a.vectors, gw_init=a.gw_init)
