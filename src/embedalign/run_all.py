"""The whole Part 2 experiment: survey, alignment, sweeps, diagnostics.

    python3 run_all.py --out /tmp/embedalign-results

Everything it needs it downloads.  On four cores the full run is about an
hour, most of it Gromov-Wasserstein.
"""

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "wordalign"))

import align  # noqa: E402
import experiment  # noqa: E402
import models  # noqa: E402

# the two headline pairs: same language, two independent pretraining runs
SAME_LANGUAGE = [("ca-plantl", "ca-aina"), ("da-botxo", "da-danskbert")]

# cross-language pairs, for continuity with Part 1
CROSS_LANGUAGE = [
    ("en-roberta", "de-bert"),
    ("en-roberta", "fr-camembert"),
    ("de-bert", "fr-camembert"),
    ("es-bert", "ca-aina"),
    ("de-bert", "es-bert"),
    ("fr-camembert", "ca-aina"),
    ("en-roberta", "da-danskbert"),
    ("zh-bert", "ja-bert"),
]


def survey(ms, out):
    """Alignment-free structure for every pair with enough shared tokens."""
    print("\n#### structure survey (no alignment run at all)\n", flush=True)
    rows = []
    print(f"{'pair':34s} {'n':>6} {'pearson':>8} {'spearman':>9} {'nn@10':>7}")
    for a, b in itertools.combinations(ms, 2):
        pair = experiment.build_pair(ms[a], ms[b], limit=10000, matched=True)
        if pair.meta["n_x"] < 400:
            continue
        st = experiment.structure(pair)
        rows.append({"a": a, "b": b, "n": pair.meta["n_x"],
                     "same_language": ms[a].lang == ms[b].lang, **st})
        print(f"{a + ' / ' + b:34s} {pair.meta['n_x']:6d} {st['pearson']:8.3f} "
              f"{st['spearman']:9.3f} {st['neighbourhood_overlap']:7.3f}",
              flush=True)
    out["survey"] = rows
    return rows


def offset_scan(ms, out):
    """Diagnostic: what a wrong fairseq offset does to the structure signal.

    This is the bug that ate the first version of these numbers.  The scan
    uses ground truth, so it is a check on data loading and not a result --
    but a reader deserves to see how sharp the peak is, because the wrong
    answer looks exactly like a clean negative result.
    """
    print("\n#### fairseq offset scan (data-loading diagnostic)\n", flush=True)
    a, b = ms["en-roberta"], ms["fr-camembert"]
    rows = []
    for delta in range(-2, 5):
        shifted = models.Model(
            b.name, ([""] * delta + list(b.vocab))[: b.matrix("input").shape[0]]
            if delta >= 0 else list(b.vocab)[-delta:],
            b.kind, b.mats, b.meta)
        pair = experiment.build_pair(a, shifted, limit=10000, matched=True)
        if pair.meta["n_x"] < 100:
            continue
        st = experiment.structure(pair)
        rows.append({"delta": delta, "n": pair.meta["n_x"],
                     "pearson": st["pearson"]})
        print(f"  applied offset {4 + delta:2d}  n={pair.meta['n_x']:5d}  "
              f"pearson {st['pearson']:6.3f}"
              f"{'   <- documented value' if delta == 0 else ''}", flush=True)
    out["offset_scan"] = rows


def main_table(ms, out, n=2000, pairs=None):
    """Every method, on a problem small enough that every method can run."""
    print(f"\n#### full method comparison (matched, n <= {n})\n", flush=True)
    res = {}
    for a, b in (pairs or SAME_LANGUAGE + CROSS_LANGUAGE):
        pair = experiment.build_pair(ms[a], ms[b], limit=10000, matched=True,
                                     n_max=n)
        if pair.meta["n_x"] < 500:
            print(f"  skipping {a}/{b}: only {pair.meta['n_x']} shared tokens")
            continue
        st = experiment.structure(pair)
        print(f"\n  {a} / {b}   n={pair.meta['n_x']}  "
              f"relational correlation {st['pearson']:.3f}", flush=True)
        r = experiment.run_methods(pair)
        r["_structure"] = st
        r["_meta"] = {k: v for k, v in pair.meta.items() if k != "tokens"}
        res[f"{a}|{b}"] = r
    out["main"] = res
    return res


def vocab_sweep(ms, out, pair_names=SAME_LANGUAGE[0],
                sizes=(500, 1000, 2000, 4000, 8000)):
    """Part 1 found a cliff between 1k and 2k items.  Is it here too?"""
    a, b = pair_names
    print(f"\n#### vocabulary-size sweep, {a} / {b}\n", flush=True)
    rows = []
    for n in sizes:
        pair = experiment.build_pair(ms[a], ms[b], limit=10000, matched=True,
                                     n_max=n)
        if pair.meta["n_x"] < n * 0.6:
            continue
        st = experiment.structure(pair)
        r = experiment.run_methods(
            pair, methods=["hungarian_profile", "vecmap",
                           "procrustes_supervised"], verbose=False)
        row = {"n": pair.meta["n_x"], "pearson": st["pearson"],
               "neighbourhood_overlap": st["neighbourhood_overlap"],
               **{k: v["accuracy"] for k, v in r.items()},
               "vecmap_top1": r["vecmap"]["retrieval"]["1"],
               "vecmap_top10": r["vecmap"]["retrieval"]["10"]}
        rows.append(row)
        print(f"  n={row['n']:5d}  r={row['pearson']:.3f}  "
              f"profile {row['hungarian_profile'] * 100:5.1f}%  "
              f"vecmap {row['vecmap'] * 100:5.1f}%  "
              f"supervised {row['procrustes_supervised'] * 100:5.1f}%",
              flush=True)
    out["vocab_sweep"] = {"pair": f"{a}|{b}", "rows": rows}
    return rows


def frequency_bands(ms, out, pair_names=SAME_LANGUAGE[0], n=8000,
                    bands=((0, 500), (500, 1000), (1000, 2000),
                           (2000, 4000), (4000, 8000))):
    """Does the agreement live in the frequent tokens or the rare ones?

    One alignment is fitted on the whole vocabulary; the accuracy is then
    broken out by where the token sits in the frequency-ordered id space.
    """
    a, b = pair_names
    print(f"\n#### accuracy by frequency band, {a} / {b}\n", flush=True)
    pair = experiment.build_pair(ms[a], ms[b], limit=20000, matched=True,
                                 n_max=n)
    q = align.vecmap_unsupervised(pair.x, pair.y,
                                  cut=min(6000, pair.meta["n_x"]))
    sim = align.cosine_sim(pair.x @ q, pair.y)
    top1 = np.argmax(sim, axis=1)
    correct = top1 == pair.gold
    rows = []
    for lo, hi in bands:
        hi = min(hi, pair.meta["n_x"])
        if lo >= hi:
            continue
        rows.append({"lo": lo, "hi": hi,
                     "top1": float(correct[lo:hi].mean())})
        print(f"  ids {lo:5d}-{hi:5d}   top-1 {rows[-1]['top1'] * 100:5.1f}%",
              flush=True)
    out["frequency_bands"] = {"pair": f"{a}|{b}", "rows": rows}
    return rows


def positional(ms, out):
    """The one place a shared 'vocabulary' really does exist.

    Every one of these models learned an absolute position embedding, and
    position 37 means the same thing in all of them.  512 items, though --
    well under the size threshold Part 1 identified, so this doubles as a
    test of that threshold on completely different data.
    """
    print("\n#### positional embeddings (identity ground truth, n=512)\n",
          flush=True)
    rows = []
    for a, b in [("ca-plantl", "ca-aina"), ("da-botxo", "da-danskbert"),
                 ("en-roberta", "de-bert"), ("de-bert", "es-bert"),
                 ("en-roberta", "ca-aina")]:
        ma, mb = ms[a].mats.get("position"), ms[b].mats.get("position")
        if ma is None or mb is None:
            continue
        # RoBERTa-family tables reserve rows 0 and 1 (padding_idx = 1)
        oa = 2 if ma.shape[0] == 514 else 0
        ob = 2 if mb.shape[0] == 514 else 0
        k = min(ma.shape[0] - oa, mb.shape[0] - ob)
        x = experiment.postprocess(ma[oa:oa + k])
        y_raw = experiment.postprocess(mb[ob:ob + k])
        rng = np.random.default_rng(0)
        perm = rng.permutation(k)
        pair = experiment.Pair(x, y_raw[perm], np.argsort(perm), {})
        st = experiment.structure(pair, max_points=k)
        r = experiment.run_methods(pair, verbose=False, gw_init=3)
        row = {"a": a, "b": b, "n": k, "pearson": st["pearson"],
               **{k2: v["accuracy"] for k2, v in r.items()}}
        rows.append(row)
        print(f"  {a} / {b}: n={k}  r={st['pearson']:.3f}  "
              f"gw {row.get('gw_match', 0) * 100:5.1f}%  "
              f"vecmap {row['vecmap'] * 100:5.1f}%  "
              f"supervised {row['procrustes_supervised'] * 100:5.1f}%",
              flush=True)
    out["positional"] = rows
    return rows


def open_setting(ms, out, pairs=SAME_LANGUAGE, limit=8000):
    """No pre-matching: top-N ids of each model, distractors and all."""
    print(f"\n#### open setting (top {limit} ids of each, unmatched)\n",
          flush=True)
    res = {}
    for a, b in pairs:
        pair = experiment.build_pair(ms[a], ms[b], limit=limit, matched=False,
                                     n_max=limit)
        st = experiment.structure(pair)
        print(f"\n  {a} / {b}  {pair.meta['n_x']}x{pair.meta['n_y']}, "
              f"{pair.meta['n_gold']} scorable, r={st['pearson']:.3f}",
              flush=True)
        r = experiment.run_methods(
            pair, methods=["frequency_rank", "hungarian_direct",
                           "hungarian_profile", "vecmap",
                           "procrustes_supervised"])
        r["_structure"] = st
        r["_meta"] = {k: v for k, v in pair.meta.items() if k != "tokens"}
        res[f"{a}|{b}"] = r
    out["open"] = res
    return res


def examples(ms, out, pair_names=SAME_LANGUAGE[0], n=8000, k=30):
    """A few tokens the unsupervised map got right, and a few it did not.

    Restricted to word-initial alphabetic pieces of three characters or more:
    the head of a byte-BPE vocabulary is punctuation and byte-fallback slots,
    which are matched correctly and tell the reader nothing.
    """
    a, b = pair_names
    pair = experiment.build_pair(ms[a], ms[b], limit=20000, matched=True,
                                 n_max=n)
    q = align.vecmap_unsupervised(pair.x, pair.y, cut=min(6000, n))
    sim = align.cosine_sim(pair.x @ q, pair.y)
    top1 = np.argmax(sim, axis=1)
    toks_a = pair.meta["tokens"]
    inv = np.argsort(pair.gold)
    toks_b = [toks_a[i] for i in inv]  # y row j holds the token toks_a[inv[j]]

    def surface(piece):
        f = models.normalise_piece(piece, ms[a].kind)
        return None if f is None else f

    right, wrong = [], []
    for i in range(len(top1)):
        f = surface(toks_a[i])
        if f is None or not f[1] or len(f[0]) < 3 or not f[0].isalpha():
            continue
        g = surface(toks_b[top1[i]])
        got = g[0] if g else toks_b[top1[i]]
        (right if top1[i] == pair.gold[i] else wrong).append((f[0], got))
    out["examples"] = {"pair": f"{a}|{b}", "right": right[:k],
                       "wrong": wrong[:k], "n_right": len(right),
                       "n_wrong": len(wrong)}
    print(f"\n#### sample matches, {a} / {b}"
          f"  ({len(right)} right / {len(wrong)} wrong among word tokens)\n")
    print("  correct:", " · ".join(x for x, _ in right[:20]))
    print("  wrong:  ", ", ".join(f"{x}->{y}" for x, y in wrong[:12]))
    return out["examples"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/embedalign-results")
    ap.add_argument("--n", type=int, default=2000,
                    help="rows for the all-methods comparison")
    ap.add_argument("--skip", default="", help="comma-separated stage names")
    a = ap.parse_args()
    skip = set(s for s in a.skip.split(",") if s)

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print("#### loading models\n", flush=True)
    ms = {}
    for name in models.MODELS:
        ms[name] = models.load(name, verbose=True)
        m = ms[name]
        print(f"  {name:14s} {m.backbone:42s} V={len(m.vocab):6,d} "
              f"d={m.matrix('input').shape[1]}  "
              f"matrices={sorted(m.mats)}", flush=True)

    out = {"models": {n: {"backbone": m.backbone, "lang": m.lang,
                          "kind": m.kind, "vocab": len(m.vocab),
                          "dim": int(m.matrix("input").shape[1]),
                          "matrices": sorted(m.mats),
                          "package": m.meta["package"],
                          "version": m.meta["version"]}
                      for n, m in ms.items()}}

    stages = [("offset_scan", lambda: offset_scan(ms, out)),
              ("survey", lambda: survey(ms, out)),
              ("main", lambda: main_table(ms, out, n=a.n)),
              ("vocab_sweep", lambda: vocab_sweep(ms, out)),
              ("frequency_bands", lambda: frequency_bands(ms, out)),
              ("positional", lambda: positional(ms, out)),
              ("open", lambda: open_setting(ms, out)),
              ("examples", lambda: examples(ms, out))]
    for name, fn in stages:
        if name in skip:
            continue
        fn()
        with open(os.path.join(a.out, "results.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)

    print(f"\nwrote {a.out}/results.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
