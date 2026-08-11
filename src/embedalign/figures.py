"""Figures for the Part 2 write-up.

Palette and styling are copied verbatim from `../wordalign/figures.py` so the
two posts look like one piece of work.  Every figure is rendered twice, once
per site theme, and referenced with mkdocs-material's `#only-light` /
`#only-dark` suffixes.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
        "grid": "#dcdbd6",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

PRETTY = {
    "ca-plantl": "roberta-base-ca", "ca-aina": "roberta-base-ca-v2",
    "da-botxo": "danish-bert-botxo", "da-danskbert": "DanskBERT",
    "en-roberta": "roberta-base", "de-bert": "bert-base-german",
    "es-bert": "beto-spanish", "fr-camembert": "camembert-base",
    "zh-bert": "bert-base-chinese", "ja-bert": "bert-japanese-char",
}


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["text"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["grid"], "grid.color": t["grid"],
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160,
    })


def finish(ax, t, title=None, sub=None):
    if title:
        ax.set_title(title, color=t["text"], loc="left", pad=14 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["muted"],
                fontsize=9.5, va="bottom")
    ax.grid(True, axis="both", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
def fig_scale(scale, t, path):
    """The headline: unsupervised recovery against vocabulary size."""
    style(t)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))

    series = [
        ("ca-plantl|ca-aina", "two Catalan models", t["series"][0], "o"),
        ("da-botxo|da-danskbert", "two Danish models", t["series"][2], "^"),
        ("en-roberta|fr-camembert", "English vs French", t["series"][1], "s"),
    ]
    for key, label, colour, mk in series:
        rows = sorted(scale[key], key=lambda r: r["n"])
        x = [r["n"] for r in rows]
        ax.plot(x, [r["unsupervised"]["1"] * 100 for r in rows], "-",
                marker=mk, color=colour, lw=2, ms=6, label=label, zorder=4,
                markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.plot(x, [r["supervised"]["1"] * 100 for r in rows], ls=(0, (3, 2)),
                marker=mk, color=colour, lw=1.4, ms=4.5, alpha=0.55, zorder=3,
                markeredgecolor=t["surface"], markeredgewidth=1.0)

    ax.set_xscale("log")
    ticks = [1000, 2000, 4000, 8000, 16000, 32000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v // 1000}k" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("shared tokens aligned (frequency-ranked)")
    ax.set_ylabel("token matched to itself at rank 1 (%)")
    ax.set_ylim(-4, 108)
    leg = ax.legend(frameon=False, fontsize=10, ncol=3,
                    loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax.text(0, 1.03, "solid: no supervision at all    dashed: rotation fitted "
                     "on half the answer", transform=ax.transAxes,
            color=t["muted"], fontsize=9.5, va="bottom")
    finish(ax, t)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_methods(main, t, path, key="ca-plantl|ca-aina", note=None):
    """Every method on the same problem."""
    style(t)
    order = [
        ("frequency_rank", "match by frequency rank alone"),
        ("hungarian_direct", "Hungarian on the raw vectors"),
        ("wp_restarts", "Wasserstein–Procrustes"),
        ("gw_match", "Gromov–Wasserstein"),
        ("hungarian_profile", "similarity profile + Hungarian"),
        ("vecmap", "stochastic self-learning"),
        ("procrustes_supervised", "Procrustes with half the answer"),
    ]
    res = main[key]
    labels, vals, colours = [], [], []
    for k, label in order:
        if k not in res:
            continue
        labels.append(label)
        vals.append(res[k]["accuracy"] * 100)
        colours.append(t["series"][1] if "supervised" in k else t["series"][0])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.barh(y, vals, 0.62, color=colours, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%" if v >= 0.05 else "0.0%", va="center", ha="left",
                color=t["text"], fontsize=10)
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("tokens matched to themselves (%)")
    ax.set_xlim(0, min(max(vals) * 1.2 + 1, 112))
    ax.grid(True, axis="x", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    a, b = key.split("|")
    ax.text(0, 1.04, note or
            f"{PRETTY.get(a, a)} against {PRETTY.get(b, b)}, "
            f"{res['_meta']['n_x']:,} tokens, strict one-to-one matching",
            transform=ax.transAxes, color=t["muted"], fontsize=9.5,
            va="bottom")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_survey(survey, t, path, benchmark=0.52, min_n=1400):
    """How alike the pairs are, before any alignment is attempted."""
    style(t)
    rows = sorted([r for r in survey if r["n"] >= min_n],
                  key=lambda r: r["pearson"])
    labels = [f"{PRETTY.get(r['a'], r['a'])} · {PRETTY.get(r['b'], r['b'])}"
              for r in rows]
    vals = [r["pearson"] for r in rows]
    colours = [t["series"][2] if r["same_language"] else t["series"][0]
               for r in rows]

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 0.26 * len(rows) + 1.5))
    ax.barh(y, vals, 0.66, color=colours, zorder=3)
    for yi, v, r in zip(y, vals, rows):
        ax.text(v + 0.008, yi, f"{v:.2f}  (n={r['n']:,})", va="center",
                ha="left", color=t["muted"], fontsize=7.6)
    ax.axvline(benchmark, color=t["series"][1], ls=(0, (4, 3)), lw=1.4,
               zorder=4)
    ax.text(benchmark + 0.006, -1.3, "German–French, Part 1",
            color=t["series"][1], fontsize=8.5, va="bottom", ha="left")
    ax.set_yticks(y, labels, fontsize=7.4)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("relational correlation between the two embedding spaces")
    ax.grid(True, axis="x", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(0, 1.0 + 1.6 / len(rows),
            "green: same language, two independent pretraining runs   "
            f"(pairs sharing at least {min_n:,} tokens)",
            transform=ax.transAxes, color=t["muted"], fontsize=9,
            va="bottom")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_offset(rows, t, path):
    """The data-loading bug, drawn."""
    style(t)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = [4 + r["delta"] for r in rows]
    v = [r["pearson"] for r in rows]
    colours = [t["series"][1] if xi == 4 else t["series"][0] for xi in x]
    bars = ax.bar(x, v, 0.62, color=colours, zorder=3)
    for bar, vi in zip(bars, v):
        ax.text(bar.get_x() + bar.get_width() / 2, vi + 0.012, f"{vi:.3f}",
                ha="center", va="bottom", color=t["text"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xlabel("row offset applied to CamemBERT's SentencePiece ids")
    ax.set_ylabel("relational correlation")
    ax.set_ylim(0, max(v) * 1.22)
    ax.annotate("the documented value", xy=(4.35, max(v) * 0.8),
                xytext=(5.3, max(v) * 0.8), color=t["series"][1], fontsize=9.5,
                va="center",
                arrowprops=dict(arrowstyle="->", color=t["series"][1], lw=1.2))
    finish(ax, t)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_bands(bands, t, path, note=None):
    """Where in the vocabulary the agreement lives."""
    style(t)
    rows = bands["rows"]
    labels = [f"{r['lo'] // 1000}k–{r['hi'] // 1000}k" if r["hi"] >= 1000
              else f"{r['lo']}–{r['hi']}" for r in rows]
    labels = [f"{r['lo']:,}–{r['hi']:,}" for r in rows]
    vals = [r["top1"] * 100 for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.bar(x, vals, 0.6, color=t["series"][0], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", va="bottom", color=t["text"], fontsize=9.5)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlabel("token id band (lower = more frequent)")
    ax.set_ylabel("matched to itself at rank 1 (%)")
    ax.set_ylim(0, 108)
    finish(ax, t, sub=note)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/embedalign-results")
    ap.add_argument("--assets", default="../../docs/assets")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.abspath(os.path.join(here, a.assets))
    os.makedirs(assets, exist_ok=True)

    def load(name):
        p = os.path.join(a.results, name)
        if not os.path.exists(p):
            print(f"  (missing {name}, skipping)")
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    res = load("results.json")
    scale = load("scale.json")

    for mode, t in THEMES.items():
        if scale:
            fig_scale(scale, t, f"{assets}/embedalign-scale-{mode}.png")
        if res:
            if "open" in res:
                fig_methods(res["open"], t,
                            f"{assets}/embedalign-methods-{mode}.png",
                            note="the two Catalan models, top 8,000 ids of "
                                 "each, strict one-to-one matching")
            if "survey" in res:
                fig_survey(res["survey"], t,
                           f"{assets}/embedalign-survey-{mode}.png")
            if "offset_scan" in res:
                fig_offset(res["offset_scan"], t,
                           f"{assets}/embedalign-offset-{mode}.png")
            if "frequency_bands" in res:
                fig_bands(res["frequency_bands"], t,
                          f"{assets}/embedalign-bands-{mode}.png",
                          "one unsupervised alignment of the two Catalan "
                          "models, broken out by token frequency")
    print(f"figures written to {assets}")


if __name__ == "__main__":
    main()
