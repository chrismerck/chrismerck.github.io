"""Figures for the write-up.

Every figure is rendered twice, once for each site theme, and referenced
from the post with mkdocs-material's `#only-light` / `#only-dark` suffixes.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
def fig_cliff(data, t, path):
    style(t)
    rows = sorted(data["synthetic_axis"], key=lambda r: r["pearson"])
    r = [x["pearson"] for x in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for i, (key, label) in enumerate([
            ("vecmap", "stochastic self-learning"),
            ("profile", "similarity profile + Hungarian")]):
        y = [x[key] * 100 for x in rows]
        ax.plot(r, y, "-o", color=t["series"][i], lw=2, ms=6,
                label=label, zorder=3,
                markeredgecolor=t["surface"], markeredgewidth=1.5)

    marks = [
        ("German–French\n(web vectors)", data["de_fr_embeddings"]["pearson"]),
        ("German–German\n(disjoint corpora)", data["de_de_split_counts"]["pearson"]),
        ("German–French\n(small corpora)", data["de_fr_counts"]["pearson"]),
    ]
    for label, xv in marks:
        ax.axvline(xv, color=t["muted"], ls=(0, (4, 3)), lw=1.2, zorder=2)
        ax.text(xv, 103, label, rotation=0, ha="center", va="bottom",
                color=t["text"], fontsize=8.5, linespacing=1.25)

    ax.set_xlabel("relational correlation between the two spaces")
    ax.set_ylabel("words matched correctly (%)")
    ax.set_ylim(-4, 104)
    ax.set_xlim(-0.03, 1.03)
    leg = ax.legend(frameon=False, loc="center left", fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    finish(ax, t, None)
    fig.subplots_adjust(top=0.78)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_structure(data, t, path):
    style(t)
    conds = [
        ("German – French\nweb-scale vectors", "de_fr_embeddings"),
        ("German – German\ndisjoint corpora, counts", "de_de_split_counts"),
        ("German – French\nsmall corpora, counts", "de_fr_counts"),
    ]
    labels = [c[0] for c in conds]
    rel = [data[c[1]]["pearson"] for c in conds]
    ov = [data[c[1]]["overlap_10"] for c in conds]

    y = np.arange(len(conds))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    b1 = ax.barh(y - h / 2 - 0.012, rel, h, color=t["series"][0],
                 label="relational correlation", zorder=3)
    b2 = ax.barh(y + h / 2 + 0.012, ov, h, color=t["series"][1],
                 label="10-nearest-neighbour overlap", zorder=3)
    for bars, vals in ((b1, rel), (b2, ov)):
        for bar, v in zip(bars, vals):
            ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left",
                    color=t["text"], fontsize=9.5)

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("agreement between the two structures (1.0 = identical shape)")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax.grid(True, axis="x", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_methods(emb, t, path):
    style(t)
    order = [
        ("hungarian-direct", "Hungarian on the raw vectors"),
        ("gromov-wasserstein", "Gromov–Wasserstein"),
        ("hungarian-profile", "similarity profile + Hungarian"),
        ("wasserstein-procrustes", "Wasserstein–Procrustes"),
        ("vecmap-unsupervised", "stochastic self-learning"),
        ("procrustes-supervised", "Procrustes with a dictionary"),
    ]
    res = emb["results"]
    labels, vals, colors = [], [], []
    for key, label in order:
        if key not in res:
            continue
        labels.append(label)
        vals.append(res[key]["accuracy"] * 100)
        colors.append(t["series"][1] if "supervised" in key else t["series"][0])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.barh(y, vals, 0.62, color=colors, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.9, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%" if v >= 0.05 else "0.0%", va="center", ha="left",
                color=t["text"], fontsize=10)
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("words matched correctly (%)")
    ax.set_xlim(0, max(vals) * 1.22 + 1)
    ax.grid(True, axis="x", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(0, 1.04,
            "everything above the last bar runs without a single "
            "translation pair", transform=ax.transAxes,
            color=t["muted"], fontsize=9.5, va="bottom")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_sweep(rows, t, path):
    style(t)
    x = [r["de_tokens"] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, [r["supervised_p1"] * 100 for r in rows], "-o",
            color=t["series"][0], lw=2, ms=6, label="with a seed dictionary",
            markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
    ax.plot(x, [r["unsupervised_p1"] * 100 for r in rows], "-o",
            color=t["series"][1], lw=2, ms=6, label="fully unsupervised",
            markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
    ax.set_xlabel("training tokens per language (millions)")
    ax.set_ylabel("translation retrieved at rank 1 (%)")
    ax.set_ylim(bottom=-1)
    leg = ax.legend(frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    finish(ax, t)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_anisotropy(struct, dist, t, path):
    """Same recovery axis, two different kinds of distortion."""
    style(t)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    iso = sorted(dist["synthetic_axis"], key=lambda r: r["pearson"])
    ax.plot([r["pearson"] for r in iso], [r["vecmap"] * 100 for r in iso],
            "-o", color=t["series"][0], lw=2, ms=6,
            label="isotropic noise", zorder=3,
            markeredgecolor=t["surface"], markeredgewidth=1.5)

    ani = sorted(struct["anisotropic_axis"], key=lambda r: r["pearson"])
    ax.plot([r["pearson"] for r in ani], [r["vecmap"] * 100 for r in ani],
            "-s", color=t["series"][1], lw=2, ms=6,
            label="anisotropic stretch", zorder=3,
            markeredgecolor=t["surface"], markeredgewidth=1.5)

    xv = dist["de_fr_embeddings"]["pearson"]
    ax.axvline(xv, color=t["muted"], ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(xv - 0.015, 50, "German–French", rotation=90, ha="right",
            va="center", color=t["text"], fontsize=9)

    ax.set_xlabel("relational correlation between the two spaces")
    ax.set_ylabel("words matched correctly (%)")
    ax.set_ylim(-4, 104)
    ax.set_xlim(-0.03, 1.03)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    finish(ax, t)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/results")
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

    dist = load("distortion.json")
    fine = load("distortion-fine.json")
    if dist and fine:
        # the fine sweep only refines the calibration curve; merge it in
        merged = {r["noise"]: r for r in dist["synthetic_axis"]}
        merged.update({r["noise"]: r for r in fine["synthetic_axis"]})
        dist["synthetic_axis"] = sorted(merged.values(), key=lambda r: r["noise"])
    emb = load("embeddings-20k.json")
    sweep = load("sweep.json")
    struct = load("structure.json")

    for mode, t in THEMES.items():
        if dist:
            fig_cliff(dist, t, f"{assets}/wordalign-cliff-{mode}.png")
            fig_structure(dist, t, f"{assets}/wordalign-structure-{mode}.png")
        if emb:
            fig_methods(emb, t, f"{assets}/wordalign-methods-{mode}.png")
        if sweep:
            fig_sweep(sweep, t, f"{assets}/wordalign-sweep-{mode}.png")
        if struct and dist:
            fig_anisotropy(struct, dist, t,
                           f"{assets}/wordalign-anisotropy-{mode}.png")
    print(f"figures written to {assets}")


if __name__ == "__main__":
    main()
