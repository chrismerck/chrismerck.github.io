"""
ppo_visual.py
================
Visualise PPO training history as an animation with three panels:

    ┌────────────────────────────────────────────┐
    │                    (1)                    │
    │            Loss vs. training step         │
    ├──────────────┬────────────────────────────┤
    │     (2)      │             (3)            │
    │ Value grid   │   Policy arrows grid       │
    └──────────────┴────────────────────────────┘

Panel‑2 shows a coloured value‑function heat‑map; Panel‑3 shows the
policy as arrows drawn **on top of a faint grey grid** so the user can
see which cell the arrow belongs to.  Each cell shows *one* arrow – the
action with the highest probability.  Arrow length is proportional to
that probability and is scaled so the arrow head reaches the edge of
the cell.

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.patches import FancyArrowPatch, Arrow

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ARROW_DIRS = np.array(
    [
        (-1, 0),  # Up
        (0, 1),   # Right
        (1, 0),   # Down
        (0, -1),  # Left
    ]
)
ARROW_SCALE = 0.45   # scale so arrow heads land roughly on cell borders
ARROW_WIDTH = 0.015  # shaft width


# --------------------------------------------------------------------------- #
# Helpers for history validation / cleaning
# --------------------------------------------------------------------------- #

def _infer_mask(values: np.ndarray, mask: np.ndarray | None):
    """Return boolean array where *True* means obstacle/wall cell."""

    if mask is not None:
        return mask.astype(bool)
    return np.isnan(values)


def _prepare_history(history_raw):
    """Convert list of dicts to a clean structure and sanity‑check shapes."""

    processed: list[dict] = []
    for i, ep in enumerate(history_raw):
        try:
            v = np.asarray(ep["values"], dtype=float)
            p = np.asarray(ep["policy"], dtype=float)
            loss = float(ep["loss"])
        except KeyError as e:
            raise KeyError(f"Epoch {i} is missing key {e}") from None

        if p.shape != v.shape + (4,):
            raise ValueError(
                f"Epoch {i}: policy shape {p.shape} incompatible with values shape {v.shape}"
            )

        mask = ep.get("mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != v.shape:
                raise ValueError("mask shape must equal values shape")

        processed.append(
            {
                "values": v,
                "policy": p,
                "loss": loss,
                "mask": _infer_mask(v, mask),
            }
        )
    return processed


# --------------------------------------------------------------------------- #
# Plot‑initialisation helpers
# --------------------------------------------------------------------------- #

def _init_value_ax(ax: plt.Axes, grid_shape: tuple[int, int]):
    rows, cols = grid_shape

    im = ax.imshow(
        np.zeros(grid_shape),
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        interpolation="none",
    )

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Value network")

    # Draw black gridlines (minor) to match screenshot vibe
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    # One text annotation per cell, updated each frame
    texts = [
        [
            ax.text(
                c,
                r,
                "",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return im, texts


def _init_policy_ax(ax: plt.Axes, grid_shape: tuple[int, int]):
    """Return a configured ax plus a list to store arrow patches."""

    rows, cols = grid_shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # invert y so [0,0] at top‑left
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.set_title("Policy network")

    # Light‑grey cell borders for spatial reference
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.5)

    # Store arrows in a list
    arrows = []
    
    return arrows


# --------------------------------------------------------------------------- #
# Panel‑update helpers
# --------------------------------------------------------------------------- #

def _update_value_panel(data: dict, im: plt.AxesImage, texts, cmap_norm):
    v = data["values"].copy()
    mask = data["mask"]

    # Masked cells → black
    display = np.ma.masked_where(mask, v)
    im.set_data(display)
    im.set_norm(cmap_norm)
    im.cmap.set_bad(color="black")

    rows, cols = v.shape
    for r in range(rows):
        for c in range(cols):
            txt = texts[r][c]
            if mask[r, c]:
                txt.set_text("")
                continue
            value = v[r, c]
            # Show 0.0 for small values to reduce visual noise
            if abs(value) < 0.1:
                display_value = 0.0
            else:
                display_value = value
            # Always show sign and one decimal place
            txt.set_text(f"{display_value:+.1f}")
            txt.set_color("black" if abs(value) < cmap_norm.vmax / 2 else "white")


def _update_policy_panel(data: dict, arrows, ax):
    p = data["policy"]
    mask = data["mask"]
    rows, cols, _ = p.shape

    # Clear all previous arrows by removing patches
    # Get all patches that are arrows and remove them
    patches_to_remove = [patch for patch in ax.patches if hasattr(patch, 'get_arrowstyle') or isinstance(patch, Arrow)]
    for patch in patches_to_remove:
        patch.remove()

    # Minimum probability threshold to show an arrow
    min_prob = 0.02
    
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                continue

            probs = p[r, c]
            
            # Draw arrows for all four directions
            for action in range(4):
                prob = probs[action]
                
                if prob < min_prob:
                    continue
                    
                # Get direction
                dy, dx = ARROW_DIRS[action]
                
                # Calculate arrow positions
                # Start from center
                start_x = c
                start_y = r
                
                # End position based on direction and probability
                # Length proportional to probability
                arrow_length = 0.45 * prob
                end_x = c + dx * arrow_length
                end_y = r + dy * arrow_length
                
                # Width also proportional to probability
                arrow_width = 0.05 + 0.3 * prob
                
                # Color intensity based on probability
                color_intensity = 0.3 + 0.7 * prob
                arrow_color = (0, 0.5 * color_intensity, color_intensity)
                
                # Draw arrow
                arrow = Arrow(
                    start_x, start_y,
                    end_x - start_x, end_y - start_y,
                    width=arrow_width,
                    color=arrow_color,
                    alpha=0.8
                )
                ax.add_patch(arrow)


# --------------------------------------------------------------------------- #
# Animation driver
# --------------------------------------------------------------------------- #

def animate(history: list[dict], save_path: str | Path | None = None, fps: int = 10):
    grid_shape = history[0]["values"].shape
    losses = [ep["loss"] for ep in history]

    # Consistent colour scale across epochs
    vmax = max(np.nanmax(np.abs(ep["values"])) for ep in history)
    cmap_norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Figure and axes layout
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    ax_loss = fig.add_subplot(gs[0, :])
    ax_value = fig.add_subplot(gs[1, 0])
    ax_policy = fig.add_subplot(gs[1, 1])

    # ---- Loss curve ---- #
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.plot(np.arange(len(losses)), losses, color="black", linewidth=1.5)
    loss_point, = ax_loss.plot([], [], "ro")  # marker showing current epoch

    # ---- Value grid ---- #
    im, texts = _init_value_ax(ax_value, grid_shape)

    # ---- Policy grid ---- #
    arrows = _init_policy_ax(ax_policy, grid_shape)

    fig.tight_layout()

    # Animation update function
    def _update(frame: int):
        data = history[frame]
        _update_value_panel(data, im, texts, cmap_norm)
        _update_policy_panel(data, arrows, ax_policy)
        loss_point.set_data([frame], [data["loss"]])  # needs sequences, not scalars
        return im, loss_point, *sum(texts, [])  # flatten text list

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(history),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    if save_path:
        print(f"Saving animation to {save_path} …")
        anim.save(save_path, writer="ffmpeg", fps=fps)
    else:
        plt.show()

    return anim


# --------------------------------------------------------------------------- #
# CLI entry‑point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PPO training history (loss + value + policy)."
    )
    parser.add_argument("history_file", type=Path, help="Pickle file produced during training.")
    parser.add_argument("--save", metavar="out.mp4", default=None, help="Save animation instead of showing interactively.")
    parser.add_argument("--fps", type=int, default=10, help="Frames/sec for saved animation (default 10).")
    args = parser.parse_args()

    with open(args.history_file, "rb") as f:
        raw_history = pickle.load(f)
    history = _prepare_history(raw_history)

    animate(history, save_path=args.save, fps=args.fps)


if __name__ == "__main__":
    main()
