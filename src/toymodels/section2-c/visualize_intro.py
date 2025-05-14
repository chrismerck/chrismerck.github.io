#!/usr/bin/env python3
"""
visualise_training_log.py

• Left panel: animated loss-vs-run curve (updates per frame)
• Center: 2D trace of each column of W, color-coded, animated
• Right: current loss bar + numeric value
• Title updates with "run X / N – loss = …"

Usage:
    python visualise_training_log.py [log_file] [--save gif|mp4] [--interval ms]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec


# ────────────────────────  log-file parsing  ────────────────────────────────
def _read_block(lines, i: int):
    rows = []
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(']'):
            line = line[:-1].strip()
            if line:
                rows.append([float(x) for x in line.split()])
            i += 1
            break
        rows.append([float(x) for x in line.split()])
        i += 1
    return np.asarray(rows, float), i


def parse_log(path: str | Path = "log"):
    runs, cur = [], {}
    with Path(path).open() as fp:
        raw = fp.readlines()

    i = 0
    while i < len(raw):
        tok = raw[i].strip()
        if tok.startswith("run:"):
            if cur:
                runs.append(cur)
                cur = {}
            cur["run"] = int(tok.split(":")[1])
            i += 1
        elif tok.startswith("L:"):
            cur["loss"] = float(tok.split(":")[1])
            i += 1
        elif tok.startswith("W^T W:"):
            i += 1
            cur["WTW"], i = _read_block(raw, i)
        elif tok.startswith("b:"):
            i += 1
            vec, i = _read_block(raw, i)
            cur["b"] = vec.flatten()
        elif tok.startswith("W:"):
            i += 1
            cur["W"], i = _read_block(raw, i)
        else:
            i += 1
    if cur:
        runs.append(cur)
    return runs


# ────────────────────────  figure & animation  ───────────────────────────────
def build_figure_and_animation(runs, interval_ms=600):
    runs_W = [r for r in runs if "W" in r]
    if not runs_W:
        raise RuntimeError("No W matrices found in the log – nothing to animate.")

    # core arrays
    Ws = np.stack([r["W"][:2] for r in runs_W])   # (frames, 2, n_feat)
    n_frames, _, n_feat = Ws.shape
    run_numbers = [r["run"] for r in runs_W]

    # loss array, map from run to value
    loss_map = {r["run"]: r["loss"] for r in runs if "loss" in r}
    loss_values = np.array([loss_map.get(rn, np.nan) for rn in run_numbers])

    # loss for full curve
    all_loss_runs = [r["run"] for r in runs if "loss" in r]
    all_loss_vals = [r["loss"] for r in runs if "loss" in r]

    # Determine plot limits for ax_loss based on ALL losses
    if all_loss_runs:
        plot_xlim_loss = (min(all_loss_runs), max(all_loss_runs))
    else:
        plot_xlim_loss = (0, 1) # Default if no loss runs

    if all_loss_vals:
        _max_val = np.nanmax(all_loss_vals)
        if np.isfinite(_max_val): # Check if _max_val is a finite number
            plot_ylim_loss_top = _max_val * 1.05
            if plot_ylim_loss_top == 0: # If max loss was 0, make ylim slightly positive
                plot_ylim_loss_top = 1.0
        else: # _max_val is nan or inf
            plot_ylim_loss_top = 1.0
    else: # all_loss_vals is empty
        plot_ylim_loss_top = 1.0

    # colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_feat)]

    # layout
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax_loss = fig.add_subplot(gs[0])
    ax_w    = fig.add_subplot(gs[1])

    # 1. loss curve setup
    ax_loss.set_title("Loss during training")
    ax_loss.set_xlabel("run")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True)
    ax_loss.set_xlim(plot_xlim_loss[0], plot_xlim_loss[1])
    ax_loss.set_ylim(0, plot_ylim_loss_top)
    if all_loss_runs: # Plot all loss points statically
        ax_loss.plot(all_loss_runs, all_loss_vals, marker='.', markersize=2, linestyle='none', color="dimgray")

        # Calculate and plot static moving average for all_loss_vals
        if len(all_loss_vals) >= 3: # Need at least a few points for a meaningful MA
            window_size = max(1, min(len(all_loss_vals) // 10, 500))
            if window_size > 0 and len(all_loss_vals) >= window_size:
                moving_avg = np.convolve(all_loss_vals, np.ones(window_size)/window_size, mode='valid')
                # Adjust x-coordinates for the moving average to align center of window or start
                # For mode='valid', the result is shorter. We align it with the end of the first window.
                moving_avg_runs = all_loss_runs[window_size-1:]
                if len(moving_avg_runs) == len(moving_avg):
                    ax_loss.plot(moving_avg_runs, moving_avg, color='black', lw=1.2, linestyle='-', label=f'MA ({window_size})')
                    ax_loss.legend(fontsize='small')

    # Add a vertical line for current run
    current_run_vline = ax_loss.axvline(x=run_numbers[0] if run_numbers else plot_xlim_loss[0], color='red', linestyle='--', lw=1)

    # 2. 2D embedding plot
    xmin, xmax = Ws[:, 0, :].min(), Ws[:, 0, :].max()
    ymin, ymax = Ws[:, 1, :].min(), Ws[:, 1, :].max()
    margin = 0.1 * max(xmax - xmin, ymax - ymin)
    ax_w.set_xlim(xmin - margin, xmax + margin)
    ax_w.set_ylim(ymin - margin, ymax + margin)
    ax_w.axhline(0, c='k', lw=.5)
    ax_w.axvline(0, c='k', lw=.5)
    ax_w.set_xlabel("hidden dim 0")
    ax_w.set_ylabel("hidden dim 1")
    ax_w.grid(True)

    # 4. trace lines + points
    line_objs = [ax_w.plot([], [], color=colors[j], lw=1.8)[0] for j in range(n_feat)]
    scat = ax_w.scatter(Ws[0, 0], Ws[0, 1], s=60, c=colors, edgecolors='k', zorder=3)
    text_objs = [ax_w.text(Ws[0, 0, j], Ws[0, 1, j], str(j), fontsize=9, ha='left', va='bottom') for j in range(n_feat)]

    def _init():
        for j, ln in enumerate(line_objs):
            ln.set_data([Ws[0, 0, j]], [Ws[0, 1, j]])
        scat.set_offsets(Ws[0].T)
        for j, txt in enumerate(text_objs):
            txt.set_position((Ws[0, 0, j], Ws[0, 1, j]))
            txt.set_text(str(j))
        # loss_line.set_data([run_numbers[0]], [loss_values[0]]) # Removed
        current_run_vline.set_xdata([run_numbers[0]] if run_numbers else plot_xlim_loss[0])
        ax_w.set_title(f"projection of features into hidden space (W)")
        return (*line_objs, scat, *text_objs, current_run_vline) # Removed loss_line, Added current_run_vline

    def _update(frame):
        for j, ln in enumerate(line_objs):
            ln.set_data(Ws[:frame + 1, 0, j], Ws[:frame + 1, 1, j])
        scat.set_offsets(Ws[frame].T)
        for j, txt in enumerate(text_objs):
            txt.set_position((Ws[frame, 0, j], Ws[frame, 1, j]))

        cur_loss = loss_values[frame]
        ax_w.set_title(f"run {run_numbers[frame]} / {run_numbers[-1]}   –   loss = {cur_loss:.4f}")

        # Update vertical line for current run
        if run_numbers: # Ensure run_numbers is not empty
            current_run_vline.set_xdata([run_numbers[frame]])

        return (*line_objs, scat, *text_objs, current_run_vline) # Removed loss_line, Added current_run_vline

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init,
        frames=n_frames, interval=interval_ms,
        blit=True, repeat_delay=1000
    )

    fig.tight_layout()
    return fig, anim


# ───────────────────────────────  CLI  ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Visualise training log with traces and live loss.")
    ap.add_argument("log", nargs="?", default="log",
                    help="path to log file (default: ./log)")
    ap.add_argument("--save", choices=("gif", "mp4"),
                    help="Save the animation instead of displaying it")
    ap.add_argument("--interval", type=int, default=100,
                    help="Animation frame interval in ms (default: 100)")
    args = ap.parse_args()

    runs = parse_log(args.log)
    if not runs:
        sys.exit("No runs parsed – aborting.")

    fig, anim = build_figure_and_animation(runs, interval_ms=args.interval)

    if args.save:
        outfile = Path(args.log).with_suffix(f".{args.save}")
        print(f"Saving animation → {outfile}")
        if args.save == "gif":
            anim.save(outfile, writer="pillow", fps=max(1, 1000 // args.interval))
        else:
            anim.save(outfile, writer="ffmpeg", dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
