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
    Ws = np.stack([r["W"][:3] for r in runs_W])   # (frames, 3, n_feat)
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
    ax_w    = fig.add_subplot(gs[1], projection='3d')

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
                # Adjust x-coordinates for the moving average to align with the center of the window.
                offset = (window_size - 1) // 2
                # The moving_avg array has length: len(all_loss_vals) - window_size + 1.
                # The x-coordinates should start from the run corresponding to the center of the first window.
                moving_avg_runs = all_loss_runs[offset : offset + len(moving_avg)]
                if len(moving_avg_runs) == len(moving_avg): # This check should generally pass with correct slicing
                    ax_loss.plot(moving_avg_runs, moving_avg, color='black', lw=1.2, linestyle='-', label=f'MA ({window_size})')
                    ax_loss.legend(fontsize='small')

    # Add a vertical line for current run
    current_run_vline = ax_loss.axvline(x=run_numbers[0] if run_numbers else plot_xlim_loss[0], color='red', linestyle='--', lw=1)

    # 2. 3D embedding plot
    xmin, xmax = Ws[:, 0, :].min(), Ws[:, 0, :].max()
    ymin, ymax = Ws[:, 1, :].min(), Ws[:, 1, :].max()
    zmin, zmax = Ws[:, 2, :].min(), Ws[:, 2, :].max()
    margin = 0.1 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    ax_w.set_xlim(xmin - margin, xmax + margin)
    ax_w.set_ylim(ymin - margin, ymax + margin)
    ax_w.set_zlim(zmin - margin, zmax + margin)
    ax_w.set_xlabel("hidden dim 0")
    ax_w.set_ylabel("hidden dim 1")
    ax_w.set_zlabel("hidden dim 2")
    ax_w.grid(True)

    # Initial camera view angles and step size for Brownian motion
    current_azim = -60.0  # Initial azimuth
    current_elev = 30.0   # Initial elevation
    # angle_step_std_dev = 0.5  # Standard deviation for angle steps (degrees) # Old parameter

    # Parameters for momentum-based camera movement
    azim_velocity = 0.0
    elev_velocity = 0.0
    acceleration_factor = 0.2  # How strongly random impulses affect velocity
    damping_factor = 0.8      # How quickly velocity decays (0 to 1)

    ax_w.view_init(elev=current_elev, azim=current_azim)

    # 4. trace lines + points
    # connector_line, = ax_w.plot([], [], [], color='gray', linestyle='-', lw=1.0, alpha=0.7) # Old single connector

    # New: lines for each pair of heads
    pair_lines = []
    if n_feat > 1:
        for i in range(n_feat):
            for k in range(i + 1, n_feat):
                line, = ax_w.plot([], [], [], color='gray', linestyle='-', lw=0.5, alpha=0.5)
                pair_lines.append(line)

    scat = ax_w.scatter(Ws[0, 0, :], Ws[0, 1, :], Ws[0, 2, :], s=60, c=colors, edgecolors='k', zorder=3)
    text_objs = [ax_w.text(Ws[0, 0, j], Ws[0, 1, j], Ws[0, 2, j], str(j), fontsize=9, ha='left', va='bottom') for j in range(n_feat)]

    def _init():
        # if n_feat > 1: # Old connector init
        #     connector_line.set_data_3d(Ws[0, 0, :], Ws[0, 1, :], Ws[0, 2, :])
        
        # New: init pair lines
        line_idx = 0
        if n_feat > 1:
            for i in range(n_feat):
                for k in range(i + 1, n_feat):
                    pair_lines[line_idx].set_data_3d(
                        [Ws[0, 0, i], Ws[0, 0, k]],  # x coordinates
                        [Ws[0, 1, i], Ws[0, 1, k]],  # y coordinates
                        [Ws[0, 2, i], Ws[0, 2, k]]   # z coordinates
                    )
                    line_idx += 1
        
        scat._offsets3d = (Ws[0, 0, :], Ws[0, 1, :], Ws[0, 2, :])
        for j, txt in enumerate(text_objs):
            txt.set_position((Ws[0, 0, j], Ws[0, 1, j]))
            txt.set_text(str(j))
            txt.set_x(Ws[0,0,j])
            txt.set_y(Ws[0,1,j])
            if hasattr(txt, 'set_z'):
                txt.set_z(Ws[0,2,j])
            else:
                txt.set_position_3d((Ws[0,0,j], Ws[0,1,j], Ws[0,2,j]))
        current_run_vline.set_xdata([run_numbers[0]] if run_numbers else plot_xlim_loss[0])
        ax_w.set_title(f"projection of features into hidden space (W)")
        # return (connector_line, scat, *text_objs, current_run_vline) # Old return
        return (*pair_lines, scat, *text_objs, current_run_vline)

    def _update(frame):
        # if n_feat > 1: # Old connector update
        #     connector_line.set_data_3d(Ws[frame, 0, :], Ws[frame, 1, :], Ws[frame, 2, :])

        # New: update pair lines
        line_idx = 0
        if n_feat > 1:
            for i in range(n_feat):
                for k in range(i + 1, n_feat):
                    pair_lines[line_idx].set_data_3d(
                        [Ws[frame, 0, i], Ws[frame, 0, k]],
                        [Ws[frame, 1, i], Ws[frame, 1, k]],
                        [Ws[frame, 2, i], Ws[frame, 2, k]]
                    )
                    line_idx += 1

        scat._offsets3d = (Ws[frame, 0, :], Ws[frame, 1, :], Ws[frame, 2, :])
        for j, txt in enumerate(text_objs):
            txt.set_x(Ws[frame,0,j])
            txt.set_y(Ws[frame,1,j])
            if hasattr(txt, 'set_z'):
                txt.set_z(Ws[frame,2,j])
            else:
                txt.set_position_3d((Ws[frame,0,j], Ws[frame,1,j], Ws[frame,2,j]))

        cur_loss = loss_values[frame]
        ax_w.set_title(f"run {run_numbers[frame]} / {run_numbers[-1]}   –   loss = {cur_loss:.4f}")

        # Update vertical line for current run
        if run_numbers: # Ensure run_numbers is not empty
            current_run_vline.set_xdata([run_numbers[frame]])

        # Update camera view with momentum
        nonlocal current_azim, current_elev, azim_velocity, elev_velocity # Allow modification
        
        import math
        current_azim += 3
        current_elev = math.sin(frame / 10) * 15 + 15
        
        # Keep elevation within reasonable bounds (e.g., 0 to 90 degrees to avoid flipping)
        current_elev = np.clip(current_elev, 5, 85) # Adjusted bounds slightly for better viewing
        # Azimuth can wrap around, so no explicit clipping needed unless you want to restrict its range.
        
        ax_w.view_init(elev=current_elev, azim=current_azim)

        # return (connector_line, scat, *text_objs, current_run_vline) # Old return
        return (*pair_lines, scat, *text_objs, current_run_vline)

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init,
        frames=n_frames, interval=interval_ms,
        blit=False, repeat_delay=1000 # Changed blit to False
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
