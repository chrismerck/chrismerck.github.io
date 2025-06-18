"""
example_history_gen.py

Generate a synthetic PPO-like training history that conforms to the
format required by ``ppo_visual.py``.

The script produces a pickled list of dictionaries – one per epoch –
with the mandatory keys:

    values : ndarray(shape=(rows, cols))
    policy : ndarray(shape=(rows, cols, 4))  # Up, Right, Down, Left
    loss   : float
    mask   : ndarray(bool, same shape as values)

Run::

    python history_gen.py example_history.pkl           # default 6×6 grid, 50 epochs
    python history_gen.py my.pkl --rows 8 --cols 8 --epochs 120

GENERATED WITH o3
FROM PROMPT:
Give an example_history_gen.py that can make an example for testing ppo_visual.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _make_obstacles(rows: int, cols: int, fraction: float = 0.1, rng: np.random.Generator | None = None):
    """Return boolean mask with randomly placed obstacles."""
    rng = rng or np.random.default_rng()
    mask = np.zeros((rows, cols), dtype=bool)
    n_obstacles = int(rows * cols * fraction)
    choices = rng.choice(rows * cols, size=n_obstacles, replace=False)
    mask.flat[choices] = True
    return mask


def _compute_final_values(rows: int, cols: int, goals: list[tuple[int, int, float]], mask: np.ndarray):
    """Compute a handcrafted value-function surface.

    `goals` is a list of (row, col, reward).
    The value of a state is the maximum over  (reward - distance).
    Distance uses Manhattan metric.
    """
    v = np.full((rows, cols), -np.inf)
    ys, xs = np.mgrid[0:rows, 0:cols]
    for gy, gx, rew in goals:
        dist = np.abs(ys - gy) + np.abs(xs - gx)
        candidate = rew - 0.5 * dist
        v = np.maximum(v, candidate)
    v[mask] = np.nan  # obstacles
    return v


def _best_action(values: np.ndarray, r: int, c: int):
    """Return argmax over neighbour value (0=Up,1=Right,2=Down,3=Left)."""
    rows, cols = values.shape
    neighbours = [
        values[r - 1, c] if r > 0 else -np.inf,
        values[r, c + 1] if c < cols - 1 else -np.inf,
        values[r + 1, c] if r < rows - 1 else -np.inf,
        values[r, c - 1] if c > 0 else -np.inf,
    ]
    
    # Handle case where all neighbors are NaN (isolated cell)
    if all(np.isnan(n) or n == -np.inf for n in neighbours):
        return 0  # Default to Up action
    
    return int(np.nanargmax(neighbours))


def _policy_array(values: np.ndarray, progress: float, rng: np.random.Generator):
    """Generate smooth policy probabilities using sinusoidal functions."""
    rows, cols = values.shape
    policy = np.empty((rows, cols, 4), dtype=float)
    
    for r in range(rows):
        for c in range(cols):
            if np.isnan(values[r, c]):
                policy[r, c] = 0.0
                continue
            
            # Get the best action based on value function
            best = _best_action(values, r, c)
            
            # Create smooth sinusoidal variations for each action
            # Different frequency and phase for each action to create interesting patterns
            probs = np.zeros(4)
            
            # Time-varying base using progress
            t = progress * 2 * np.pi * 3  # 3 full cycles over training
            
            # Position-based phase offset for spatial variation
            spatial_phase = (r / rows + c / cols) * np.pi
            
            for action in range(4):
                # Different frequency for each action
                freq = 1 + action * 0.5
                
                # Phase offset for each action
                phase = action * np.pi / 2 + spatial_phase
                
                # Rectified sine wave (only positive values)
                raw_value = np.sin(t * freq + phase)
                rectified = max(0, raw_value)
                
                # Scale based on whether this is the best action
                if action == best:
                    # Best action gets higher baseline and amplitude
                    probs[action] = 0.4 + 0.4 * rectified
                else:
                    # Other actions get lower baseline and amplitude
                    probs[action] = 0.1 + 0.2 * rectified
            
            # Ensure probabilities sum to 1
            probs = np.clip(probs, 0.01, None)
            probs /= probs.sum()
            policy[r, c] = probs
    
    return policy

# --------------------------------------------------------------------------- #
# Synthetic history generator
# --------------------------------------------------------------------------- #

def generate_history(rows: int = 6, cols: int = 6, epochs: int = 50, obstacle_frac: float = 0.1, seed: int | None = 0):
    rng = np.random.default_rng(seed)
    mask = _make_obstacles(rows, cols, obstacle_frac, rng)
    # Hand-crafted goal states: (row, col, reward)
    goals = [
        (0, cols - 1, 5.0),          # top-right +5
        (rows // 2, 0, 4.0),         # mid-left +4
        (rows - 1, cols - 1, -1.0),  # bottom-right –1
    ]

    final_values = _compute_final_values(rows, cols, goals, mask)
    initial_values = np.zeros_like(final_values)

    history: list[dict] = []
    base_loss = 1.0
    for ep in range(epochs):
        progress = (ep + 1) / epochs
        # Smooth interpolation of values using a sigmoid-like curve
        smooth_progress = 0.5 * (1 + np.sin((progress - 0.5) * np.pi))
        values = (1 - smooth_progress) * initial_values + smooth_progress * final_values
        values[mask] = np.nan

        policy = _policy_array(final_values, progress, rng)
        # Smooth exponential decay for loss
        loss = base_loss * (0.95 ** ep)

        history.append({
            "values": values,
            "policy": policy,
            "loss": float(loss),
            "mask": mask.copy(),
        })
    return history


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PPO training history.")
    parser.add_argument("outfile", type=Path, help="Filename for pickled history.")
    parser.add_argument("--rows", type=int, default=6, help="Grid rows (default: 6)")
    parser.add_argument("--cols", type=int, default=6, help="Grid cols (default: 6)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--obstacles", type=float, default=0.1, help="Fraction of cells that are obstacles (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility (default: 0)")
    args = parser.parse_args()

    hist = generate_history(args.rows, args.cols, args.epochs, args.obstacles, args.seed)
    with open(args.outfile, "wb") as f:
        pickle.dump(hist, f)
    print(f"History written to {args.outfile!s}  (epochs: {args.epochs}, grid: {args.rows}×{args.cols})")


if __name__ == "__main__":
    main()
