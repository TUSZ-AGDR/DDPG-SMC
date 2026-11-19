"""Plot saved fixed-gain and variable-gain DDPG-SMC trajectories.

This script does not retrain the actor and does not rerun the quadrotor
simulation.  It reads the ``*_adaptive.npz`` and ``*_fixed.npz`` files saved
by ``online_xyz_ddpg.py`` and creates comparison figures.

Example:
    python plot_online_ddpg_comparison.py \
        --results online_ddpg_60_smooth \
        --output online_ddpg_60_smooth/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXIS_LABELS = ("X", "Y", "Z")
COLORS = {"adaptive": "#1565c0", "fixed": "#d84315", "reference": "#222222"}


def load_case(results_dir: Path, name: str):
    adaptive_path = results_dir / f"{name}_adaptive.npz"
    fixed_path = results_dir / f"{name}_fixed.npz"
    if not adaptive_path.exists() or not fixed_path.exists():
        raise FileNotFoundError(
            f"Expected {adaptive_path.name} and {fixed_path.name} in {results_dir}"
        )
    adaptive = np.load(adaptive_path)
    fixed = np.load(fixed_path)
    n = min(len(adaptive["state"]), len(fixed["state"]))
    t = np.arange(n) * 0.005
    return t, adaptive, fixed, n


def plot_xyz(t, adaptive, fixed, name, output):
    n = len(t)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i, label in enumerate(AXIS_LABELS):
        axes[i].plot(t, adaptive["reference"][:n, i], "--", color=COLORS["reference"],
                     linewidth=1.1, label="Reference")
        axes[i].plot(t, adaptive["state"][:n, i], color=COLORS["adaptive"],
                     linewidth=1.15, label="Variable-gain DDPG-SMC")
        # Draw the fixed-gain result last with a dashed red line so it remains
        # visible even when both controllers produce nearly identical paths.
        axes[i].plot(t, fixed["state"][:n, i], color=COLORS["fixed"],
                     linestyle="--", linewidth=1.35, label="Fixed-gain SMC")
        axes[i].set_ylabel(f"{label} (m)")
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(ncol=3, loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{name.capitalize()} trajectory: fixed versus variable SMC gains")
    fig.tight_layout()
    path = output / f"{name}_xyz_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_3d(t, adaptive, fixed, name, output):
    n = len(t)
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ref = adaptive["reference"][:n]
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "--", color=COLORS["reference"],
            linewidth=1.2, label="Reference")
    ax.plot(adaptive["state"][:n, 0], adaptive["state"][:n, 1], adaptive["state"][:n, 2],
            color=COLORS["adaptive"], linewidth=1.25, label="Variable-gain DDPG-SMC")
    ax.plot(fixed["state"][:n, 0], fixed["state"][:n, 1], fixed["state"][:n, 2],
            color=COLORS["fixed"], linestyle="--", linewidth=1.45,
            label="Fixed-gain SMC")
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="black", s=24, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{name.capitalize()} 3-D trajectory comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    path = output / f"{name}_3d_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gains(results_dir, name, output):
    adaptive_path = results_dir / f"{name}_adaptive.npz"
    data = np.load(adaptive_path)
    actions = data["actions"]
    t = np.arange(len(actions)) * 0.005
    base = np.array([2.0, 1.5, 0.25,
                     0.9, 1.9, 0.25,
                     1.0, 5.0, 0.25])
    # Must match MAX_RELATIVE_GAIN_CHANGE in online_xyz_ddpg.py.
    relative_limit = np.array([0.30, 0.15, 0.15,
                               0.30, 0.15, 0.15,
                               0.30, 0.15, 0.15])
    gains = base * (1.0 + relative_limit * actions)
    labels = (r"$\lambda$", r"$K$", r"$\delta$")
    axis_labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    groups = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    for i, (label, indices) in enumerate(zip(labels, groups)):
        for axis_label, index in zip(axis_labels, indices):
            axes[i].plot(t, gains[:, index], linewidth=1.2,
                         label=f"{label}_{axis_label}")
            axes[i].axhline(base[index], color=COLORS["fixed"], linestyle="--",
                            linewidth=0.8)
        axes[i].set_ylabel(label)
        axes[i].legend(ncol=3)
        axes[i].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{name.capitalize()} variable position SMC gains")
    fig.tight_layout()
    path = output / f"{name}_gain_history.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot saved DDPG-SMC comparisons")
    parser.add_argument("--results", type=Path, required=True,
                        help="Folder containing *_adaptive.npz and *_fixed.npz")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output if args.output is not None else args.results / "plots"
    output.mkdir(parents=True, exist_ok=True)

    for name in ("hover", "square"):
        t, adaptive, fixed, _ = load_case(args.results, name)
        paths = [plot_xyz(t, adaptive, fixed, name, output),
                 plot_3d(t, adaptive, fixed, name, output),
                 plot_gains(args.results, name, output)]
        for path in paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
