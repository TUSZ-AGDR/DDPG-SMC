"""
Simulation and comparison: Fixed-gain SMC vs DDPG time-varying-gain SMC.

Loads a trained DDPG actor, runs the square trajectory with both
fixed and adaptive gains using the *exact* run_episode loop from
online_xyz_ddpg, and produces publication-style plots matching
the pid_smc.py colour scheme.  All figures are saved as SVG.

Usage:
    python online_xyz_ddpg/simulation.py
    python online_xyz_ddpg/simulation.py --results online_xyz_ddpg/results_final
    python online_xyz_ddpg/simulation.py --fixed-gains 1.8 1.3 0.3 0.8 1.7 0.3 1.2 5.5 0.2
"""
#python online_xyz_ddpg\simulation.py --fixed-gains 1.8 2.5 0.3 0.5 1.2 0.3 1.2 4.9 0.2
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_xyz_ddpg as core

COLORS = {
    'ADAPTIVE': 'tab:blue',
    'FIXED':    'tab:orange',
    'REF':      'k--',
}

AXIS_NAMES = ['X', 'Y', 'Z']
POS_LABELS = ['X Distance (m)', 'Y Distance (m)', 'Z Distance (m)']
ERR_LABELS = ['Error X (m)', 'Error Y (m)', 'Error Z (m)']
GAIN_NAMES = [r'$\lambda$', r'$K_D$', r'$\delta$']


class ZeroActor(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 9), dtype=x.dtype, device=x.device)


def make_dummy_optimizers(actor, device):
    critic = core.Critic().to(device)
    target_actor = core.ResidualActor().to(device)
    target_critic = core.Critic().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-5)
    return (actor, critic, target_actor, target_critic, actor_opt, critic_opt)


def compute_rmse(state_log, ref_log):
    err = state_log[:, :3] - ref_log
    rmse_axis = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_3d = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))
    rmse_pooled = np.sqrt(np.mean(err ** 2))
    return rmse_axis, rmse_3d, rmse_pooled


def print_rmse_table(rmse_fixed, rmse_adaptive):
    labels = ['X', 'Y', 'Z', '3D', 'Pooled']
    fixed_vals = list(rmse_fixed[0]) + [rmse_fixed[1], rmse_fixed[2]]
    adap_vals = list(rmse_adaptive[0]) + [rmse_adaptive[1], rmse_adaptive[2]]
    print("\n" + "=" * 72)
    print("  RMSE COMPARISON (Square Trajectory)")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Fixed-Gain':>14} {'Adaptive-Gain':>14} {'Reduction':>12}")
    print("  " + "-" * 58)
    for label, fv, av in zip(labels, fixed_vals, adap_vals):
        red = (fv - av) / fv * 100.0
        print(f"  {label:<10} {fv:>14.6f} {av:>14.6f} {red:>11.2f}%")
    print("=" * 72)


def plot_position_tracking(t, ref, fixed, adaptive):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Position vs Time -- Fixed-Gain vs Adaptive-Gain SMC',
                 fontsize=14, fontweight='bold')
    for i in range(3):
        axes[i].plot(t, ref[:, i], COLORS['REF'], lw=2, label='Reference')
        axes[i].plot(t, adaptive[:, i], color=COLORS['ADAPTIVE'],
                     lw=2.5, label='Variable-Gain DDPG-SMC')
        axes[i].plot(t, fixed[:, i], color=COLORS['FIXED'],
                     lw=2.5, linestyle='--', label='Fixed-Gain SMC')
        axes[i].set_ylabel(POS_LABELS[i], fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(fontsize=10, loc='best')
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_tracking_errors(t, ref, fixed, adaptive):
    err_fixed = fixed[:, :3] - ref
    err_adaptive = adaptive[:, :3] - ref
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Tracking Error -- Fixed-Gain vs Adaptive-Gain SMC',
                 fontsize=14, fontweight='bold')
    for i in range(3):
        axes[i].plot(t, err_adaptive[:, i], color=COLORS['ADAPTIVE'],
                     lw=2, label='Variable-Gain DDPG-SMC')
        axes[i].plot(t, err_fixed[:, i], color=COLORS['FIXED'],
                     lw=2, linestyle='--', label='Fixed-Gain SMC')
        axes[i].axhline(0, color='k', lw=0.8, ls='--')
        axes[i].set_ylabel(ERR_LABELS[i], fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(fontsize=10, loc='best')
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_3d_trajectory(ref, fixed, adaptive):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
            COLORS['REF'], lw=2, label='Reference')
    ax.plot(adaptive[:, 0], adaptive[:, 1], adaptive[:, 2],
            color=COLORS['ADAPTIVE'], lw=2, label='Variable-Gain DDPG-SMC')
    ax.plot(fixed[:, 0], fixed[:, 1], fixed[:, 2],
            color=COLORS['FIXED'], lw=2, linestyle='--',
            label='Fixed-Gain SMC')
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2],
               color='black', s=40, zorder=5, label='Start')
    ax.set_xlabel('X Distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Distance (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Distance (m)', fontsize=12, fontweight='bold')
    ax.set_title('3D Trajectory -- Fixed-Gain vs Adaptive-Gain SMC',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    return fig


def plot_gain_history(t, gains_log):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Time-Varying Position SMC Gains (DDPG Actor)',
                 fontsize=14, fontweight='bold')
    groups = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
    for panel, (indices, gain_label) in enumerate(zip(groups, GAIN_NAMES)):
        for idx, axis_name in zip(indices, AXIS_NAMES):
            axes[panel].plot(t, gains_log[:, idx], lw=2,
                             label=f'{gain_label}_{axis_name}')

        axes[panel].set_ylabel(gain_label, fontsize=13, fontweight='bold')
        axes[panel].legend(fontsize=10, ncol=3, loc='best')
        axes[panel].grid(True, alpha=0.25)
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fixed vs Adaptive DDPG-SMC simulation and plots")
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "figures",
                        help="Directory to save SVG figures.")
    parser.add_argument("--fixed-gains", type=float, nargs=9, default=None,
                        metavar="Lx Kx dx Ly Ky dy Lz Kz dz",
                        help="Override fixed position gains (9 values: "
                             "lam_x K_x delta_x lam_y K_y delta_y "
                             "lam_z K_z delta_z). Default uses the baseline "
                             "from online_xyz_ddpg.py.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    weights_path = args.results / "online_xyz_ddpg_actor_critic.pt"
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        print("Run training first: python online_xyz_ddpg/online_xyz_ddpg.py "
              "--episodes 60 --train-scenario both "
              "--output online_xyz_ddpg/results")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = core.ResidualActor().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    print(f"Loaded actor from {weights_path}")
    print(f"Device: {device}")

    replay = core.ReplayBuffer()
    zero_actor = ZeroActor().to(device)
    dummy_opts = make_dummy_optimizers(actor, device)

    original_base_gains = core.BASE_POSITION_GAINS.copy()
    original_gain_min = core.GAIN_MIN.copy()
    original_gain_max = core.GAIN_MAX.copy()

    if args.fixed_gains is not None:
        custom_gains = np.array(args.fixed_gains, dtype=np.float64)
        core.BASE_POSITION_GAINS[:] = custom_gains
        core.BASE_LAM_P[:] = custom_gains[0::3]
        core.BASE_K_P[:] = custom_gains[1::3]
        core.BASE_DELTA_P[:] = custom_gains[2::3]
        core.GAIN_MIN[:] = custom_gains * (1.0 - core.MAX_RELATIVE_GAIN_CHANGE)
        core.GAIN_MAX[:] = custom_gains * (1.0 + core.MAX_RELATIVE_GAIN_CHANGE)
        print(f"Overriding fixed gains: {custom_gains}")

    print("\nRunning fixed-gain square trajectory...")
    fixed_result = core.run_episode(
        zero_actor, core.TRAJ_SQUARE, False, replay, dummy_opts, device,
        core.MASS_NOMINAL, None, gamma=0.99, tau=0.005,
        policy_decimation=core.DEFAULT_POLICY_DECIMATION,
        learn_every=core.DEFAULT_LEARN_EVERY,
        batch_size=core.DEFAULT_BATCH_SIZE)

    print("Running adaptive-gain square trajectory...")
    adap_result = core.run_episode(
        actor, core.TRAJ_SQUARE, False, replay, dummy_opts, device,
        core.MASS_NOMINAL, None, gamma=0.99, tau=0.005,
        policy_decimation=core.DEFAULT_POLICY_DECIMATION,
        learn_every=core.DEFAULT_LEARN_EVERY,
        batch_size=core.DEFAULT_BATCH_SIZE,
        initial_action=core.POLICY_PRIOR_ACTION)

    fixed_state = fixed_result["state"]
    fixed_ref = fixed_result["reference"]
    adap_state = adap_result["state"]
    adap_ref = adap_result["reference"]
    adap_actions = adap_result["actions"]

    core.BASE_POSITION_GAINS[:] = original_base_gains
    core.GAIN_MIN[:] = original_gain_min
    core.GAIN_MAX[:] = original_gain_max

    gains_log = np.array([core.gains_from_action(a) for a in adap_actions])

    steps = len(fixed_state)
    t = np.arange(steps) * core.DT

    rmse_fixed = compute_rmse(fixed_state, fixed_ref)
    rmse_adaptive = compute_rmse(adap_state, adap_ref)
    print_rmse_table(rmse_fixed, rmse_adaptive)

    gain_names_9 = ['lam_x', 'K_x', 'delta_x',
                    'lam_y', 'K_y', 'delta_y',
                    'lam_z', 'K_z', 'delta_z']
    fixed_gains_display = np.array(args.fixed_gains, dtype=np.float64) if args.fixed_gains is not None else core.BASE_POSITION_GAINS
    print("\n  Gain Summary: Fixed vs DDPG Actual Range")
    print("  " + "=" * 62)
    print(f"  {'Gain':<10} {'Fixed':>8} {'DDPG Min':>10} {'DDPG Max':>10} {'Admissible':>14}")
    print("  " + "-" * 62)
    for i, name in enumerate(gain_names_9):
        gmin = gains_log[:, i].min()
        gmax = gains_log[:, i].max()
        print(f"  {name:<10} {fixed_gains_display[i]:>8.4f} {gmin:>10.4f} {gmax:>10.4f}"
              f"    [{core.GAIN_MIN[i]:.4f}, {core.GAIN_MAX[i]:.4f}]")
    print("  " + "=" * 62)

    fig_pos = plot_position_tracking(t, fixed_ref, fixed_state[:, :3], adap_state[:, :3])
    fig_err = plot_tracking_errors(t, fixed_ref, fixed_state, adap_state)
    fig_3d = plot_3d_trajectory(fixed_ref, fixed_state[:, :3], adap_state[:, :3])
    fig_gain = plot_gain_history(t, gains_log)

    svg_names = [
        "position_tracking.svg",
        "tracking_error.svg",
        "3d_trajectory.svg",
        "gain_history.svg",
    ]
    for fig, name in zip([fig_pos, fig_err, fig_3d, fig_gain], svg_names):
        svg_path = args.output / name
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Saved {svg_path}")

    plt.show()


if __name__ == "__main__":
    main()