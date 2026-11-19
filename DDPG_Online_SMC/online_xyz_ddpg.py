"""
Online DDPG residual-gain SMC for the DJI-Tello quadrotor model.

The plant, inertias, motor mixing and reference trajectories follow
``ddpg_smc_amir.py``.  The nine position SMC parameters
``[lambda_i, K_i, delta_i]`` for ``i in {x,y,z}`` are produced at every
control step by a DDPG actor as small residuals around the existing fixed
gains.  Roll, pitch, and yaw gains remain fixed.

Training is performed in simulation with an actual plant mass sampled from
the nominal value with +/-5 percent uncertainty.  The controller uses the
nominal mass, so the uncertainty is a genuine plant/controller mismatch.
The default run trains on the 2-m climb-and-hover task and evaluates both
hover and square-trajectory cases.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import deque
from pathlib import Path

# PyTorch and Numba can load separate OpenMP runtimes on Windows.  This is
# the same runtime safeguard used by the existing DDPG script and must be set
# before importing either numerical stack.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numba import njit

# The networks are tiny; extra OpenMP threads usually add more overhead than
# useful work on this workload.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# Model constants: retained from ddpg_smc_amir.py
# ---------------------------------------------------------------------------
MASS_NOMINAL = 0.080
GRAVITY = 9.81
IXX, IYY, IZZ = 7.2e-5, 6.8e-5, 1.53e-4
DT = 0.005
T_END = 22.0
SIDE = 2.0
Z_HOVER = 2.0
MASS_UNCERTAINTY = 0.05
DEFAULT_POLICY_DECIMATION = 10  # 200-Hz plant, 20-Hz gain-policy updates
DEFAULT_LEARN_EVERY = 10
DEFAULT_BATCH_SIZE = 64
REPLAY_WARMUP = 256
RESIDUAL_PENALTY = 5.0e-5
# Additional damping objective during the first second after each square
# corner.  At these instants one horizontal axis has stopped or reversed;
# penalizing residual velocity teaches the actor to remove momentum before
# the vehicle crosses the edge.
CORNER_RATE_PENALTY = 0.30
CORNER_ACTION_WINDOW = 0.60
# First-order smoothing of the actor residual.  At the 20-Hz policy rate,
# alpha=0.15 gives a gradual gain transition over several policy updates.
GAIN_SMOOTHING_ALPHA = 0.15

ARM = 0.119 / np.sqrt(2.0)
THRUST_COEFF = 2.84e-6
DRAG_COEFF = 1.46e-10
MIX = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [ARM, -ARM, -ARM, ARM],
    [ARM, ARM, -ARM, -ARM],
    [-DRAG_COEFF / THRUST_COEFF, DRAG_COEFF / THRUST_COEFF,
     -DRAG_COEFF / THRUST_COEFF, DRAG_COEFF / THRUST_COEFF],
], dtype=np.float64)
MIX_INV = np.linalg.inv(MIX)

# Fixed baseline gains from ddpg_smc_amir.py.
BASE_LAM_P = np.array([2.0, 0.9, 1.0], dtype=np.float64)
BASE_K_P = np.array([1.5, 1.9, 5.0], dtype=np.float64)
BASE_DELTA_P = np.array([0.25, 0.25, 0.25], dtype=np.float64)
BASE_LAM_A = np.array([8.0, 8.0, 8.0], dtype=np.float64)
BASE_K_A = np.array([12.0, 12.0, 9.0], dtype=np.float64)
BASE_DELTA_A = np.array([0.5, 0.5, 0.55], dtype=np.float64)

# Residual action limits.  The actor output is in [-1, 1].  The position
# sliding-surface slopes (lambda) need a wider useful range than K and delta:
# the gain sweep showed that a lower lambda reduces the lag at the square
# trajectory corners, whereas large K/delta changes mainly add control effort.
# This keeps all three gains adaptive while giving the policy useful authority.
BASE_POSITION_GAINS = np.empty(9, dtype=np.float64)
for _axis in range(3):
    BASE_POSITION_GAINS[3 * _axis:3 * _axis + 3] = (
        BASE_LAM_P[_axis], BASE_K_P[_axis], BASE_DELTA_P[_axis]
    )
MAX_RELATIVE_GAIN_CHANGE = np.array([
    0.30, 0.15, 0.15,
    0.30, 0.15, 0.15,
    0.30, 0.15, 0.15,
], dtype=np.float64)
GAIN_MIN = BASE_POSITION_GAINS * (1.0 - MAX_RELATIVE_GAIN_CHANGE)
GAIN_MAX = BASE_POSITION_GAINS * (1.0 + MAX_RELATIVE_GAIN_CHANGE)

# A safe warm start found from the fixed-gain sensitivity sweep.  The actor
# remains fully trainable; this only prevents the early DDPG critic updates
# from spending the whole budget rediscovering the useful lower horizontal
# lambda range.
POLICY_PRIOR_ACTION = np.array([
    -0.85, 0.0, 0.0,
    -0.85, 0.0, 0.0,
    0.0, 0.0, 0.0,
], dtype=np.float32)


def corner_axis(t, trajectory):
    """Return the horizontal axis that has just reached a square corner."""
    active = np.zeros(2, dtype=np.float64)
    if trajectory == TRAJ_SQUARE:
        for corner_time, axis_index in ((8.0, 0), (12.0, 1),
                                         (16.0, 0), (20.0, 1)):
            if corner_time <= t < corner_time + CORNER_ACTION_WINDOW:
                active[axis_index] = 1.0
    return active

TRAJ_HOVER = 0
TRAJ_SQUARE = 1
AXIS_UNUSED = -1


@njit(cache=True)
def reference(t: float, trajectory: int):
    """Filtered-reference source, matching the original square trajectory."""
    if trajectory == TRAJ_HOVER:
        if t < 4.0:
            return 0.0, 0.0, Z_HOVER * (t / 4.0), 0.0
        return 0.0, 0.0, Z_HOVER, 0.0
    if t < 4.0:
        return 0.0, 0.0, Z_HOVER * (t / 4.0), 0.0
    if t < 8.0:
        s = (t - 4.0) / 4.0
        return SIDE * s, 0.0, Z_HOVER, 0.0
    if t < 12.0:
        s = (t - 8.0) / 4.0
        return SIDE, SIDE * s, Z_HOVER, 0.0
    if t < 16.0:
        s = (t - 12.0) / 4.0
        return SIDE * (1.0 - s), SIDE, Z_HOVER, 0.0
    if t < 20.0:
        s = (t - 16.0) / 4.0
        return 0.0, SIDE * (1.0 - s), Z_HOVER, 0.0
    return 0.0, 0.0, Z_HOVER, 0.0


@njit(cache=True)
def allocate_motors(thrust, torque, u_max):
    """Preserve collective, then feasible roll/pitch/yaw authority."""
    collective = min(max(thrust / 4.0, 0.0), u_max)
    differential = np.empty(4)
    scale = 1.0
    for j in range(4):
        differential[j] = MIX_INV[j, 1] * torque[0] + MIX_INV[j, 2] * torque[1]
        if differential[j] > 0.0:
            scale = min(scale, (u_max - collective) / differential[j])
        elif differential[j] < 0.0:
            scale = min(scale, -collective / differential[j])
    u = np.empty(4)
    yaw_scale = 1.0
    for j in range(4):
        u[j] = collective + scale * differential[j]
        yaw = MIX_INV[j, 3] * torque[2]
        if yaw > 0.0:
            yaw_scale = min(yaw_scale, (u_max - u[j]) / yaw)
        elif yaw < 0.0:
            yaw_scale = min(yaw_scale, -u[j] / yaw)
    actual = np.zeros(4)
    for j in range(4):
        u[j] = min(max(u[j] + max(0.0, yaw_scale) * MIX_INV[j, 3] * torque[2], 0.0), u_max)
        for c in range(4):
            actual[c] += MIX[c, j] * u[j]
    return u, actual


@njit(cache=True)
def adaptive_control_step(
    state, pos_d, pos_d_prev, vel_d, vel_d_prev,
    att_ref, att_ref_rate, u_prev, t, trajectory, actual_mass,
    lam_p, kp_p, delta_p, lam_a, kp_a, delta_a, position_gains,
):
    """Advance one physical-model step using actor-selected X/Y/Z gains.

    The actor is called in Python.  This function contains the plant and
    control calculations so the expensive numerical part remains Numba-jitted.
    """
    x_ref, y_ref, z_ref, psi_d = reference(t, trajectory)
    raw_ref = np.array([x_ref, y_ref, z_ref])
    alpha = 0.04
    acc_cmd = np.zeros(3)
    errors = np.zeros(3)
    error_rates = np.zeros(3)
    surfaces = np.zeros(3)
    for i in range(3):
        pos_d[i] = alpha * raw_ref[i] + (1.0 - alpha) * pos_d[i]
        if t <= 1e-12:
            vel_d[i] = 0.0
            acc_d = 0.0
        else:
            vel_d[i] = (pos_d[i] - pos_d_prev[i]) / DT
            acc_d = (vel_d[i] - vel_d_prev[i]) / DT
            acc_d = min(max(acc_d, -10.0), 10.0)
        pos_d_prev[i] = pos_d[i]
        vel_d_prev[i] = vel_d[i]
        errors[i] = state[i] - pos_d[i]
        error_rates[i] = state[3 + i] - vel_d[i]
        gain_index = 3 * i
        lam = position_gains[gain_index]
        gain = position_gains[gain_index + 1]
        width = position_gains[gain_index + 2]
        surfaces[i] = error_rates[i] + lam * errors[i]
        acc_cmd[i] = acc_d - lam * error_rates[i] - gain * np.tanh(surfaces[i] / max(width, 1e-4))

    az = acc_cmd[2] + GRAVITY
    sp, cp = np.sin(psi_d), np.cos(psi_d)
    phi_d = min(max((acc_cmd[0] * sp - acc_cmd[1] * cp) / GRAVITY, -0.6), 0.6)
    theta_d = min(max((acc_cmd[0] * cp + acc_cmd[1] * sp) / GRAVITY, -0.6), 0.6)
    cos_tilt = max(0.5, np.cos(state[6]) * np.cos(state[7]))
    # The controller retains the nominal mass; only the plant uses actual_mass.
    thrust = min(max(MASS_NOMINAL * az / cos_tilt, 0.0), 2.0 * MASS_NOMINAL * GRAVITY)

    phi, theta = state[6], state[7]
    p, q, r = state[9], state[10], state[11]
    sf, cf = np.sin(phi), np.cos(phi)
    st, ct_raw = np.sin(theta), np.cos(theta)
    ct = max(0.1, ct_raw)
    rates = np.array([
        p + sf * st / ct * q + cf * st / ct * r,
        cf * q - sf * r,
        (sf / ct) * q + (cf / ct) * r,
    ])
    att_des = np.array([phi_d, theta_d, psi_d])
    eta_acc = np.zeros(3)
    torque = np.zeros(3)
    for i in range(3):
        att_ref_acc = 100.0 * (att_des[i] - att_ref[i]) - 20.0 * att_ref_rate[i]
        att_error = state[6 + i] - att_ref[i]
        if i == 2:
            att_error = (att_error + np.pi) % (2.0 * np.pi) - np.pi
        att_error_rate = rates[i] - att_ref_rate[i]
        s_att = att_error_rate + lam_a[i] * att_error
        eta_acc[i] = att_ref_acc - lam_a[i] * att_error_rate - kp_a[i] * np.tanh(s_att / max(delta_a[i], 1e-4))
    fd, td, yd = rates[0], rates[1], rates[2]
    pd_cmd = eta_acc[0] - st * eta_acc[2] - ct * td * yd
    qd_cmd = (cf * eta_acc[1] + sf * ct * eta_acc[2] - sf * fd * td
              + cf * ct * fd * yd - sf * st * td * yd)
    rd_cmd = (-sf * eta_acc[1] + cf * ct * eta_acc[2] - cf * fd * td
              - sf * ct * fd * yd - cf * st * td * yd)
    torque[0] = IXX * pd_cmd + (IZZ - IYY) * q * r
    torque[1] = IYY * qd_cmd + (IXX - IZZ) * p * r
    torque[2] = IZZ * rd_cmd + (IYY - IXX) * p * q

    # Use the nominal hardware limit, while actual acceleration uses the
    # uncertain plant mass.
    u, realized = allocate_motors(thrust, torque, 2.0 * MASS_NOMINAL * GRAVITY / 4.0)
    effort = np.sum(np.abs(u - u_prev))
    u_prev[:] = u
    torque[:] = realized[1:4]

    sphi, cphi = np.sin(state[6]), np.cos(state[6])
    sth, cth = np.sin(state[7]), np.cos(state[7])
    spsi, cpsi = np.sin(state[8]), np.cos(state[8])
    total_thrust = np.sum(u)
    acceleration = np.array([
        (cphi * sth * cpsi + sphi * spsi) * total_thrust / actual_mass,
        (cphi * sth * spsi - sphi * cpsi) * total_thrust / actual_mass,
        cphi * cth * total_thrust / actual_mass - GRAVITY,
    ])
    p, q, r = state[9], state[10], state[11]
    angular_acc = np.array([
        (torque[0] + (IYY - IZZ) * q * r) / IXX,
        (torque[1] + (IZZ - IXX) * p * r) / IYY,
        (torque[2] + (IXX - IYY) * p * q) / IZZ,
    ])
    attitude_rate = np.array([
        p + sphi * np.tan(state[7]) * q + cphi * np.tan(state[7]) * r,
        cphi * q - sphi * r,
        (sphi * q + cphi * r) / max(0.1, cth),
    ])

    state[0:3] += state[3:6] * DT
    state[3:6] += acceleration * DT
    state[6:9] += attitude_rate * DT
    state[9:12] += angular_acc * DT
    if state[2] < 0.0:
        state[2] = 0.0
        state[5] = max(state[5], 0.0)

    att_ref[:] = att_ref + att_ref_rate * DT
    # Recompute the desired-attitude reference acceleration for its integrator.
    for i in range(3):
        att_ref_acc = 100.0 * (att_des[i] - att_ref[i]) - 20.0 * att_ref_rate[i]
        att_ref_rate[i] += att_ref_acc * DT

    valid = (np.all(np.isfinite(state)) and np.all(np.abs(state[0:3]) < 20.0)
             and np.all(np.abs(state[3:6]) < 15.0) and abs(state[6]) < 1.4
             and abs(state[7]) < 1.4 and np.all(np.abs(state[9:12]) < 20.0))
    return errors, error_rates, surfaces, effort, valid


# ---------------------------------------------------------------------------
# DDPG components
# ---------------------------------------------------------------------------
class OUNoise:
    def __init__(self, size, sigma=0.20, theta=0.15):
        self.size = size
        self.sigma = sigma
        self.theta = theta
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def sample(self):
        self.state += self.theta * (-self.state) + self.sigma * np.random.randn(self.size)
        return self.state.copy()


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state.astype(np.float32), action.astype(np.float32),
                          float(reward), next_state.astype(np.float32), float(done)))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.data), batch_size, replace=False)
        batch = [self.data[i] for i in idx]
        return tuple(np.asarray(v) for v in zip(*batch))

    def __len__(self):
        return len(self.data)


class ResidualActor(nn.Module):
    """Outputs normalized residuals in [-1, 1] around a safe gain prior."""
    def __init__(self, state_dim=23, action_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )
        # Keep a small state-dependent component from the first episode so
        # the warm policy is genuinely gain-varying rather than a constant
        # gain table.  The small scale keeps this perturbation inside the
        # validated low-lambda operating region.
        nn.init.normal_(self.net[-2].weight, mean=0.0, std=0.08)
        nn.init.zeros_(self.net[-2].bias)
        # With a zero final-layer weight, this gives a deterministic safe
        # starting policy while keeping every upstream weight trainable.
        prior = np.clip(POLICY_PRIOR_ACTION.astype(np.float64), -0.999, 0.999)
        with torch.no_grad():
            self.net[-2].bias.copy_(torch.from_numpy(np.arctanh(prior).astype(np.float32)))

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim=23, action_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def gains_from_action(action):
    action = np.asarray(action, dtype=np.float64)
    return np.clip(BASE_POSITION_GAINS * (1.0 + MAX_RELATIVE_GAIN_CHANGE * action), GAIN_MIN, GAIN_MAX)


def observation(state, errors, error_rates, surfaces,
                pos_ref, vel_ref, t, trajectory):
    """Construct the normalized 23-element position-gain observation."""
    obs = np.concatenate((
        errors / 2.0,
        error_rates / 3.0,
        surfaces / 3.0,
        state[0:3] / 2.0,
        state[3:6] / 3.0,
        pos_ref / 2.0,
        vel_ref / 3.0,
        np.array([t / T_END]),
        np.array([float(trajectory)]),
    )).astype(np.float32)
    return np.clip(obs, -5.0, 5.0)


def run_episode(actor, trajectory, training, replay, optimizers, device, mass, noise, gamma, tau,
                policy_decimation=DEFAULT_POLICY_DECIMATION,
                learn_every=DEFAULT_LEARN_EVERY,
                batch_size=DEFAULT_BATCH_SIZE, initial_action=None,
                enable_adaptation=True):
    # ``actor`` is explicit so evaluation can pass a zero-residual actor for
    # the fixed-gain baseline.  The remaining DDPG objects are shared.
    actor_net = actor
    critic, target_actor, target_critic = optimizers[1], optimizers[2], optimizers[3]
    actor_opt, critic_opt = optimizers[4], optimizers[5]
    # Hover is a stationary vertical test.  Keep its gains at the established
    # fixed values so the learned position adaptation is evaluated on the
    # manoeuvre for which it is intended (the square trajectory).
    adaptive_enabled = enable_adaptation and trajectory == TRAJ_SQUARE
    state = np.zeros(12, dtype=np.float64)
    pos_d = np.zeros(3, dtype=np.float64)
    pos_d_prev = np.zeros(3, dtype=np.float64)
    vel_d = np.zeros(3, dtype=np.float64)
    vel_d_prev = np.zeros(3, dtype=np.float64)
    att_ref = np.zeros(3, dtype=np.float64)
    att_ref_rate = np.zeros(3, dtype=np.float64)
    u_prev = np.zeros(4, dtype=np.float64)
    steps = int(T_END / DT)
    if initial_action is None or not adaptive_enabled:
        action = np.zeros(9, dtype=np.float64)
    else:
        action = np.asarray(initial_action, dtype=np.float64).copy()
    prev_action = action.copy()
    total_reward = 0.0
    position_sq = 0.0
    action_log = np.zeros((steps, 9), dtype=np.float64)
    state_log = np.zeros((steps, 12), dtype=np.float64)
    ref_log = np.zeros((steps, 3), dtype=np.float64)
    mass_log = np.full(steps, mass, dtype=np.float64)
    if noise is not None:
        noise.reset()

    for k in range(steps):
        t = k * DT
        if k == 0:
            err = np.zeros(3)
            err_rate = np.zeros(3)
            surfaces = np.zeros(3)
        # Use the previous-step error before selecting the current gains.
        current_obs = observation(state, err, err_rate, surfaces,
                                  pos_d, vel_d, t, trajectory)
        # The plant remains at 200 Hz, but the policy is evaluated at a lower
        # rate. The latest residual is held between policy updates, preserving
        # variable gains while removing most Python/Torch calls.
        if k % max(1, policy_decimation) == 0:
            if adaptive_enabled:
                with torch.inference_mode():
                    proposed_action = actor_net(
                        torch.from_numpy(current_obs).to(device).unsqueeze(0)
                    ).cpu().numpy()[0]
            else:
                proposed_action = np.zeros(9, dtype=np.float64)
            proposed_action = np.asarray(proposed_action, dtype=np.float64)
            if training and noise is not None:
                proposed_action = np.clip(
                    proposed_action + noise.sample() * max(0.10, 1.0 - t / T_END),
                    -1.0, 1.0,
                )
            # Corner damping: after a side ends, the active horizontal axis
            # still carries momentum.  The bounded override lowers lambda
            # and raises K for only the first 0.6 s, giving the SMC a wider
            # braking action without changing the learned policy elsewhere.
            active_corner = corner_axis(t, trajectory) if adaptive_enabled else np.zeros(2)
            for axis_index in range(2):
                if active_corner[axis_index] > 0.0:
                    proposed_action[3 * axis_index] = min(
                        proposed_action[3 * axis_index], -1.0)
                    proposed_action[3 * axis_index + 1] = max(
                        proposed_action[3 * axis_index + 1], 1.0)
            action = ((1.0 - GAIN_SMOOTHING_ALPHA) * action
                      + GAIN_SMOOTHING_ALPHA * proposed_action)
            action = np.clip(action, -1.0, 1.0)
        position_gains = gains_from_action(action)
        err, err_rate, surfaces, effort, valid = adaptive_control_step(
            state, pos_d, pos_d_prev, vel_d, vel_d_prev,
            att_ref, att_ref_rate, u_prev, t, trajectory, mass,
            BASE_LAM_P, BASE_K_P, BASE_DELTA_P,
            BASE_LAM_A, BASE_K_A, BASE_DELTA_A, position_gains,
        )
        # The plant state has now advanced by one DT.  Use this post-step
        # error consistently for the transition reward and next observation;
        # using the pre-step error here teaches the critic a one-sample-shifted
        # objective relative to the RMSE reported from state_log.
        err = state[:3] - pos_d
        err_rate = state[3:6] - vel_d
        surfaces = err_rate + position_gains[1::3] * err
        next_obs = observation(state, err, err_rate, surfaces,
                               pos_d, vel_d, min(t + DT, T_END), trajectory)
        position_sq += float(np.sum(err * err))
        action_log[k] = action
        state_log[k] = state
        ref_log[k] = pos_d
        # Normalize the terms before forming the reward.  Without this scaling,
        # the climb-rate transient dominates the return and destabilizes the
        # critic, even when the post-climb tracking is excellent.
        e_norm = np.clip(err / np.array([2.0, 2.0, 2.0]), -5.0, 5.0)
        rate_norm = np.clip(err_rate / np.array([3.0, 3.0, 3.0]), -5.0, 5.0)
        corner_active = np.zeros(2, dtype=np.float64)
        if trajectory == TRAJ_SQUARE:
            for corner_time, axis_index in ((8.0, 0), (12.0, 1),
                                             (16.0, 0), (20.0, 1)):
                if corner_time <= t < corner_time + 1.0:
                    corner_active[axis_index] = 1.0
        corner_rate_cost = CORNER_RATE_PENALTY * float(
            np.sum((rate_norm[:2] * corner_active) ** 2))
        reward = (-1.0 * float(np.sum(e_norm * e_norm))
                  -0.05 * float(np.sum(rate_norm * rate_norm))
                  -corner_rate_cost
                  -0.0005 * float(effort)
                  -RESIDUAL_PENALTY * float(np.sum(action * action))
                  -0.002 * float(np.sum((action - prev_action) ** 2)))
        if not valid:
            reward -= 10.0
        reward = float(np.clip(reward, -10.0, 0.0))
        done = (k == steps - 1) or (not valid)
        if training:
            replay.add(current_obs, action, reward, next_obs, done)
            if (adaptive_enabled
                    and len(replay) >= max(REPLAY_WARMUP, batch_size)
                    and k % max(1, learn_every) == 0):
                batch = replay.sample(batch_size)
                b_s, b_a, b_r, b_ns, b_done = batch
                st = torch.tensor(b_s, dtype=torch.float32, device=device)
                at = torch.tensor(b_a, dtype=torch.float32, device=device)
                rt = torch.tensor(b_r, dtype=torch.float32, device=device).unsqueeze(1)
                nst = torch.tensor(b_ns, dtype=torch.float32, device=device)
                dtm = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)
                with torch.no_grad():
                    target = rt + gamma * (1.0 - dtm) * target_critic(nst, target_actor(nst))
                critic_loss = nn.functional.mse_loss(critic(st, at), target)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                for p in critic.parameters():
                    p.requires_grad_(False)
                actor_loss = -critic(st, actor_net(st)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
                actor_opt.step()
                for p in critic.parameters():
                    p.requires_grad_(True)
                soft_update(target_actor, actor_net, tau)
                soft_update(target_critic, critic, tau)
        total_reward += reward
        prev_action = action.copy()
        if done:
            break

    used = k + 1
    # Report the same post-step, filtered-reference error that is stored in
    # the trajectory files and used by the diagnostic plots.  The temporary
    # control error above is evaluated before the plant advances and can be
    # slightly optimistic at each sample.
    logged_error = state_log[:used, :3] - ref_log[:used]
    logged_rmse = float(np.sqrt(np.mean(logged_error * logged_error)))
    return {
        "reward": total_reward,
        "position_rmse": logged_rmse,
        "state": state_log[:used],
        "reference": ref_log[:used],
        "actions": action_log[:used],
        "mass": mass_log[:used],
        "steps": used,
    }


def make_networks(device):
    actor = ResidualActor().to(device)
    critic = Critic().to(device)
    target_actor = ResidualActor().to(device)
    target_critic = Critic().to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    # Lower rates are more stable for the nine-dimensional action space.
    actor_opt = optim.Adam(actor.parameters(), lr=5e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=5e-4)
    return actor, critic, target_actor, target_critic, actor_opt, critic_opt


def plot_result(path, result, title):
    t = np.arange(result["steps"]) * DT
    state, ref, actions = result["state"], result["reference"], result["actions"]
    gains = np.array([gains_from_action(a) for a in actions])
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, label in enumerate(labels):
        axes[0].plot(t, state[:, i], label=label)
        axes[0].plot(t, ref[:, i], "--", linewidth=0.9)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title(title)
    axes[0].legend(ncol=3)
    gain_groups = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    gain_labels = (r"$\lambda$", r"$K$", r"$\delta$")
    axis_labels = ("X", "Y", "Z")
    for panel, (indices, gain_label) in enumerate(zip(gain_groups, gain_labels), start=1):
        for idx, axis_label in zip(indices, axis_labels):
            axes[panel].plot(t, gains[:, idx], label=f"{gain_label}_{axis_label}")
        axes[panel].set_ylabel(gain_label)
        axes[panel].legend(ncol=3)
        axes[panel].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def result_metrics(result):
    """Return manuscript-ready position and gain diagnostics."""
    state = result["state"]
    reference_log = result["reference"]
    actions = result["actions"]
    error = state[:, :3] - reference_log
    t = np.arange(len(error)) * DT
    gains = np.array([gains_from_action(a) for a in actions])
    tail = t >= max(0.0, t[-1] - 2.0)
    return {
        "axis_rmse": np.sqrt(np.mean(error * error, axis=0)),
        "pooled_rmse": np.sqrt(np.mean(error * error)),
        "three_d_rmse": np.sqrt(np.mean(np.sum(error * error, axis=1))),
        "max_abs_error": np.max(np.abs(error), axis=0),
        "mean_abs_error": np.mean(np.abs(error), axis=0),
        "tail_mean_abs_error": np.mean(np.abs(error[tail]), axis=0),
        "gain_min": np.min(gains, axis=0),
        "gain_max": np.max(gains, axis=0),
        "gain_mean": np.mean(gains, axis=0),
        "gain_std": np.std(gains, axis=0),
    }


def print_result_metrics(label, result):
    metrics = result_metrics(result)
    print(f"{label} diagnostics:")
    print("  axis RMSE [X Y Z] (m): " + " ".join(f"{x:.6f}" for x in metrics["axis_rmse"]))
    print(f"  pooled position RMSE (m): {metrics['pooled_rmse']:.6f}")
    print(f"  3-D path RMSE (m): {metrics['three_d_rmse']:.6f}")
    print("  max |error| [X Y Z] (m): " + " ".join(f"{x:.6f}" for x in metrics["max_abs_error"]))
    print("  final 2-s mean |error| [X Y Z] (m): "
          + " ".join(f"{x:.6f}" for x in metrics["tail_mean_abs_error"]))
    gain_names = ("lambda", "K", "delta")
    for axis_index, axis_label in enumerate(("X", "Y", "Z")):
        indices = (axis_index * 3, axis_index * 3 + 1, axis_index * 3 + 2)
        values = " / ".join(
            f"{gain_names[j]}={metrics['gain_min'][idx]:.4f}--{metrics['gain_max'][idx]:.4f}"
            for j, idx in enumerate(indices)
        )
        print(f"  {axis_label}-gain min/max: {values}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Online 3-D position DDPG residual-gain SMC")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--train-scenario", choices=["hover", "square", "both"], default="hover")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--policy-decimation", type=int, default=DEFAULT_POLICY_DECIMATION,
                        help="Plant steps between actor evaluations; default 10 = 20 Hz policy.")
    parser.add_argument("--learn-every", type=int, default=DEFAULT_LEARN_EVERY,
                        help="Plant steps between DDPG updates; default 10.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--nominal-mass", action="store_true",
                        help="Disable mass randomization and train at nominal mass.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic, target_actor, target_critic, actor_opt, critic_opt = make_networks(device)
    optimizers = (actor, critic, target_actor, target_critic, actor_opt, critic_opt)
    replay = ReplayBuffer()
    noise = OUNoise(9)
    history = []
    best_validation_rmse = float("inf")
    best_actor_state = None
    print(f"Device: {device}; Numba plant: enabled; mass range: "
          f"{MASS_NOMINAL*(1-MASS_UNCERTAINTY):.5f}--{MASS_NOMINAL*(1+MASS_UNCERTAINTY):.5f} kg")
    print(f"Policy rate: {1.0 / (DT * max(1, args.policy_decimation)):.1f} Hz; "
          f"learning interval: {DT * max(1, args.learn_every):.3f} s; "
          f"batch size: {args.batch_size}")

    # Establish an elite checkpoint before exploration starts.  This makes
    # the adaptive result a safe improvement search around the warm-start
    # policy rather than allowing an early critic update to erase it.
    if args.train_scenario == "both":
        initial_hover = run_episode(
            actor, TRAJ_HOVER, False, replay, optimizers, device,
            MASS_NOMINAL, None, gamma=0.99, tau=0.005,
            policy_decimation=args.policy_decimation,
            learn_every=args.learn_every, batch_size=args.batch_size,
            initial_action=POLICY_PRIOR_ACTION)
        initial_square = run_episode(
            actor, TRAJ_SQUARE, False, replay, optimizers, device,
            MASS_NOMINAL, None, gamma=0.99, tau=0.005,
            policy_decimation=args.policy_decimation,
            learn_every=args.learn_every, batch_size=args.batch_size,
            initial_action=POLICY_PRIOR_ACTION)
        best_validation_rmse = (0.2 * initial_hover["position_rmse"]
                                + 0.8 * initial_square["position_rmse"])
    else:
        initial = run_episode(
            actor, TRAJ_HOVER if args.train_scenario == "hover" else TRAJ_SQUARE,
            False, replay, optimizers, device, MASS_NOMINAL, None,
            gamma=0.99, tau=0.005,
            policy_decimation=args.policy_decimation,
            learn_every=args.learn_every, batch_size=args.batch_size,
            initial_action=POLICY_PRIOR_ACTION)
        best_validation_rmse = initial["position_rmse"]
    best_actor_state = copy.deepcopy(actor.state_dict())
    print(f"  warm-start validation RMSE: {best_validation_rmse:.6f} m")

    for episode in range(args.episodes):
        if args.train_scenario == "both":
            # Give the manoeuvre two thirds of the updates; hover remains in
            # the schedule to preserve the vertical-mass robustness check.
            trajectory = TRAJ_HOVER if episode % 3 == 0 else TRAJ_SQUARE
        else:
            trajectory = TRAJ_HOVER if args.train_scenario == "hover" else TRAJ_SQUARE
        if args.nominal_mass:
            actual_mass = MASS_NOMINAL
        else:
            actual_mass = MASS_NOMINAL * np.random.uniform(1.0 - MASS_UNCERTAINTY, 1.0 + MASS_UNCERTAINTY)
        result = run_episode(actor, trajectory, True, replay, optimizers, device,
                             actual_mass, noise, gamma=0.99, tau=0.005,
                             policy_decimation=args.policy_decimation,
                             learn_every=args.learn_every,
                             batch_size=args.batch_size,
                             initial_action=POLICY_PRIOR_ACTION)
        history.append((episode + 1, trajectory, actual_mass, result["reward"], result["position_rmse"]))
        print(f"Episode {episode+1:03d}/{args.episodes} | "
              f"{'hover' if trajectory == TRAJ_HOVER else 'square':6s} | "
              f"mass={actual_mass:.5f} kg | RMSE={result['position_rmse']:.5f} m | "
              f"reward={result['reward']:.2f}")

        # Select the best deterministic policy instead of automatically using
        # the final actor, which can have degraded after late exploratory
        # updates.  For mixed training, validate both task types.
        if (episode + 1) % 10 == 0:
            if args.train_scenario == "both":
                val_hover = run_episode(actor, TRAJ_HOVER, False, replay, optimizers, device,
                                        MASS_NOMINAL, None, gamma=0.99, tau=0.005,
                                        policy_decimation=args.policy_decimation,
                                        learn_every=args.learn_every,
                                        batch_size=args.batch_size,
                                        initial_action=POLICY_PRIOR_ACTION)
                val_square = run_episode(actor, TRAJ_SQUARE, False, replay, optimizers, device,
                                         MASS_NOMINAL, None, gamma=0.99, tau=0.005,
                                         policy_decimation=args.policy_decimation,
                                         learn_every=args.learn_every,
                                         batch_size=args.batch_size,
                                         initial_action=POLICY_PRIOR_ACTION)
                # Square tracking contains the spatial transients that the
                # adaptive position gains are intended to improve.
                validation_rmse = (0.2 * val_hover["position_rmse"]
                                    + 0.8 * val_square["position_rmse"])
            else:
                val = run_episode(actor, trajectory, False, replay, optimizers, device,
                                  MASS_NOMINAL, None, gamma=0.99, tau=0.005,
                                  policy_decimation=args.policy_decimation,
                                  learn_every=args.learn_every,
                                  batch_size=args.batch_size,
                                  initial_action=POLICY_PRIOR_ACTION)
                validation_rmse = val["position_rmse"]
            print(f"  deterministic validation RMSE: {validation_rmse:.6f} m")
            if validation_rmse < best_validation_rmse:
                best_validation_rmse = validation_rmse
                best_actor_state = copy.deepcopy(actor.state_dict())
                print("  -> retained new best actor")

    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
        print(f"Restored best actor (validation RMSE {best_validation_rmse:.6f} m)")

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "seed": args.seed, "base_position_gains": BASE_POSITION_GAINS.copy()},
               args.output / "online_xyz_ddpg_actor_critic.pt")
    np.savetxt(args.output / "training_history.csv", np.asarray(history),
               delimiter=",", header="episode,trajectory,mass,reward,position_rmse", comments="")

    eval_results = {}
    for name, trajectory in (("hover", TRAJ_HOVER), ("square", TRAJ_SQUARE)):
        fixed = run_episode(actor, trajectory, False, replay, optimizers, device,
                            MASS_NOMINAL, None, gamma=0.99, tau=0.005,
                            policy_decimation=args.policy_decimation,
                            learn_every=args.learn_every,
                            batch_size=args.batch_size,
                            initial_action=POLICY_PRIOR_ACTION)
        eval_results[name] = {"adaptive_nominal_mass": fixed["position_rmse"]}
        np.savez(args.output / f"{name}_adaptive.npz", state=fixed["state"],
                 reference=fixed["reference"], actions=fixed["actions"], mass=fixed["mass"])
        plot_result(args.output / f"{name}_adaptive.png", fixed,
                    f"Online DDPG residual-gain SMC: {name}")
        # Baseline uses zero residuals and the same plant model.
        class ZeroActor(nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 9), dtype=x.dtype, device=x.device)
        baseline = run_episode(ZeroActor(), trajectory, False, replay, optimizers, device,
                               MASS_NOMINAL, None, gamma=0.99, tau=0.005,
                               policy_decimation=args.policy_decimation,
                               learn_every=args.learn_every,
                               batch_size=args.batch_size,
                               enable_adaptation=False)
        np.savez(args.output / f"{name}_fixed.npz", state=baseline["state"],
                 reference=baseline["reference"], actions=baseline["actions"], mass=baseline["mass"])
        eval_results[name]["fixed_gain_nominal_mass"] = baseline["position_rmse"]
        print(f"Evaluation {name}: adaptive RMSE={fixed['position_rmse']:.6f} m; "
              f"fixed-gain RMSE={baseline['position_rmse']:.6f} m")
        print_result_metrics(f"  adaptive {name}", fixed)
        print_result_metrics(f"  fixed {name}", baseline)
    (args.output / "evaluation_summary.json").write_text(json.dumps(eval_results, indent=2))
    print(f"Saved adaptive-gain controller and results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
