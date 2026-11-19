"""

Quadcopter SMC Position Gain Optimization via DDPG - S-Curve Trajectory

- Optimizes 3 parameters per POSITION axis only: [lambda, K_smc, delta (phi)]

- Channels: X, Y, Z (inner attitude loop stays fixed at hand-picked gains)

- Trajectory: S-curve (sine-wave sweep in XY plane at fixed hover height)

"""

import os

import random

from collections import deque

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

torch.manual_seed(42)

np.random.seed(42)

random.seed(42)

def run_quad_simulation(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, kd_p, active_axis='xyz', traj_type='s_curve', evaluate_reward=False):

    m   = 0.080

    g   = 9.81

    Ixx = 7.2e-5

    Iyy = 6.8e-5

    Izz = 1.53e-4

    l   = 0.119 / np.sqrt(2)

    b   = 2.84e-6

    d   = 1.46e-10

    mix = np.array([

        [1.0,    1.0,    1.0,    1.0],

        [0.0,   -l,      0.0,    l  ],

        [l,      0.0,   -l,      0.0],

        [-d/b,   d/b,   -d/b,    d/b]

    ])

    mix_inv = np.linalg.inv(mix)

    dt     = 0.005

    alpha  = 0.04

    phi_p = phi_p.copy()

    phi_a_in = phi_a_in.copy()

    for i in range(3):

        if kp_smc[i] / max(phi_p[i], 1e-6) > 8.0:

            phi_p[i] = kp_smc[i] / 8.0

        if ka_smc[i] / max(phi_a_in[i], 1e-6) > 12.0:

            phi_a_in[i] = ka_smc[i] / 12.0

    T_end = 24.0

    steps = int(T_end / dt)

    state = np.zeros(12)

    log     = np.zeros((steps, 12))

    ref_log = np.zeros((steps, 3))

    t_log   = np.zeros(steps)

    pos_d_filtered = np.array([0.0, 0.0, 0.0])

    psi_d_target = 0.0

    pos_d_prev   = np.array([0.0, 0.0, 0.0])

    vel_d_prev   = np.zeros(3)

    def step_reference(t):

        if traj_type == 's_curve':

            z_start  = 2.0

            z_peak   = 4.0

            ramp_T   = 3.0

            if t < ramp_T:

                z_ref = z_start * (t / ramp_T)

                x_ref, y_ref = 0.0, 0.0

            else:

                tt = t - ramp_T

                T_s = 16.0

                phase = min(tt / T_s, 1.0)

                X_span = 3.0

                A_y    = 1.5

                A_z    = 0.5

                z_delta = z_peak - z_start

                x_ref = X_span * phase

                y_ref = A_y * np.sin(2.0 * np.pi * phase)

                z_ref = z_start + z_delta * phase + A_z * np.sin(2.0 * np.pi * phase)

            return np.array([x_ref, y_ref, z_ref]), 0.0

        else:

            side    = 2.0

            T       = 4.0

            z_hover = 2.0

            if t < T:

                x_ref, y_ref, z_ref = 0.0, 0.0, z_hover * (t / T)

            elif t < 2*T:

                s = (t - T) / T

                x_ref, y_ref, z_ref = side * s, 0.0, z_hover

            elif t < 3*T:

                s = (t - 2*T) / T

                x_ref, y_ref, z_ref = side, side * s, z_hover

            elif t < 4*T:

                s = (t - 3*T) / T

                x_ref, y_ref, z_ref = side * (1 - s), side, z_hover

            elif t < 5*T:

                s = (t - 4*T) / T

                x_ref, y_ref, z_ref = 0.0, side * (1 - s), z_hover

            else:

                x_ref, y_ref, z_ref = 0.0, 0.0, z_hover

            return np.array([x_ref, y_ref, z_ref]), 0.0

    def sat(s, phi_val):

        return np.tanh(s / np.maximum(phi_val, 1e-3))

    control_effort_total = 0.0

    error_integral = 0.0

    total_episode_reward = 0.0

    u_prev = np.zeros(4)

    for k in range(steps):

        t = k * dt

        pos   = state[0:3]

        vel   = state[3:6]

        att   = state[6:9]

        omega = state[9:12]

        if np.any(np.isnan(state)) or np.any(np.isinf(state)):

            return 1e6, -1e5, log, ref_log, t_log

        omega = np.clip(omega, -20.0, 20.0)

        pos_d_raw, psi_d = step_reference(t)

        pos_d_filtered = alpha * pos_d_raw + (1.0 - alpha) * pos_d_filtered

        pos_d = pos_d_filtered

        if k == 0:

            vel_d = np.zeros(3)

            acc_d = np.zeros(3)

        else:

            vel_d = (pos_d - pos_d_prev) / dt

            acc_d = (vel_d - vel_d_prev) / dt

            vel_d = 0.5 * vel_d + 0.5 * ((pos_d - pos_d_prev) / dt)

            acc_d = np.clip(acc_d, -10.0, 10.0)

        pos_d_prev = pos_d.copy()

        vel_d_prev = vel_d.copy()

        e     = pos - pos_d

        e_dot = vel - vel_d

        s     = e_dot + lam_p * e

        acc_cmd = acc_d - kp_smc * sat(s, phi_p)

        az = acc_cmd[2] + g

        phi_d   = (acc_cmd[0] * np.sin(psi_d) - acc_cmd[1] * np.cos(psi_d)) / g

        theta_d = (acc_cmd[0] * np.cos(psi_d) + acc_cmd[1] * np.sin(psi_d)) / g

        phi_d   = min(max(phi_d, -0.6), 0.6)

        theta_d = min(max(theta_d, -0.6), 0.6)

        cos_tilt = max(0.5, np.cos(phi_d) * np.cos(theta_d))

        T_max = 2.0 * m * g

        T = min(max(m * az / cos_tilt, 0.0), T_max)

        att_d = np.array([phi_d, theta_d, psi_d])

        att_e     = att - att_d

        att_e_dot = omega - np.zeros(3)

        att_s     = att_e_dot + lam_a * att_e

        I_diag = np.array([Ixx, Iyy, Izz])

        tau = I_diag * (-ka_smc * sat(att_s, phi_a_in))

        u_max = 2.0 * m * g / 4.0

        collective = min(max(T / 4.0, 0.0), u_max)

        differential = np.empty(4)

        scale = 1.0

        for j in range(4):

            differential[j] = mix_inv[j, 1] * tau[0] + mix_inv[j, 2] * tau[1]

            if differential[j] > 0.0:

                scale = min(scale, (u_max - collective) / differential[j])

            elif differential[j] < 0.0:

                scale = min(scale, -collective / differential[j])

        u = np.empty(4)

        yaw_scale = 1.0

        for j in range(4):

            u[j] = collective + scale * differential[j]

            yaw = mix_inv[j, 3] * tau[2]

            if yaw > 0.0:

                yaw_scale = min(yaw_scale, (u_max - u[j]) / yaw)

            elif yaw < 0.0:

                yaw_scale = min(yaw_scale, -u[j] / yaw)

        for j in range(4):

            u[j] = min(max(u[j] + max(0.0, yaw_scale) * mix_inv[j, 3] * tau[2], 0.0), u_max)

        actual_wrench = mix @ u

        error_integral += np.sum(np.abs(e)) * dt

        control_effort_total += np.sum(np.abs(u[1:] - u[:-1])) * dt

        if evaluate_reward:

            ex, ey, ez = e[0], e[1], e[2]

            ex_dot, ey_dot, ez_dot = e_dot[0], e_dot[1], e_dot[2]

            e_sq = np.sum(e ** 2)

            r1 = 6.0 * np.exp(-2.0 * (ex**2 + ey**2 + ez**2))

            dev_sq = np.sum((pos - pos_d)**2)

            sdes_sq = np.sum(pos_d ** 2)

            excess_dev = max(0.0, dev_sq - 0.25 * sdes_sq)

            r2 = -5.0 * excess_dev

            r3 = -4.0 * e_sq - 5.0 * (ex ** 2) - 5.0 * (ey ** 2) - 6.0 * (ez ** 2)

            u_change = np.sum((u - u_prev) ** 2)

            r4 = -4.0 * u_change - 4.0 * (ex_dot ** 2) - 4.0 * (ey_dot ** 2) - 5.0 * (ez_dot ** 2)

            u_prev = u.copy()

            r5 = -0.01

            r_step = (r1 + r2 + r3 + r4 + r5) * dt

            total_episode_reward += r_step

        R = np.array([

            [np.cos(att[1])*np.cos(att[2]), np.sin(att[0])*np.sin(att[1])*np.cos(att[2]) - np.cos(att[0])*np.sin(att[2]), np.cos(att[0])*np.sin(att[1])*np.cos(att[2]) + np.sin(att[0])*np.sin(att[2])],

            [np.cos(att[1])*np.sin(att[2]), np.sin(att[0])*np.sin(att[1])*np.sin(att[2]) + np.cos(att[0])*np.cos(att[2]), np.cos(att[0])*np.sin(att[1])*np.sin(att[2]) - np.sin(att[0])*np.cos(att[2])],

            [-np.sin(att[1]),               np.sin(att[0])*np.cos(att[1]),                                           np.cos(att[0])*np.cos(att[1])]

        ])

        thrust_world = R @ np.array([0, 0, np.sum(u)])

        acc_lin = thrust_world / m - np.array([0, 0, g])

        p, q, r = omega

        tau_realized = actual_wrench[1:4]

        acc_ang = np.array([

            (tau_realized[0] + (Iyy - Izz) * q * r) / Ixx,

            (tau_realized[1] + (Izz - Ixx) * p * r) / Iyy,

            (tau_realized[2] + (Ixx - Iyy) * p * q) / Izz

        ])

        phi, theta = att[0], att[1]

        cth = max(0.1, np.cos(theta))

        tth = np.tan(theta)

        sphi, cphi = np.sin(phi), np.cos(phi)

        att_dot = np.array([

            p + sphi * tth * q + cphi * tth * r,

            cphi * q - sphi * r,

            (sphi / cth) * q + (cphi / cth) * r

        ])

        state[0:3]  += vel   * dt

        state[3:6]  += acc_lin * dt

        state[6:9]  += att_dot * dt

        state[9:12] += acc_ang * dt

        if state[2] < 0:

            state[2] = 0

            state[5] = max(state[5], 0)

        state[3:6]  = np.clip(state[3:6], -15.0, 15.0)

        state[6:9]  = np.clip(state[6:9], -np.pi, np.pi)

        state[9:12] = np.clip(state[9:12], -20.0, 20.0)

        log[k]     = state

        ref_log[k] = pos_d

        t_log[k]   = t

    cost = error_integral + 0.01 * control_effort_total

    return cost, total_episode_reward, log, ref_log, t_log

class OUNoise:

    def __init__(self, size=3, mu=0.0, theta=0.15, sigma=0.15):

        self.mu = mu * np.ones(size)

        self.theta = theta

        self.sigma = sigma

        self.reset()

    def reset(self):

        self.state = self.mu.copy()

    def sample(self):

        x = self.state

        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))

        self.state = x + dx

        return self.state

class ReplayBuffer:

    def __init__(self, capacity=5000):

        self.buffer = deque(maxlen=capacity)

        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):

        self.buffer.append((state, action, reward, next_state))

        self.priorities.append(abs(reward) + 1e-6)

    def sample(self, batch_size=32):

        probs = np.array(self.priorities, dtype=np.float64)

        probs = probs ** 0.6

        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        batch = [self.buffer[i] for i in indices]

        state, action, reward, next_state = map(np.array, zip(*batch))

        return state, action, reward, next_state

    def __len__(self):

        return len(self.buffer)

class MultiParamActor(nn.Module):

    def __init__(self, state_dim=5, action_dim=3, gain_mins=[1.5, 1.5, 0.05], gain_maxs=[3.5, 5.0, 0.5]):

        super(MultiParamActor, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(state_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, action_dim),

            nn.Tanh()

        )

        self.register_buffer('gain_mins', torch.tensor(gain_mins, dtype=torch.float32))

        self.register_buffer('gain_maxs', torch.tensor(gain_maxs, dtype=torch.float32))

    def forward(self, state):

        raw = self.net(state)

        return (raw + 1.0) / 2.0 * (self.gain_maxs - self.gain_mins) + self.gain_mins

class MultiParamCritic(nn.Module):

    def __init__(self, state_dim=5, action_dim=3):

        super(MultiParamCritic, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(state_dim + action_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 1)

        )

    def forward(self, state, action):

        return self.net(torch.cat([state, action], dim=1))

hp_lam_p  = np.array([1.9476, 3.2, 2.524])

hp_kp_smc = np.array([4.0887, 4.0, 2.0616])

hp_phi_p  = np.array([2.0, 5.0, 4.53])

hp_lam_a  = np.array([16.885, 22.04, 12.21])

hp_ka_smc = np.array([28.47, 10.55, 13.85])

hp_phi_a  = np.array([0.1531, 0.15, 0.15])

hp_kd_p   = np.array([3, 0.8, 1.2])

print("Evaluating Hand-Picked Baseline on S-Curve Trajectory...")

hp_cost, _, hp_log, hp_ref, hp_t = run_quad_simulation(

    hp_lam_p, hp_kp_smc, hp_phi_p, hp_lam_a, hp_ka_smc, hp_phi_a, hp_kd_p,

    active_axis='xyz', traj_type='s_curve', evaluate_reward=False

)

print(f"Hand-Picked Cost Baseline: {hp_cost:.4f}\n")

num_episodes = 400

num_tuning_passes = 3

if num_episodes < 1 or num_tuning_passes < 1:

    raise ValueError('num_episodes and num_tuning_passes must be positive')

batch_size = 32

gamma = 0.95

tau = 0.005

def train_channel_3param(name, index, g_mins, g_maxs, init_params, patience=50,

                         cur_lam_p=None, cur_kp_p=None, cur_phi_p=None):

    if cur_lam_p is None:  cur_lam_p  = hp_lam_p.copy()

    if cur_kp_p is None:   cur_kp_p   = hp_kp_smc.copy()

    if cur_phi_p is None:  cur_phi_p  = hp_phi_p.copy()

    print(f"--- Training {name} (position) [λ, K_smc, δ] ---")

    actor = MultiParamActor(gain_mins=g_mins, gain_maxs=g_maxs)

    critic = MultiParamCritic()

    t_actor = MultiParamActor(gain_mins=g_mins, gain_maxs=g_maxs)

    t_critic = MultiParamCritic()

    t_actor.load_state_dict(actor.state_dict())

    t_critic.load_state_dict(critic.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=5e-5)

    opt_c = optim.Adam(critic.parameters(), lr=5e-5)

    buf = ReplayBuffer()

    noise = OUNoise(3, sigma=0.25, theta=0.15)

    best_cost = float('inf')

    best_params = list(init_params)

    lp0, kpp0, pp0 = cur_lam_p.copy(), cur_kp_p.copy(), cur_phi_p.copy()

    lp0[index], kpp0[index], pp0[index] = init_params

    base_cost, _, _, _, _ = run_quad_simulation(

        lp0, kpp0, pp0, hp_lam_a, hp_ka_smc, hp_phi_a, hp_kd_p,

        active_axis='xyz', traj_type='s_curve', evaluate_reward=False

    )

    best_cost = base_cost

    best_params = list(init_params)

    current_action = np.array(init_params, dtype=np.float32)

    current_cost = base_cost

    prev_cost = base_cost

    print(f"  Baseline cost for this channel: {base_cost:.4f}")

    no_improve_count = 0

    for ep in range(num_episodes):

        cost_trend = (current_cost - prev_cost) / 100.0

        state = np.array([current_action[0], current_action[1], current_action[2],

                          current_cost / 100.0, cost_trend], dtype=np.float32)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():

            action = actor(s_tensor).detach().numpy()[0]

        noise_scale = max(0.2, 1.0 - (ep / (num_episodes * 0.7)))

        action_noisy = np.clip(action + noise.sample() * noise_scale, g_mins, g_maxs)

        lp, kpp, pp = cur_lam_p.copy(), cur_kp_p.copy(), cur_phi_p.copy()

        lp[index]  = action_noisy[0]

        kpp[index] = action_noisy[1]

        pp[index]  = action_noisy[2]

        try:

            cost, reward, _, _, _ = run_quad_simulation(

                lp, kpp, pp, hp_lam_a, hp_ka_smc, hp_phi_a, hp_kd_p,

                active_axis='xyz', traj_type='s_curve', evaluate_reward=True

            )

            if np.isnan(cost) or np.isinf(cost): cost = 1e6

        except Exception:

            cost = 1e6

        prev_cost = current_cost

        current_action = action_noisy.copy()

        current_cost = cost

        cost_trend_next = (current_cost - prev_cost) / 100.0

        next_state = np.array([current_action[0], current_action[1], current_action[2],

                               current_cost / 100.0, cost_trend_next], dtype=np.float32)

        buf.push(state, action_noisy, reward, next_state)

        if cost < best_cost:

            best_cost = cost

            best_params = action_noisy.copy()

            no_improve_count = 0

            print(f"{name} Episode {ep+1} | NEW BEST Cost: {best_cost:.4f}")

        else:

            no_improve_count += 1

        if no_improve_count >= patience:

            print(f"--> {name} converged at Episode {ep+1}.\n")

            break

        if len(buf) >= batch_size:

            b_s, b_a, b_r, b_ns = buf.sample(batch_size)

            st = torch.tensor(b_s, dtype=torch.float32)

            at = torch.tensor(b_a, dtype=torch.float32)

            rt = torch.tensor(b_r, dtype=torch.float32).unsqueeze(1)

            nst = torch.tensor(b_ns, dtype=torch.float32)

            with torch.no_grad():

                target_q = rt + gamma * t_critic(nst, t_actor(nst))

            current_q = critic(st, at)

            c_loss = nn.MSELoss()(current_q, target_q)

            opt_c.zero_grad(); c_loss.backward()

            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)

            opt_c.step()

            a_loss = -critic(st, actor(st)).mean()

            opt_a.zero_grad(); a_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)

            opt_a.step()

            for tp, p in zip(t_actor.parameters(), actor.parameters()): tp.data.copy_(tau*p.data + (1-tau)*tp.data)

            for tp, p in zip(t_critic.parameters(), critic.parameters()): tp.data.copy_(tau*p.data + (1-tau)*tp.data)

    return best_params

best_lam_p  = hp_lam_p.copy()

best_kp_smc = hp_kp_smc.copy()

best_phi_p  = hp_phi_p.copy()

channels = [

    ('X', 0, [1.0, 1.5, 0.1], [3.5, 6.0, 1.0]),

    ('Y', 1, [2.0, 2.5, 0.1], [5.5, 6.0, 1.5]),

    ('Z', 2, [1.5, 1.0, 0.1], [7.0, 7.0, 1.0]),

]

chan_results = {}

for tuning_pass in range(num_tuning_passes):

    print(f"\n{'='*55}")

    print(f"{'='*55}")

    print(f"--- FINE-TUNING PASS {tuning_pass+1} / {num_tuning_passes} ---")

    print(f"{'='*55}")

    for name, idx, g_mins, g_maxs in channels:

        init = [best_lam_p[idx], best_kp_smc[idx], best_phi_p[idx]]

        result = train_channel_3param(

            name, idx, g_mins, g_maxs, init,

            cur_lam_p=best_lam_p.copy(),

            cur_kp_p=best_kp_smc.copy(),

            cur_phi_p=best_phi_p.copy(),

        )

        best_lam_p[idx]  = result[0]

        best_kp_smc[idx] = result[1]

        best_phi_p[idx]  = result[2]

        chan_results[f"{name}_pass{tuning_pass+1}"] = result.copy()

        cur_cost, _, _, _, _ = run_quad_simulation(

            best_lam_p, best_kp_smc, best_phi_p,

            hp_lam_a, hp_ka_smc, hp_phi_a,

            hp_kd_p, active_axis='xyz', traj_type='s_curve', evaluate_reward=False

        )

        print(f"  -> Full-system cost after {name}: {cur_cost:.4f}\n")

final_lam_p = best_lam_p.copy()

final_kp_p  = best_kp_smc.copy()

final_phi_p = best_phi_p.copy()

def fmt(arr):

    return ", ".join(f"{v:.8f}" for v in arr)

eval_cost, _, eval_log, eval_ref, t_log = run_quad_simulation(

    final_lam_p, final_kp_p, final_phi_p, hp_lam_a, hp_ka_smc, hp_phi_a, hp_kd_p,

    active_axis='xyz', traj_type='s_curve', evaluate_reward=False

)

improvement = (hp_cost - eval_cost) / hp_cost * 100.0

print("\n" + "=" * 65)

print("--- TRAINING SUMMARY ---")

print("=" * 65)

print(f"Baseline cost:        {hp_cost:.4f}")

print(f"Final DDPG cost:      {eval_cost:.4f}")

print(f"Improvement:          {improvement:+.2f}%")

print("-" * 65)

print(f"{'Axis':<5} {'λ (lam_p)':>14} {'K_smc':>14} {'δ (phi_p)':>14}")

print("-" * 65)

axis_labels = ['X', 'Y', 'Z']

for i in range(3):

    bl = [hp_lam_p[i], hp_kp_smc[i], hp_phi_p[i]]

    fl = [final_lam_p[i], final_kp_p[i], final_phi_p[i]]

    print(f"{axis_labels[i]:<5} {'BASE':>14} {'BASE':>14} {'BASE':>14}")

    print(f"{'':<5} {fl[0]:>14.8f} {fl[1]:>14.8f} {fl[2]:>14.8f}")

print("-" * 65)

print("\n--- COPY-PASTE GAINS ---")

print("lam_p  = np.array([" + fmt(final_lam_p) + "])")

print("kp_smc = np.array([" + fmt(final_kp_p) + "])")

print("phi_p  = np.array([" + fmt(final_phi_p) + "])")

print("=" * 65)

fig0 = plt.figure(figsize=(10, 8))

ax0 = fig0.add_subplot(111, projection='3d')

ax0.plot(eval_ref[:, 0], eval_ref[:, 1], eval_ref[:, 2], 'k--', label='Reference', linewidth=2.0, alpha=0.7)

ax0.plot(hp_log[:, 0], hp_log[:, 1], hp_log[:, 2], 'b-', label=f'Baseline ({hp_cost:.2f})', linewidth=1.2)

ax0.plot(eval_log[:, 0], eval_log[:, 1], eval_log[:, 2], 'r-', label=f'DDPG ({eval_cost:.2f})', linewidth=1.5)

ax0.set_xlabel('X (m)')

ax0.set_ylabel('Y (m)')

ax0.set_zlabel('Z (m)')

ax0.set_title(f'3D S-Curve Trajectory (Improvement: {improvement:+.1f}%)')

ax0.legend(loc='upper left', fontsize=9)

ax0.view_init(elev=25, azim=-55)

fig0.tight_layout()

fig1, ax1 = plt.subplots(figsize=(9, 7))

ax1.plot(eval_ref[:, 0], eval_ref[:, 1], 'k--', label='Reference S-Curve', linewidth=2.0, alpha=0.7)

ax1.plot(hp_log[:, 0], hp_log[:, 1], 'b-', label=f'Baseline ({hp_cost:.2f})', linewidth=1.2)

ax1.plot(eval_log[:, 0], eval_log[:, 1], 'r-', label=f'DDPG ({eval_cost:.2f})', linewidth=1.5)

ax1.set_xlabel('X Position (m)')

ax1.set_ylabel('Y Position (m)')

ax1.set_title(f'S-Curve Trajectory: Top-Down XY View (Improvement: {improvement:+.1f}%)')

ax1.legend(loc='upper left', fontsize=9)

ax1.grid(True)

ax1.set_aspect('equal')

fig1.tight_layout()

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8))

fig2.suptitle(f'S-Curve Trajectory - Position vs Time (Cost: {eval_cost:.2f}, {improvement:+.1f}%)', fontsize=12)

pos_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']

for i in range(3):

    axes2[i].plot(t_log, eval_ref[:, i], 'k--', label='Reference', alpha=0.7)

    axes2[i].plot(t_log, hp_log[:, i], 'b-', label=f'Baseline', linewidth=1.2)

    axes2[i].plot(t_log, eval_log[:, i], 'r-', label='DDPG-Optimized', linewidth=1.5)

    axes2[i].set_ylabel(pos_labels[i])

    axes2[i].grid(True)

    if i == 0: axes2[i].legend(loc='upper right', fontsize=8)

axes2[2].set_xlabel('Time (s)')

fig2.tight_layout()

fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7))

fig3.suptitle('Per-Axis Tracking Error (DDPG vs Baseline)', fontsize=12)

err_labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)']

for i in range(3):

    err_ddpg = eval_log[:, i] - eval_ref[:, i]

    err_hp   = hp_log[:, i] - hp_ref[:, i]

    axes3[i].plot(t_log, err_hp, 'b-', label='Baseline Error', linewidth=1.0, alpha=0.7)

    axes3[i].plot(t_log, err_ddpg, 'r-', label='DDPG Error', linewidth=1.2)

    axes3[i].axhline(0.0, color='k', linestyle=':', linewidth=0.8)

    axes3[i].set_ylabel(err_labels[i])

    axes3[i].grid(True)

    if i == 0: axes3[i].legend(loc='upper right', fontsize=8)

axes3[2].set_xlabel('Time (s)')

fig3.tight_layout()

plt.show()

