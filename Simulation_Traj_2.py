"""
Quadcopter: SMC vs PD Position Control on 3D S-Curve Trajectory
----------------------------------------------------------------
Simulation engine is a VERBATIM copy from ddpg_smc_s.py.
The ONLY difference between the two controllers is the position control law.
User enters SMC gains and PD gains below.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

plt.rcParams.update({
    'font.size': 10,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})




# ============================================================
#  USER GAINS  —  EDIT THESE
# ============================================================
# --- SMC #1 Position Gains (per axis X, Y, Z) ---
SMC1_LAM_P   = np.array([3.50000000, 2.00000000, 6.22547666])
SMC1_KP_SMC  = np.array([1.88694007, 3.87125618, 2.64881070])
SMC1_PHI_P   = np.array([0.60964995, 0.95312983, 0.10000000])

# --- SMC #2 Position Gains (per axis X, Y, Z) ---
SMC2_LAM_P   = np.array([3, 4, 5])
SMC2_KP_SMC  = np.array([4, 5.64, 4.0])
SMC2_PHI_P   = np.array([1, 1.2, 1.0])

# --- PD Position Gains (per axis X, Y, Z) ---
PD_KP       = np.array([6.0, 6.0, 8.0])
PD_KD       = np.array([1.0, 1.0, 0.5])

# --- Inner Attitude SMC Gains (shared, fixed) ---
ATT_LAM_A   = np.array([16.885, 22.04, 12.21])
ATT_KA_SMC  = np.array([28.47, 10.55, 13.85])
ATT_PHI_A   = np.array([0.1531, 0.15, 0.15])
# ============================================================


# ============================================================
#  SIMULATION ENGINE  —  VERBATIM COPY FROM ddpg_smc_s.py
#  EXCEPT: position control law can switch between SMC and PD
# ============================================================
def run_quad_position(mode, lam_p, kp_smc, phi_p, kp_pd, kd_pd,
                       lam_a, ka_smc, phi_a_in):
    """
    mode: 'smc' or 'pd'
    Returns: cost, log(N,12), ref_log(N,3), t_log(N), smc_log_pos(N,6), smc_log_att(N,6)
    smc_log: col 0-2 = s,  col 3-5 = s_dot
    """
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
    smc_log_pos = np.zeros((steps, 6))
    smc_log_att = np.zeros((steps, 6))
    s_pos_prev = np.zeros(3)
    s_att_prev = np.zeros(3)
    pos_d_filtered = np.array([0.0, 0.0, 0.0])
    psi_d_target = 0.0
    pos_d_prev   = np.array([0.0, 0.0, 0.0])
    vel_d_prev   = np.zeros(3)

    def step_reference(t):
        if True:  # s_curve — VERBATIM from ddpg_smc_s.py
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

    def sat(s, phi_val):
        return np.tanh(s / np.maximum(phi_val, 1e-3))

    control_effort_total = 0.0
    error_integral = 0.0

    for k in range(steps):
        t = k * dt
        pos   = state[0:3]
        vel   = state[3:6]
        att   = state[6:9]
        omega = state[9:12]

        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return 1e6, log, ref_log, t_log, smc_log_pos, smc_log_att

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

        s_pos = e_dot + lam_p * e

        # --- POSITION CONTROL LAW (only difference between SMC and PD) ---
        if mode == 'smc':
            acc_cmd = acc_d - kp_smc * sat(s_pos, phi_p)
        else:  # 'pd'
            acc_cmd = acc_d - kp_pd * e - kd_pd * e_dot

        az = acc_cmd[2] + g
        phi_d   = (acc_cmd[0] * np.sin(psi_d) - acc_cmd[1] * np.cos(psi_d)) / g
        theta_d = (acc_cmd[0] * np.cos(psi_d) + acc_cmd[1] * np.sin(psi_d)) / g
        phi_d   = np.clip(phi_d,   -0.6, 0.6)
        theta_d = np.clip(theta_d, -0.6, 0.6)
        cos_tilt = max(0.5, np.cos(phi_d) * np.cos(theta_d))
        T_max = 2.0 * m * g
        T = np.clip(m * az / cos_tilt, 0.0, T_max)

        att_d = np.array([phi_d, theta_d, psi_d])
        att_e     = att - att_d
        att_e_dot = omega - np.zeros(3)
        att_s     = att_e_dot + lam_a * att_e

        if k == 0:
            s_pos_dot = np.zeros(3)
            s_att_dot = np.zeros(3)
        else:
            s_pos_dot = (s_pos - s_pos_prev) / dt
            s_att_dot = (att_s - s_att_prev) / dt
        s_pos_prev = s_pos.copy()
        s_att_prev = att_s.copy()

        smc_log_pos[k, 0:3] = s_pos
        smc_log_pos[k, 3:6] = s_pos_dot
        smc_log_att[k, 0:3] = att_s
        smc_log_att[k, 3:6] = s_att_dot

        I_diag = np.array([Ixx, Iyy, Izz])
        tau = I_diag * (-ka_smc * sat(att_s, phi_a_in))

        u = np.linalg.solve(mix, np.array([T, tau[0], tau[1], tau[2]]))
        u = np.clip(u, 0.0, 2.0 * m * g / 4.0)

        error_integral += np.sum(np.abs(e)) * dt
        control_effort_total += np.sum(np.abs(np.diff(u, prepend=u[0]))) * dt

        R = np.array([
            [np.cos(att[1])*np.cos(att[2]), np.sin(att[0])*np.sin(att[1])*np.cos(att[2]) - np.cos(att[0])*np.sin(att[2]), np.cos(att[0])*np.sin(att[1])*np.cos(att[2]) + np.sin(att[0])*np.sin(att[2])],
            [np.cos(att[1])*np.sin(att[2]), np.sin(att[0])*np.sin(att[1])*np.sin(att[2]) + np.cos(att[0])*np.cos(att[2]), np.cos(att[0])*np.sin(att[1])*np.sin(att[2]) - np.sin(att[0])*np.cos(att[2])],
            [-np.sin(att[1]),               np.sin(att[0])*np.cos(att[1]),                                           np.cos(att[0])*np.cos(att[1])]
        ])
        thrust_world = R @ np.array([0, 0, np.sum(u)])
        acc_lin = thrust_world / m - np.array([0, 0, g])
        p, q, r = omega
        acc_ang = np.array([
            (tau[0] + (Iyy - Izz) * q * r) / Ixx,
            (tau[1] + (Izz - Ixx) * p * r) / Iyy,
            (tau[2] + (Ixx - Iyy) * p * q) / Izz
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
    return cost, log, ref_log, t_log, smc_log_pos, smc_log_att


# ============================================================
#  RUN ALL THREE SIMULATIONS
# ============================================================
print("=" * 65)
print("  SMC1 vs SMC2 vs PD  --  3D S-Curve Trajectory")
print("  (Sim engine: VERBATIM from ddpg_smc_s.py)")
print("=" * 65)
print(f"\nSMC1 Gains : lam_p={SMC1_LAM_P},  K_smc={SMC1_KP_SMC},  phi_p={SMC1_PHI_P}")
print(f"SMC2 Gains : lam_p={SMC2_LAM_P},  K_smc={SMC2_KP_SMC},  phi_p={SMC2_PHI_P}")
print(f"PD  Gains  : kp={PD_KP},  kd={PD_KD}")
print(f"Attitude   : lam_a={ATT_LAM_A},  K_a={ATT_KA_SMC},  phi_a={ATT_PHI_A}")

print("\nRunning SMC #1 simulation...")
smc1_cost, smc1_log, ref_log, t_log, smc1_pos_log, smc1_att_log = run_quad_position(
    mode='smc',
    lam_p=SMC1_LAM_P, kp_smc=SMC1_KP_SMC, phi_p=SMC1_PHI_P,
    kp_pd=np.zeros(3), kd_pd=np.zeros(3),
    lam_a=ATT_LAM_A, ka_smc=ATT_KA_SMC, phi_a_in=ATT_PHI_A,
)
print(f"  SMC #1 cost: {smc1_cost:.4f}")

print("Running SMC #2 simulation...")
smc2_cost, smc2_log, _, _, smc2_pos_log, smc2_att_log = run_quad_position(
    mode='smc',
    lam_p=SMC2_LAM_P, kp_smc=SMC2_KP_SMC, phi_p=SMC2_PHI_P,
    kp_pd=np.zeros(3), kd_pd=np.zeros(3),
    lam_a=ATT_LAM_A, ka_smc=ATT_KA_SMC, phi_a_in=ATT_PHI_A,
)
print(f"  SMC #2 cost: {smc2_cost:.4f}")

print("Running PD simulation...")
pd_cost, pd_log, _, _, _, _ = run_quad_position(
    mode='pd',
    lam_p=np.zeros(3), kp_smc=np.zeros(3), phi_p=np.ones(3),
    kp_pd=PD_KP, kd_pd=PD_KD,
    lam_a=ATT_LAM_A, ka_smc=ATT_KA_SMC, phi_a_in=ATT_PHI_A,
)
print(f"  PD    cost: {pd_cost:.4f}")

# --- Extract positions ---
smc1_pos = smc1_log[:, 0:3]
smc2_pos = smc2_log[:, 0:3]
pd_pos   = pd_log[:, 0:3]
ref_pos  = ref_log

smc1_err = smc1_pos - ref_pos
smc2_err = smc2_pos - ref_pos
pd_err   = pd_pos - ref_pos

dt_step = t_log[1] - t_log[0]
smc1_axis = np.sum(np.abs(smc1_err), axis=0) * dt_step
smc2_axis = np.sum(np.abs(smc2_err), axis=0) * dt_step
pd_axis   = np.sum(np.abs(pd_err),   axis=0) * dt_step

# --- Cost Table ---
all_costs = {"SMC1": smc1_cost, "SMC2": smc2_cost, "PD": pd_cost}
winner = min(all_costs, key=all_costs.get)

print("\n" + "=" * 65)
print("  COST COMPARISON")
print("=" * 65)
print(f"{'Controller':<12} {'Total Cost':>12} {'X Cost':>12} {'Y Cost':>12} {'Z Cost':>12}")
print("-" * 65)
print(f"{'RL-SMC':<12} {smc1_cost:>12.4f} {smc1_axis[0]:>12.4f} {smc1_axis[1]:>12.4f} {smc1_axis[2]:>12.4f}")
print(f"{'Man. SMC':<12} {smc2_cost:>12.4f} {smc2_axis[0]:>12.4f} {smc2_axis[1]:>12.4f} {smc2_axis[2]:>12.4f}")
print(f"{'PD':<12}    {pd_cost:>12.4f} {pd_axis[0]:>12.4f} {pd_axis[1]:>12.4f} {pd_axis[2]:>12.4f}")
print("-" * 65)
print(f"Winner: {winner}")
print("=" * 65)

# ---- Full position metrics: RMSE / IAE / MAE ----
def print_metrics(name, log, ref, dt=dt_step):
    err_pos = log[:, 0:3] - ref[:, 0:3]

    rmse_xyz = np.sqrt(np.mean(err_pos ** 2, axis=0))
    rmse_3d  = np.sqrt(np.mean(np.sum(err_pos**2, axis=1)))

    pos_norm = np.sqrt(np.sum(err_pos**2, axis=1))

    iae_xyz = np.sum(np.abs(err_pos), axis=0) * dt
    iae_3d  = np.sum(pos_norm) * dt

    mae_xyz = np.max(np.abs(err_pos), axis=0)
    mae_3d  = np.max(pos_norm)

    print(f"\n  Controller: {name}")
    print(f"  {'Metric':<10} {'X':>12} {'Y':>12} {'Z':>12} {'3D':>12}")
    print(f"  {'RMSE (m)':<10} {rmse_xyz[0]:>12.5f} {rmse_xyz[1]:>12.5f} {rmse_xyz[2]:>12.5f} {rmse_3d:>12.5f}")
    print(f"  {'IAE (m*s)':<10} {iae_xyz[0]:>12.5f} {iae_xyz[1]:>12.5f} {iae_xyz[2]:>12.5f} {iae_3d:>12.5f}")
    print(f"  {'MAE (m)':<10} {mae_xyz[0]:>12.5f} {mae_xyz[1]:>12.5f} {mae_xyz[2]:>12.5f} {mae_3d:>12.5f}")

print("\n" + "=" * 62)
print("  POSITION METRICS (full trajectory)")
print("=" * 62)
print_metrics("RL-SMC",              smc1_log, ref_pos)
print_metrics("Manually Tuned SMC",  smc2_log, ref_pos)
print_metrics("PD",                  pd_log,   ref_pos)
print("=" * 62)

# ============================================================
#  PLOTS  (4 controllers: SMC1 red, SMC2 blue, PD green, ref black)
# ============================================================

# --- Plot 0: 3D Trajectory ---
fig0 = plt.figure(figsize=(11, 8))
ax0 = fig0.add_subplot(111, projection='3d')
ax0.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 'k--', label='Reference',     linewidth=2.0, alpha=0.7)
ax0.plot(smc1_pos[:, 0], smc1_pos[:, 1], smc1_pos[:, 2], 'r-', label='RL-SMC', linewidth=2)
ax0.plot(smc2_pos[:, 0], smc2_pos[:, 1], smc2_pos[:, 2], 'b-', label='Manually Tuned SMC', linewidth=2)
ax0.plot(pd_pos[:, 0],   pd_pos[:, 1],   pd_pos[:, 2],   'g-', label='PD',    linewidth=1.3)
ax0.set_xlabel('X Distance (m)'); ax0.set_ylabel('Y Distance (m)'); ax0.set_zlabel('Z Distance (m)')
ax0.set_title(f'3D S-Curve Trajectory  —  Winner: {winner}')
ax0.legend(fontsize=10)
ax0.view_init(elev=25, azim=-55)
fig0.tight_layout()

# --- Plot 1: XY Top-Down ---
fig1, ax1 = plt.subplots(figsize=(9, 7))
ax1.plot(ref_pos[:, 0], ref_pos[:, 1], 'k--', label='Reference',     linewidth=2.0, alpha=0.7)
ax1.plot(smc1_pos[:, 0], smc1_pos[:, 1], 'r-', label='RL-SMC', linewidth=2)
ax1.plot(smc2_pos[:, 0], smc2_pos[:, 1], 'b-', label='Manually Tuned SMC', linewidth=2)
ax1.plot(pd_pos[:, 0],   pd_pos[:, 1],   'g-', label='PD',    linewidth=1.3)
ax1.set_xlabel('X Distance (m)')
ax1.set_ylabel('Y Distance (m)')
ax1.set_title(f'Top-Down XY View  —  Winner: {winner}')
ax1.legend(fontsize=10)
ax1.grid(True)
ax1.set_aspect('equal')
fig1.tight_layout()

# --- Plot 2: Position vs Time ---
fig2, axes2 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
for i, ax in enumerate(axes2):
    ax.plot(t_log, ref_pos[:, i],     'k--', label='Reference',  linewidth=2.5, alpha=0.7)
    ax.plot(t_log, smc1_pos[:, i],    'r-',  label='RL-SMC',         linewidth=2.5)
    ax.plot(t_log, smc2_pos[:, i],    'b-',  label='Manually Tuned SMC', linewidth=2.5)
    ax.plot(t_log, pd_pos[:, i],      'g-',  label=f'PD',        linewidth=2.5)
    ax.set_ylabel(axis_labels[i])
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_title('Distance vs Time  —  SMC1, SMC2, PD on 3D S-Curve')
        ax.legend(loc='upper right', fontsize=10)
axes2[-1].set_xlabel('Time (s)')
fig2.tight_layout()

# --- Plot 3: Per-Axis Tracking Error ---
fig3, axes3 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for i, ax in enumerate(axes3):
    ax.plot(t_log, smc1_err[:, i], 'r-', label='RL-SMC', linewidth=2.5)
    ax.plot(t_log, smc2_err[:, i], 'b-', label='Manually Tuned SMC', linewidth=2.5)
    ax.plot(t_log, pd_err[:, i],   'g-', label=f'PD', linewidth=2.5, alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel(f'Error {axis_labels[i]}')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_title('Tracking Error per Axis  —  SMC1, SMC2, PD')
        ax.legend(loc='upper right', fontsize=10)
axes3[-1].set_xlabel('Time (s)')
fig3.tight_layout()

# --- Plot 4: Phase Plane s vs s_dot (RL-SMC top 2 rows, HT-SMC bottom 2 rows) ---
pos_names  = [r'$X$',  r'$Y$',  r'$Z$']
att_names  = [r'$\phi$', r'$\theta$', r'$\psi$']
pos_s      = [r'$s_{X}$',   r'$s_{Y}$',   r'$s_{Z}$']
pos_sdot   = [r'$\dot{s}_{X}$', r'$\dot{s}_{Y}$', r'$\dot{s}_{Z}$']
att_s      = [r'$s_{\phi}$',   r'$s_{\theta}$',   r'$s_{\psi}$']
att_sdot   = [r'$\dot{s}_{\phi}$', r'$\dot{s}_{\theta}$', r'$\dot{s}_{\psi}$']
LW = 1.8
LABEL_KW = dict(fontsize=13, fontweight='bold')

fig4, axes4 = plt.subplots(4, 3, figsize=(14, 14))
fig4.suptitle(r'Phase Plane: $s$ vs $\dot{s}$  —  RL-SMC vs Manually Tuned SMC', fontsize=14)

for i in range(3):
    axes4[0, i].plot(smc1_pos_log[:, i], smc1_pos_log[:, 3 + i], 'r', lw=LW)
    axes4[0, i].scatter(smc1_pos_log[0, i], smc1_pos_log[0, 3 + i], color='green', s=40, zorder=5)
    axes4[0, i].scatter(smc1_pos_log[-1, i], smc1_pos_log[-1, 3 + i], color='black', s=40, zorder=5)
    axes4[0, i].axhline(0, color='gray', lw=2.5, ls='--')
    axes4[0, i].axvline(0, color='gray', lw=2.5, ls='--')
    axes4[0, i].set_title(f'RL-SMC pos: {pos_names[i]}')
    axes4[0, i].set_xlabel(pos_s[i], **LABEL_KW)
    axes4[0, i].set_ylabel(pos_sdot[i], **LABEL_KW)
    axes4[0, i].grid(True)

    axes4[1, i].plot(smc1_att_log[:, i], smc1_att_log[:, 3 + i], 'r', lw=LW)
    axes4[1, i].scatter(smc1_att_log[0, i], smc1_att_log[0, 3 + i], color='green', s=40, zorder=5)
    axes4[1, i].scatter(smc1_att_log[-1, i], smc1_att_log[-1, 3 + i], color='black', s=40, zorder=5)
    axes4[1, i].axhline(0, color='gray', lw=2.5, ls='--')
    axes4[1, i].axvline(0, color='gray', lw=2.5, ls='--')
    axes4[1, i].set_title(f'RL-SMC att: {att_names[i]}')
    axes4[1, i].set_xlabel(att_s[i], **LABEL_KW)
    axes4[1, i].set_ylabel(att_sdot[i], **LABEL_KW)
    axes4[1, i].grid(True)

    axes4[2, i].plot(smc2_pos_log[:, i], smc2_pos_log[:, 3 + i], 'b', lw=LW)
    axes4[2, i].scatter(smc2_pos_log[0, i], smc2_pos_log[0, 3 + i], color='green', s=40, zorder=5)
    axes4[2, i].scatter(smc2_pos_log[-1, i], smc2_pos_log[-1, 3 + i], color='black', s=40, zorder=5)
    axes4[2, i].axhline(0, color='gray', lw=2.5, ls='--')
    axes4[2, i].axvline(0, color='gray', lw=2.5, ls='--')
    axes4[2, i].set_title(f'Manually Tuned SMC pos: {pos_names[i]}')
    axes4[2, i].set_xlabel(pos_s[i], **LABEL_KW)
    axes4[2, i].set_ylabel(pos_sdot[i], **LABEL_KW)
    axes4[2, i].grid(True)

    axes4[3, i].plot(smc2_att_log[:, i], smc2_att_log[:, 3 + i], 'b', lw=LW)
    axes4[3, i].scatter(smc2_att_log[0, i], smc2_att_log[0, 3 + i], color='green', s=40, zorder=5)
    axes4[3, i].scatter(smc2_att_log[-1, i], smc2_att_log[-1, 3 + i], color='black', s=40, zorder=5)
    axes4[3, i].axhline(0, color='gray', lw=2.5, ls='--')
    axes4[3, i].axvline(0, color='gray', lw=2.5, ls='--')
    axes4[3, i].set_title(f'Manually Tuned SMC att: {att_names[i]}')
    axes4[3, i].set_xlabel(att_s[i], **LABEL_KW)
    axes4[3, i].set_ylabel(att_sdot[i], **LABEL_KW)
    axes4[3, i].grid(True)

fig4.tight_layout()

# ---- 3-D trajectory animation --------------------------------------------
# The animation uses the trajectories already returned by the simulations.
def save_scurve_animation():
    out_dir = Path(__file__).resolve().parent / "animations"
    out_dir.mkdir(exist_ok=True)
    paths = [ref_pos[:, :3], smc1_pos, smc2_pos, pd_pos]
    labels = ["Reference", "RL-SMC", "Manual SMC", "PD"]
    colors = ["black", "red", "blue", "green"]
    styles = ["--", "-", "--", ":"]
    nframes = min(240, len(t_log))
    frame_idx = np.linspace(0, len(t_log) - 1, nframes).astype(int)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_xyz = np.vstack(paths)
    lo = all_xyz.min(axis=0) - 0.1
    hi = all_xyz.max(axis=0) + 0.1
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Trajectory 2: S-curve-flight animation")
    lines = [ax.plot([], [], [], color=c, ls=ls, lw=2, label=lab)[0]
             for c, ls, lab in zip(colors, styles, labels)]
    points = [ax.plot([], [], [], marker="o", color=c, ms=5)[0] for c in colors]
    ax.legend(loc="upper left", fontsize=8)

    def update(frame):
        idx = frame_idx[frame]
        for line, point, path in zip(lines, points, paths):
            line.set_data(path[:idx + 1, 0], path[:idx + 1, 1])
            line.set_3d_properties(path[:idx + 1, 2])
            point.set_data([path[idx, 0]], [path[idx, 1]])
            point.set_3d_properties([path[idx, 2]])
        ax.set_title(f"Trajectory 2: S-curve flight  (t = {t_log[idx]:.2f} s)")
        return lines + points

    anim = FuncAnimation(fig, update, frames=nframes, interval=45, blit=False)
    gif_path = out_dir / "trajectory_2_scurve.gif"
    anim.save(gif_path, writer=PillowWriter(fps=22))
    print(f"Saved 3-D animation: {gif_path}")
    try:
        mp4_path = out_dir / "trajectory_2_scurve.mp4"
        anim.save(mp4_path, writer=FFMpegWriter(fps=22, bitrate=1800))
        print(f"Saved 3-D animation: {mp4_path}")
    except Exception as exc:
        print(f"MP4 not saved (FFmpeg unavailable): {exc}")
    plt.close(fig)

save_scurve_animation()

plt.show()
