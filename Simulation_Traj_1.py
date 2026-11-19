"""
Combined SMC1 + SMC2 + PID Quadrotor Simulation
================================================
Three controllers tracking the same square trajectory:
  - SMC1: original gains (moderate switching, larger boundary layer)
  - SMC2: different gains   (lower switching, medium boundary layer)
  - PID:  cascaded position/velocity/yaw PID

Plots overlay all three for direct comparison.
"""
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ===================== REFERENCE TRAJECTORY =====================
def square_reference(t):
    side    = 2.0
    T       = 4.0
    z_hover = 2.0
    if t < T:
        x, y, z = 0.0, 0.0, z_hover * (t / T)
    elif t < 2*T:
        s = (t - T) / T
        x, y, z = side * s, 0.0, z_hover
    elif t < 3*T:
        s = (t - 2*T) / T
        x, y, z = side, side * s, z_hover
    elif t < 4*T:
        s = (t - 3*T) / T
        x, y, z = side * (1 - s), side, z_hover
    elif t < 5*T:
        s = (t - 4*T) / T
        x, y, z = 0.0, side * (1 - s), z_hover
    else:
        x, y, z = 0.0, 0.0, z_hover
    return np.array([x, y, z])

# ===================== SIMULATION SETTINGS =====================
dt    = 0.005
T_end = 22.0
steps = int(T_end / dt)
alpha = 0.04

# ===================== QUADROTOR PHYSICAL PARAMETERS =====================
m   = 0.08
g   = 9.81
Ixx = 7.2e-5
Iyy = 6.8e-5
Izz = 1.53e-4
l   = 0.119 / np.sqrt(2)
b   = 2.84e-6
d   = 1.46e-10
mix = np.array([
    [1,        1,         1,         1     ],
    [l,       -l,        -l,         l     ],
    [l,        l,        -l,        -l     ],
    [-d/b,     d/b,      -d/b,       d/b   ]
])

# ===================== HELPER FUNCTIONS =====================
def sat(s, phi):
    return np.tanh(s / phi)

def rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cps,  sps  = np.cos(psi), np.sin(psi)
    return np.array([
        [cth*cps, sphi*sth*cps - cphi*sps, cphi*sth*cps + sphi*sps],
        [cth*sps, sphi*sth*sps + cphi*cps, cphi*sth*sps - sphi*cps],
        [-sth,    sphi*cth,                cphi*cth]
    ])

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def allocate_motors(thrust, tau, u_max):
    """Allocate a commanded wrench while respecting individual rotor limits."""
    collective = np.clip(thrust / 4.0, 0.0, u_max)
    differential = np.linalg.solve(mix, np.array([0.0, tau[0], tau[1], 0.0]))
    scale = 1.0
    for value in differential:
        if value > 0.0:
            scale = min(scale, (u_max - collective) / value)
        elif value < 0.0:
            scale = min(scale, -collective / value)
    u = collective + max(0.0, scale) * differential
    yaw = np.linalg.solve(mix, np.array([0.0, 0.0, 0.0, tau[2]]))
    yaw_scale = 1.0
    for j in range(4):
        value = yaw[j]
        if value > 0.0:
            yaw_scale = min(yaw_scale, (u_max - u[j]) / value)
        elif value < 0.0:
            yaw_scale = min(yaw_scale, -u[j] / value)
    u = np.clip(u + max(0.0, yaw_scale) * yaw, 0.0, u_max)
    return u, mix @ u

def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

# ===================== SMC CONTROLLERS (gain-parameterized) =====================
def smc_position(pos, vel, pos_d, vel_d, acc_d, psi_d,
                 lam_p, kp_smc, phi_p):
    e       = pos - pos_d
    e_dot = vel - vel_d
    s       = e_dot + lam_p * e
    # Complete the equivalent-control term for s = e_dot + lambda*e.
    # This velocity feedback damps corner transitions and reduces overshoot.
    acc_cmd = acc_d - lam_p * e_dot - kp_smc * sat(s, phi_p)
    az = acc_cmd[2] + g
    phi_d   = (acc_cmd[0] * np.sin(psi_d) - acc_cmd[1] * np.cos(psi_d)) / g
    theta_d = (acc_cmd[0] * np.cos(psi_d) + acc_cmd[1] * np.sin(psi_d)) / g
    phi_d   = np.clip(phi_d,   -0.6, 0.6)
    theta_d = np.clip(theta_d, -0.6, 0.6)
    cos_tilt = max(0.5, np.cos(phi_d) * np.cos(theta_d))
    T_max = 2.0 * m * g
    T = np.clip(m * az / cos_tilt, 0.0, T_max)
    return T, phi_d, theta_d, s

def smc_attitude(att, omega, att_d, omega_d,
                 lam_a, ka_smc, phi_a):
    e       = att - att_d
    e_dot = omega - omega_d
    s       = e_dot + lam_a * e
    I_diag = np.array([Ixx, Iyy, Izz])
    u   = -ka_smc * sat(s, phi_a)
    tau = I_diag * u
    return tau, s

# ===================== SMC SIMULATOR (single function, different gains) =====================
def run_smc(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a):
    state   = np.zeros(12)
    log     = np.zeros((steps, 12))
    ref_log = np.zeros((steps, 3))
    att_ref_log = np.zeros((steps, 3))
    t_log   = np.zeros(steps)
    smc_log_pos = np.zeros((steps, 6))
    smc_log_att = np.zeros((steps, 6))

    psi_d        = 0.0
    pos_d_prev = square_reference(0.0)
    vel_d_prev = np.zeros(3)
    acc_d_prev = np.zeros(3)
    pos_d_filt = square_reference(0.0)
    s_pos_prev = np.zeros(3)
    s_att_prev = np.zeros(3)

    for k in range(steps):
        t = k * dt
        pos   = state[0:3]
        vel   = state[3:6]
        att   = state[6:9]
        omega = state[9:12]

        pos_d_raw  = square_reference(t)
        pos_d_filt = alpha * pos_d_raw + (1.0 - alpha) * pos_d_filt
        pos_d      = pos_d_filt

        if k == 0:
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
        else:
            vel_d = (pos_d - pos_d_prev) / dt
            acc_d = (vel_d - vel_d_prev) / dt
            vel_d = 0.5 * vel_d + 0.5 * ((pos_d - pos_d_prev) / dt)
            acc_d = np.clip(acc_d, -10.0, 10.0)
            acc_d = 0.20 * acc_d + 0.80 * acc_d_prev

        pos_d_prev = pos_d
        vel_d_prev = vel_d
        acc_d_prev = acc_d.copy()

        T, phi_d, theta_d, s_pos = smc_position(
            pos, vel, pos_d, vel_d, acc_d, psi_d,
            lam_p, kp_smc, phi_p)
        att_d = np.array([phi_d, theta_d, psi_d])
        tau, s_att = smc_attitude(
            att, omega, att_d, np.zeros(3),
            lam_a, ka_smc, phi_a)

        if k == 0:
            s_pos_dot = np.zeros(3)
            s_att_dot = np.zeros(3)
        else:
            s_pos_dot = (s_pos - s_pos_prev) / dt
            s_att_dot = (s_att - s_att_prev) / dt
        s_pos_prev = s_pos.copy()
        s_att_prev = s_att.copy()

        smc_log_pos[k, 0:3] = s_pos
        smc_log_pos[k, 3:6] = s_pos_dot
        smc_log_att[k, 0:3] = s_att
        smc_log_att[k, 3:6] = s_att_dot

        u, realized_wrench = allocate_motors(T, tau, 2.0 * m * g / 4.0)
        # Use the torque delivered by the saturated rotors in the dynamics.
        tau = realized_wrench[1:4]

        R = rotation_matrix(*att)
        thrust_world = R @ np.array([0, 0, np.sum(u)])
        acc_lin = thrust_world / m - np.array([0, 0, g])

        p, q, r = omega
        tau_phi, tau_theta, tau_psi = tau
        acc_ang = np.array([
            (tau_phi   + (Iyy - Izz) * q * r) / Ixx,
            (tau_theta + (Izz - Ixx) * p * r) / Iyy,
            (tau_psi   + (Ixx - Iyy) * p * q) / Izz
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

        log[k]        = state
        ref_log[k]    = pos_d
        att_ref_log[k] = att_d
        t_log[k]      = t

    return log, ref_log, att_ref_log, t_log, smc_log_pos, smc_log_att

# ===================== RUN SMC1 (original gains) =====================








print("Running SMC1 simulation ...")
SMC1_GAINS = dict(
    # lam_p  = np.array([1.57620340, 2.64245938, 6.83352996]),   
    # kp_smc = np.array([3.26710984, 3.65208615, 4.79980070]),  # from the ddpg rl blcok
    # phi_p  = np.array([0.91794046, 1.00000000, 0.55570719]),
   
    # lam_a  = np.array([9.20000000, 9.70000000, 4.61273596]),
    # ka_smc = np.array([13.36565,        14.731565,        9.946545]),
    # phi_a  = np.array([0.50000000, 0.50000000, 0.18926444]),

    lam_p  = np.array([1.190876071431711, 0.94239100808200904, 1.0533619091897717]),
    kp_smc = np.array([1.5, 2.4530474878721673, 5.1241980227168149]),
    phi_p  = np.array([0.25799481673966668, 0.50148808299166492, 0.10600450438946374]),
    lam_a  = np.array([7.727775775076875, 8.9703460569480917, 3]),
    ka_smc = np.array([13, 13, 10.880039247316759]),
    phi_a  = np.array([0.45620099580035844, 0.10000000000000001, 0.25214252584880992]),








)




log_smc1, ref_smc1, att_ref_smc1, t_log, smc_pos1, smc_att1 = run_smc(**SMC1_GAINS)

# ===================== RUN SMC2 (different gains) =====================
print("Running SMC2 simulation ...")

# hp_lam_p  = np.array([2.0, 0.9, 1.0])
# hp_kp_smc = np.array([1.5, 1.9, 5.0])
# hp_phi_p  = np.array([0.25, 0.25, 0.25])
# hp_lam_a  = np.array([8.0, 8.0, 8.0])
# hp_ka_smc = np.array([12.0, 12.0, 9.0])
# hp_phi_a  = np.array([0.5, 0.5, 0.55])
# hp_kd_p   = np.array([0, 0.0, 0])





SMC2_GAINS = dict(
    lam_p  = np.array([2.0, 0.9, 1.0]),
    kp_smc = np.array([1.5, 1.9, 5.0]),   #fixed as paper [6] in david reference list
    phi_p  = np.array([0.25, 0.25, 0.25]),

    lam_a  = np.array([8.0, 8.0, 8.0]),
    ka_smc = np.array([12.0, 12.0, 9.0]),
    phi_a  = np.array([0.5, 0.5, 0.55]),
    
    
    # lam_a  = np.array([6.0, 8.0, 8.0]),
    # ka_smc = np.array([9.5, 10.0, 9]),
    # phi_a  = np.array([0.2, 0.5, 0.55]),
)
log_smc2, ref_smc2, att_ref_smc2, _, smc_pos2, smc_att2 = run_smc(**SMC2_GAINS)

# ===================== PID OUTER-LOOP + SMC INNER-LOOP SIMULATOR =====================
def run_pid(lam_a, ka_smc, phi_a):
    state   = np.zeros(12)
    log     = np.zeros((steps, 12))
    ref_log = np.zeros((steps, 3))
    att_ref_log = np.zeros((steps, 3))
    t_log   = np.zeros(steps)

    kp_pos = np.array([2.0, 2.5, 4.0])
    ki_pos = np.array([0.5, 0.5, 1.0])
    kd_pos = np.array([2.0, 3.0, 3.0])
    int_lim = np.array([0.30, 0.30, 0.30])

    psi_d        = 0.0
    pos_d_prev = square_reference(0.0)
    vel_d_prev = np.zeros(3)
    pos_d_filt = square_reference(0.0)
    int_pos    = np.zeros(3)

    for k in range(steps):
        t = k * dt
        pos   = state[0:3]
        vel   = state[3:6]
        att   = state[6:9]
        omega = state[9:12]

        pos_d_raw  = square_reference(t)
        pos_d_filt = alpha * pos_d_raw + (1.0 - alpha) * pos_d_filt
        pos_d      = pos_d_filt

        if k == 0:
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
        else:
            vel_d = (pos_d - pos_d_prev) / dt
            acc_d = (vel_d - vel_d_prev) / dt
        pos_d_prev = pos_d
        vel_d_prev = vel_d

        e       = pos_d - pos
        e_dot   = vel_d - vel
        int_pos = np.clip(int_pos + e * dt, -int_lim, int_lim)
        acc_cmd = acc_d + kp_pos * e + ki_pos * int_pos + kd_pos * e_dot

        az = acc_cmd[2] + g
        phi_d   = (acc_cmd[0] * np.sin(psi_d) - acc_cmd[1] * np.cos(psi_d)) / g
        theta_d = (acc_cmd[0] * np.cos(psi_d) + acc_cmd[1] * np.sin(psi_d)) / g
        phi_d   = np.clip(phi_d,   -0.6, 0.6)
        theta_d = np.clip(theta_d, -0.6, 0.6)
        cos_tilt = max(0.5, np.cos(phi_d) * np.cos(theta_d))
        T = np.clip(m * az / cos_tilt, 0.0, 2.0 * m * g)
        att_d = np.array([phi_d, theta_d, psi_d])

        tau, _ = smc_attitude(att, omega, att_d, np.zeros(3),
                              lam_a, ka_smc, phi_a)

        u = np.linalg.solve(mix, np.array([T, tau[0], tau[1], tau[2]]))
        u = np.clip(u, 0.0, 2.0 * m * g / 4.0)

        R = rotation_matrix(*att)
        thrust_world = R @ np.array([0, 0, np.sum(u)])
        acc_lin = thrust_world / m - np.array([0, 0, g])

        p, q, r = omega
        tau_phi, tau_theta, tau_psi = tau
        acc_ang = np.array([
            (tau_phi   + (Iyy - Izz) * q * r) / Ixx,
            (tau_theta + (Izz - Ixx) * p * r) / Iyy,
            (tau_psi   + (Ixx - Iyy) * p * q) / Izz
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

        log[k]        = state
        ref_log[k]    = pos_d
        att_ref_log[k] = att_d
        t_log[k]      = t

    return log, ref_log, att_ref_log, t_log

# ===================== RUN PID SIMULATION =====================
print("Running PID simulation ...")
log_pid, ref_log_pid, att_ref_pid, t_log_pid = run_pid(
    lam_a =np.array([5.0, 6.0, 7.0]),  
    ka_smc=np.array([8.0, 8.0, 7.0]),
    phi_a=np.array([0.5, 0.5, 0.55]))
t_log = t_log_pid





psi_d = 0.0

# ===================== PLOTTING =====================
print("Plotting comparison graphs ...")

COLORS = {
    'RL-SMC': 'tab:blue',
    "Manual Tunned SMC": 'tab:orange',
    'PID':  'tab:red',
    'REF':  'k',
}

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

labels_pos = ['X Distance(m)', 'Y Distance(m)', 'Z Distance(m)']
labels_vel = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)']
labels_att = [r'Roll ($\phi$) (rad)', r'Pitch ($\theta$) (rad)', r'Yaw ($\psi$) (rad)']
all_labels = labels_pos + labels_vel + labels_att
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 16), sharex=True)
fig.suptitle('State Tracking — RL-SMC vs Manual Tunned SMC vs PID', fontsize=14)

for idx in range(3):
    ax = axes[idx]
    ax.plot(t_log, ref_smc1[:, idx], COLORS['REF'], lw=1.0, ls='-.',  label='Reference')
    ax.plot(t_log, log_smc1[:, idx], COLORS['RL-SMC'], lw=2.0, ls='-',  label='RL-SMC')
    ax.plot(t_log, log_smc2[:, idx], COLORS["Manual Tunned SMC"], lw=2.0, ls='--', label="Manual Tunned SMC")
    ax.plot(t_log, log_pid[:, idx], COLORS['PID'], lw=2.0, ls=':',  label='PID')
    ax.set_ylabel(all_labels[idx], fontsize=12)
    ax.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax.grid(True)

axes[2].set_xlabel('Time (s)', fontsize=12)

for offset, idx in enumerate(range(3,6)):
    ax = axes[idx]
    ax.plot(t_log, log_smc1[:, 3+offset], COLORS['RL-SMC'], lw=2.0, ls='-',  label='RL-SMC')
    ax.plot(t_log, log_smc2[:, 3+offset], COLORS["Manual Tunned SMC"], lw=2.0, ls='--', label="Manual Tunned SMC")
    ax.plot(t_log, log_pid[:, 3+offset], COLORS['PID'], lw=2.0, ls=':',  label='PID')
    ax.set_ylabel(all_labels[idx], fontsize=12)
    ax.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax.grid(True)

axes[5].set_xlabel('Time (s)', fontsize=12)

for offset, idx in enumerate(range(6,9)):
    ax = axes[idx]
    ax.plot(t_log, att_ref_smc1[:, offset], COLORS['REF'], lw=1.5, ls='--',  label='Reference')
    ax.plot(t_log, log_smc1[:, 6+offset], COLORS['RL-SMC'], lw=2.0, ls='-',  label='RL-SMC')
    ax.plot(t_log, log_smc2[:, 6+offset], COLORS["Manual Tunned SMC"], lw=2.0, ls='--', label="Manual Tunned SMC")
    ax.plot(t_log, log_pid[:, 6+offset], COLORS['PID'], lw=2.0, ls=':',  label='PID')
    ax.set_ylabel(all_labels[idx], fontsize=12)
    ax.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax.grid(True)

    if offset == 0:
        ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=2)

axes[8].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()
plt.savefig("state_tracking_fig1.pdf", bbox_inches="tight")

# ---- 2. Tracking Errors ----
err_pos1 = log_smc1[:, 0:3] - ref_smc1
err_pos2 = log_smc2[:, 0:3] - ref_smc2
err_pos_p = log_pid[:, 0:3]  - ref_log_pid
err_att1 = log_smc1[:, 6:9] - att_ref_smc1
err_att2 = log_smc2[:, 6:9] - att_ref_smc2
err_att_p = log_pid[:, 6:9]  - att_ref_pid

labels_pos_err = ['Error X (m)', 'Error Y (m)', 'Error Z (m)']
labels_att_err = [r'Error Roll ($\phi$) (rad)', r'Error Pitch ($\theta$) (rad)', r'Error Yaw ($\psi$) (rad)']
all_err_labels = labels_pos_err + labels_att_err

fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)
fig.suptitle('Tracking Errors — RL-SMC vs Manual Tunned SMC vs PID', fontsize=14)

for idx in range(3):
    ax = axes[idx]
    ax.plot(t_log, err_pos1[:, idx], COLORS['RL-SMC'], lw=2.5, label='RL-SMC')
    ax.plot(t_log, err_pos2[:, idx], COLORS["Manual Tunned SMC"], lw=2.5, label="Manual Tunned SMC")
    ax.plot(t_log, err_pos_p[:, idx], COLORS['PID'],  lw=2.5, label='PID')
    ax.set_ylabel(all_err_labels[idx], fontsize=12)
    ax.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax.grid(True)

axes[2].set_xlabel('Time (s)', fontsize=12)

for offset, idx in enumerate(range(3,6)):
    ax = axes[idx]
    ax.plot(t_log, err_att1[:, offset], COLORS['RL-SMC'], lw=2.5, label='RL-SMC')
    ax.plot(t_log, err_att2[:, offset], COLORS["Manual Tunned SMC"], lw=2.5, label="Manual Tunned SMC")
    ax.plot(t_log, err_att_p[:, offset], COLORS['PID'], lw=2.5, label='PID')
    ax.set_ylabel(all_err_labels[idx], fontsize=12)
    ax.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax.grid(True)

axes[5].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()
plt.savefig("tracking_error_fig2.pdf", bbox_inches="tight")

# ---- 3. Sliding Surfaces vs Time ----
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle('Sliding Surfaces over Time — RL-SMC vs Manual Tunned SMC', fontsize=14)

for i, name in enumerate(['x', 'y', 'z']):
    axes[0].plot(t_log, smc_pos1[:, i], COLORS['RL-SMC'], lw=2, label=rf'RL-SMC $s_{name}$')
    axes[0].plot(t_log, smc_pos2[:, i], COLORS["Manual Tunned SMC"], lw=2, linestyle='--', label=rf'Manual Tunned SMC $s_{name}$')

axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[0].set_ylabel(r'$s$ (position)', fontsize=12)
axes[0].grid(True)
axes[0].legend(fontsize=12, ncol=3, frameon=False)

for i, sym in enumerate([r'\phi', r'\theta', r'\psi']):
    axes[1].plot(t_log, smc_att1[:, i], COLORS['RL-SMC'], lw=2, label=rf'RL-SMC $s_{{{sym}}}$')
    axes[1].plot(t_log, smc_att2[:, i], COLORS["Manual Tunned SMC"], lw=2, linestyle='--', label=rf'Manual Tunned SMC $s_{{{sym}}}$')

axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[1].set_ylabel(r'$s$ (attitude)', fontsize=12)
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].grid(True)
axes[1].legend(fontsize=12, ncol=3, frameon=False)

plt.tight_layout()
plt.savefig("sliding_surfaces_fig3.pdf", bbox_inches="tight")

# ---- 4. Phase Plane ----
pos_sym = [r'X', r'Y', r'Z']
att_sym = [r'\phi', r'\theta', r'\psi']

fig, axes = plt.subplots(4, 3, figsize=(14, 14))
fig.suptitle('Phase Plane: $s$ vs $\dot{s}$', fontsize=16)

for i in range(3):
    axes[0, i].plot(smc_pos1[:, i], smc_pos1[:, 3 + i], COLORS['RL-SMC'], lw=2.5)
    axes[0, i].scatter(smc_pos1[0, i], smc_pos1[0, 3 + i], color='green', s=30, zorder=5)
    axes[0, i].scatter(smc_pos1[-1, i], smc_pos1[-1, 3 + i], color='black', s=30, zorder=5)
    axes[0, i].axhline(0, color='gray', lw=1.5, ls='--')
    axes[0, i].axvline(0, color='gray', lw=1.5, ls='--')
    axes[0, i].set_title(r'RL-SMC pos: $%s$' % pos_sym[i])
    axes[0, i].set_xlabel(r'$s_{%s}$' % pos_sym[i], fontsize=16)
    axes[0, i].set_ylabel(r'$\dot{s}_{%s}$' % pos_sym[i], fontsize=16)
    axes[0, i].grid(True)

    axes[1, i].plot(smc_att1[:, i], smc_att1[:, 3 + i], COLORS['RL-SMC'], lw=2.5)
    axes[1, i].scatter(smc_att1[0, i], smc_att1[0, 3 + i], color='green', s=30, zorder=5)
    axes[1, i].scatter(smc_att1[-1, i], smc_att1[-1, 3 + i], color='black', s=30, zorder=5)
    axes[1, i].axhline(0, color='gray', lw=1.5, ls='--')
    axes[1, i].axvline(0, color='gray', lw=1.5, ls='--')
    axes[1, i].set_title(r'RL-SMC att: $%s$' % att_sym[i])
    axes[1, i].set_xlabel(r'$s_{%s}$' % att_sym[i], fontsize=16)
    axes[1, i].set_ylabel(r'$\dot{s}_{%s}$' % att_sym[i], fontsize=16)
    axes[1, i].grid(True)

    axes[2, i].plot(smc_pos2[:, i], smc_pos2[:, 3 + i], COLORS["Manual Tunned SMC"], lw=2.5)
    axes[2, i].scatter(smc_pos2[0, i], smc_pos2[0, 3 + i], color='green', s=30, zorder=5)
    axes[2, i].scatter(smc_pos2[-1, i], smc_pos2[-1, 3 + i], color='black', s=30, zorder=5)
    axes[2, i].axhline(0, color='gray', lw=1.5, ls='--')
    axes[2, i].axvline(0, color='gray', lw=1.5, ls='--')
    axes[2, i].set_title(r'Manual Tunned SMC pos: $%s$' % pos_sym[i])
    axes[2, i].set_xlabel(r'$s_{%s}$' % pos_sym[i], fontsize=16)
    axes[2, i].set_ylabel(r'$\dot{s}_{%s}$' % pos_sym[i], fontsize=16)
    axes[2, i].grid(True)

    axes[3, i].plot(smc_att2[:, i], smc_att2[:, 3 + i], COLORS["Manual Tunned SMC"], lw=2.5)
    axes[3, i].scatter(smc_att2[0, i], smc_att2[0, 3 + i], color='green', s=30, zorder=5)
    axes[3, i].scatter(smc_att2[-1, i], smc_att2[-1, 3 + i], color='black', s=30, zorder=5)
    axes[3, i].axhline(0, color='gray', lw=1.5, ls='--')
    axes[3, i].axvline(0, color='gray', lw=1.5, ls='--')
    axes[3, i].set_title(r'Manual Tunned SMC att: $%s$' % att_sym[i])
    axes[3, i].set_xlabel(r'$s_{%s}$' % att_sym[i], fontsize=16)
    axes[3, i].set_ylabel(r'$\dot{s}_{%s}$' % att_sym[i], fontsize=16)
    axes[3, i].grid(True)

plt.tight_layout()
plt.savefig("phase_plane_fig4.pdf", bbox_inches="tight")

# ---- 5. 3D Trajectory Overlay ----
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.plot(ref_smc1[:, 0], ref_smc1[:, 1], ref_smc1[:, 2], COLORS['REF'], lw=1, ls='-.', label='Reference (square)')
ax.plot(log_smc1[:, 0], log_smc1[:, 1], log_smc1[:, 2], COLORS['RL-SMC'], lw=2.5, label='RL-SMC')
ax.plot(log_smc2[:, 0], log_smc2[:, 1], log_smc2[:, 2], COLORS["Manual Tunned SMC"], lw=2.5, label="Manual Tunned SMC")
ax.plot(log_pid[:, 0], log_pid[:, 1], log_pid[:, 2], COLORS['PID'], lw=2.5, label='PID')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory — RL-SMC vs Manual Tunned SMC vs PID')
leg = ax.legend(loc='upper left', fontsize=14)
for text in leg.get_texts():
    text.set_fontweight('bold')
ax.grid(True)
plt.tight_layout()

# ---- 8. RMSE / IAE / MAE per axis (full trajectory) ----
def print_metrics(name, log, ref, att_ref):
    err_pos = log[:, 0:3] - ref[:, 0:3]
    err_att = log[:, 6:9] - att_ref

    rmse_xyz = np.sqrt(np.mean(err_pos ** 2, axis=0))
    rmse_rpy = np.sqrt(np.mean(err_att ** 2, axis=0))
    rmse_3d  = np.sqrt(np.mean(np.sum(err_pos**2, axis=1)))
    rmse_att = np.sqrt(np.mean(np.sum(err_att**2, axis=1)))

    pos_norm = np.sqrt(np.sum(err_pos**2, axis=1))
    att_norm = np.sqrt(np.sum(err_att**2, axis=1))

    iae_xyz = np.sum(np.abs(err_pos), axis=0) * dt
    iae_rpy = np.sum(np.abs(err_att), axis=0) * dt
    iae_3d  = np.sum(pos_norm) * dt
    iae_att = np.sum(att_norm) * dt

    mae_xyz = np.max(np.abs(err_pos), axis=0)
    mae_rpy = np.max(np.abs(err_att), axis=0)
    mae_3d  = np.max(pos_norm)
    mae_att = np.max(att_norm)

    print(f"\n  Controller: {name}")
    print(f"  {'Metric':<10} {'X':>12} {'Y':>12} {'Z':>12} {'3D':>12}")
    print(f"  {'RMSE (m)':<10} {rmse_xyz[0]:>12.5f} {rmse_xyz[1]:>12.5f} {rmse_xyz[2]:>12.5f} {rmse_3d:>12.5f}")
    print(f"  {'IAE (m*s)':<10} {iae_xyz[0]:>12.5f} {iae_xyz[1]:>12.5f} {iae_xyz[2]:>12.5f} {iae_3d:>12.5f}")
    print(f"  {'MAE (m)':<10} {mae_xyz[0]:>12.5f} {mae_xyz[1]:>12.5f} {mae_xyz[2]:>12.5f} {mae_3d:>12.5f}")

    print(f"  {'':<10} {'Roll':>12} {'Pitch':>12} {'Yaw':>12} {'Att':>12}")
    print(f"  {'Metric':<10} {'(rad)':>12} {'(rad)':>12} {'(rad)':>12} {'(rad)':>12}")
    print(f"  {'RMSE':<10} {rmse_rpy[0]:>12.5f} {rmse_rpy[1]:>12.5f} {rmse_rpy[2]:>12.5f} {rmse_att:>12.5f}")
    print(f"  {'IAE (rad*s)':<10} {iae_rpy[0]:>12.5f} {iae_rpy[1]:>12.5f} {iae_rpy[2]:>12.5f} {iae_att:>12.5f}")
    print(f"  {'MAE (rad)':<10} {mae_rpy[0]:>12.5f} {mae_rpy[1]:>12.5f} {mae_rpy[2]:>12.5f} {mae_att:>12.5f}")

print("\n" + "=" * 78)
print("  PERFORMANCE METRICS (full trajectory)")
print("=" * 78)
print_metrics("RL-SMC",           log_smc1, ref_smc1,     att_ref_smc1)
print_metrics("Manual Tunned SMC", log_smc2, ref_smc2,     att_ref_smc2)
print_metrics("PID",              log_pid,  ref_log_pid,  att_ref_pid)
print("=" * 78)

# ============================================================
# ---- 9. Table 5 Peak Attitude Calculation (in Degrees) ----
# ============================================================
def get_peak_deg(log_data):
    # Attitude columns are indices 6, 7, 8 (roll phi, pitch theta, yaw psi)
    # Convert from radians to degrees: rad * (180 / pi)
    peaks_rad = np.max(np.abs(log_data[:, 6:9]), axis=0)
    peaks_deg = peaks_rad * (180.0 / np.pi)
    return peaks_deg

rl_peaks = get_peak_deg(log_smc1)
ht_peaks = get_peak_deg(log_smc2)
pid_peaks = get_peak_deg(log_pid)

print("\n" + "=" * 65)
print("  TABLE 5: PEAK ATTITUDE ANGLES (DEGREES)")
print("=" * 65)
print(f"{'Controller':<18} | {'Peak Roll (deg)':<15} | {'Peak Pitch (deg)':<15} | {'Peak Yaw (deg)':<15}")
print("-" * 65)
print(f"{'RL SMC':<18} | {rl_peaks[0]:<15.2f} | {rl_peaks[1]:<15.2f} | {rl_peaks[2]:<15.2f}")
print(f"{'Fixed-gain SMC':<18} | {ht_peaks[0]:<15.2f} | {ht_peaks[1]:<15.2f} | {ht_peaks[2]:<15.2f}")
print(f"{'Tuned PID':<18} | {pid_peaks[0]:<15.2f} | {pid_peaks[1]:<15.2f} | {pid_peaks[2]:<15.2f}")
print("=" * 65)

# ---- 10. 3-D trajectory animation ---------------------------------------
# The animation uses the already computed trajectories; it does not rerun or
# alter the plant/controller simulation.
def save_square_animation():
    out_dir = Path(__file__).resolve().parent / "animations"
    out_dir.mkdir(exist_ok=True)
    paths = [ref_smc1[:, :3], log_smc1[:, :3], log_smc2[:, :3], log_pid[:, :3]]
    labels = ["Reference", "RL-SMC", "Manual SMC", "PID"]
    colors = ["black", COLORS['RL-SMC'], COLORS["Manual Tunned SMC"], COLORS['PID']]
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
    ax.set_title("Trajectory 1: square-flight animation")
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
        ax.set_title(f"Trajectory 1: square flight  (t = {t_log[idx]:.2f} s)")
        return lines + points

    anim = FuncAnimation(fig, update, frames=nframes, interval=45, blit=False)
    gif_path = out_dir / "trajectory_1_square.gif"
    anim.save(gif_path, writer=PillowWriter(fps=22))
    print(f"Saved 3-D animation: {gif_path}")
    try:
        mp4_path = out_dir / "trajectory_1_square.mp4"
        anim.save(mp4_path, writer=FFMpegWriter(fps=22, bitrate=1800))
        print(f"Saved 3-D animation: {mp4_path}")
    except Exception as exc:
        print(f"MP4 not saved (FFmpeg unavailable): {exc}")
    plt.close(fig)

save_square_animation()

plt.show()
