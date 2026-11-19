import os
import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')



_l = 0.119 / np.sqrt(2)
_b = 2.84e-6
_d = 1.46e-10
MIX = np.array([
    [1.0,    1.0,    1.0,    1.0],
    [ _l,    -_l,    -_l,     _l ],
    [ _l,     _l,    -_l,    -_l ],
    [-_d/_b, _d/_b, -_d/_b,  _d/_b]
])
MIX_INV = np.linalg.inv(MIX)

TRAJ_SQUARE, TRAJ_SINGLE_STEP = 0, 1
AXES = {'x': 0, 'y': 1, 'z': 2, 'yaw': 3}
SIMULATION_MODEL = 'comparison'

EFFORT_WEIGHT = 0.002  #effort weoght from 

OVERSHOOT_WEIGHT = 0.5
OVERSHOOT_VEL_THRESHOLD = 0.05


def step_reference(t, traj, axis):
    if traj == TRAJ_SINGLE_STEP:
        x_ref, y_ref, z_ref, psi_d = 0.0, 0.0, 2.0, 0.0
        if 4.0 <= t < 16.0:
            if axis == 0: x_ref = 2.0
            elif axis == 1: y_ref = 2.0
            elif axis == 2: z_ref = 3.0
            elif axis == 3: psi_d = 0.785
        return x_ref, y_ref, z_ref, psi_d
    side    = 2.0
    T       = 4.0
    z_hover = 2.0
    if t < T:
        return 0.0, 0.0, z_hover * (t / T), 0.0
    elif t < 2*T:
        s = (t - T) / T
        return side * s, 0.0, z_hover, 0.0
    elif t < 3*T:
        s = (t - 2*T) / T
        return side, side * s, z_hover, 0.0
    elif t < 4*T:
        s = (t - 3*T) / T
        return side * (1 - s), side, z_hover, 0.0
    elif t < 5*T:
        s = (t - 4*T) / T
        return 0.0, side * (1 - s), z_hover, 0.0
    return 0.0, 0.0, z_hover, 0.0


def allocate_motors(thrust, tau, u_max):
    """Saturation-aware allocation that preserves baseline flight stability."""
    collective = min(max(thrust / 4.0, 0.0), u_max)
    differential = np.empty(4)
    scale = 1.0
    for j in range(4):
        differential[j] = MIX_INV[j, 1]*tau[0] + MIX_INV[j, 2]*tau[1]
        if differential[j] > 0.0:
            scale = min(scale, (u_max-collective)/differential[j])
        elif differential[j] < 0.0:
            scale = min(scale, -collective/differential[j])
    u = np.empty(4)
    yaw_scale = 1.0
    for j in range(4):
        u[j] = collective + scale*differential[j]
        yaw = MIX_INV[j, 3]*tau[2]
        if yaw > 0.0:
            yaw_scale = min(yaw_scale, (u_max-u[j])/yaw)
        elif yaw < 0.0:
            yaw_scale = min(yaw_scale, -u[j]/yaw)
    actual = np.zeros(4)
    for j in range(4):
        u[j] = min(max(u[j] + max(0.0, yaw_scale)*MIX_INV[j, 3]*tau[2], 0.0), u_max)
        for c in range(4):
            actual[c] += MIX[c, j]*u[j]
    return u, actual


def _simulate(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, traj, axis, evaluate_reward, effort_weight, overshoot_weight):
    
    m, g, dt = 0.08, 9.81, 0.005
    Ixx, Iyy, Izz = 7.2e-5, 6.8e-5, 1.53e-4
    steps = int(22.0 / dt)
    state = np.zeros(12)
    log, ref_log = np.zeros((steps, 12)), np.zeros((steps, 3))
    t_log = np.arange(steps) * dt
    pos_d, pos_prev, vel_prev = np.zeros(3), np.zeros(3), np.zeros(3)
    acc_prev = np.zeros(3)
    acc_cmd, tau, u = np.empty(3), np.empty(3), np.empty(4)
    u_prev = np.zeros(4)
    square_error = 0.0
    overshoot_error = 0.0
    control_effort_total = 0.0
    for k in range(steps):
        if not np.all(np.isfinite(state)):
            log[k:], ref_log[k:] = np.nan, np.nan
            return 1e6, -1e6, log, ref_log, t_log
        xref, yref, zref, psi_d = step_reference(t_log[k], traj, axis)
        raw = (xref, yref, zref)
        for j in range(3):
            pos_d[j] = 0.04 * raw[j] + 0.96 * pos_d[j]
            velocity, acceleration = 0.0, 0.0
            if k > 0:
                velocity = (pos_d[j] - pos_prev[j]) / dt
                acceleration = min(max((velocity - vel_prev[j]) / dt, -10.0), 10.0)
           
            acceleration = 0.20 * acceleration + 0.80 * acc_prev[j]
            acc_prev[j] = acceleration
            pos_prev[j], vel_prev[j] = pos_d[j], velocity
            surface = state[3+j] - velocity + lam_p[j] * (state[j] - pos_d[j])
            
            acc_cmd[j] = (acceleration - lam_p[j] * (state[3+j] - velocity)
                          - kp_smc[j] * np.tanh(surface / phi_p[j]))
        sp, cp = np.sin(psi_d), np.cos(psi_d)
        phi_d = min(max((acc_cmd[0]*sp - acc_cmd[1]*cp) / g, -0.6), 0.6)
        theta_d = min(max((acc_cmd[0]*cp + acc_cmd[1]*sp) / g, -0.6), 0.6)
        thrust = min(max(m * (acc_cmd[2] + g) /
                         max(0.5, np.cos(phi_d)*np.cos(theta_d)), 0.0), 2*m*g)
        desired = (phi_d, theta_d, psi_d)
        inertia = (Ixx, Iyy, Izz)
        for j in range(3):
            surface = state[9+j] + lam_a[j] * (state[6+j] - desired[j])
            tau[j] = -inertia[j] * ka_smc[j] * np.tanh(surface / phi_a_in[j])
        
        u, actual = allocate_motors(thrust, tau, 2*m*g/4)
        
        tau[:] = actual[1:4]
        control_effort_total += np.sum(np.abs(u - u_prev))
        u_prev[:] = u
        phi, theta, psi = state[6], state[7], state[8]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
        sy, cy = np.sin(psi), np.cos(psi)
        thrust_actual = np.sum(u)
        
        rotation = np.array([
            [ct*cy, sf*st*cy - cf*sy, cf*st*cy + sf*sy],
            [ct*sy, sf*st*sy + cf*cy, cf*st*sy - sf*cy],
            [-st, sf*ct, cf*ct]
        ])
        thrust_world = rotation @ np.array([0.0, 0.0, thrust_actual])
        acceleration = thrust_world / m - np.array([0.0, 0.0, g])
        p, q, r = state[9], state[10], state[11]
        angular_acc = np.array([(tau[0] + (Iyy-Izz)*q*r)/Ixx,
                                (tau[1] + (Izz-Ixx)*p*r)/Iyy,
                                (tau[2] + (Ixx-Iyy)*p*q)/Izz])
        angular_rate = np.array([p + sf*np.tan(theta)*q + cf*np.tan(theta)*r,
                                 cf*q - sf*r, (sf*q + cf*r)/max(0.1, ct)])
        state[0:3] += state[3:6] * dt
        state[3:6] += acceleration * dt
        state[6:9] += angular_rate * dt
        state[9:12] += angular_acc * dt
        if state[2] < 0:
            state[2], state[5] = 0.0, max(state[5], 0.0)
        
        if (not np.all(np.isfinite(state)) or np.any(np.abs(state[0:3]) > 20.0)
                or np.any(np.abs(state[3:6]) > 15.0)
                or abs(state[6]) >= 1.4 or abs(state[7]) >= 1.4
                or np.any(np.abs(state[9:12]) > 20.0)):
            log[k:], ref_log[k:] = np.nan, np.nan
            return 1e6, -1e6, log, ref_log, t_log
        log[k], ref_log[k] = state, pos_d
        for j in range(3):
            err = state[j] - pos_d[j]
            square_error += err * err
            ref_velocity = (pos_d[j] - pos_prev[j]) / dt
            
            if (abs(ref_velocity) < OVERSHOOT_VEL_THRESHOLD
                    and err * state[3+j] > 0.0):
                overshoot_error += err * err
    
    pos_rmse = np.sqrt(square_error / steps)
    overshoot_rmse = np.sqrt(overshoot_error / steps)
    avg_effort = control_effort_total / steps
    cost = pos_rmse + effort_weight * avg_effort + overshoot_weight * overshoot_rmse
    return cost, -cost, log, ref_log, t_log


def _simulate_physical(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, traj, axis, evaluate_reward, effort_weight, overshoot_weight):
    m   = 0.080
    g   = 9.81
    Ixx = 7.2e-5
    Iyy = 6.8e-5
    Izz = 1.53e-4
    dt     = 0.005
    alpha  = 0.04

    

    T_end = 22.0
    steps = int(T_end / dt)
    state = np.zeros(12)
    log     = np.zeros((steps, 12))
    ref_log = np.zeros((steps, 3))
    t_log   = np.zeros(steps)
    pos_d      = np.zeros(3)  
    pos_d_prev = np.zeros(3)
    vel_d      = np.zeros(3)
    vel_d_prev = np.zeros(3)
    acc_d      = np.zeros(3)
    acc_d_prev = np.zeros(3)
    acc_cmd    = np.zeros(3)
    e          = np.zeros(3)
    e_dot      = np.zeros(3)
    omega      = np.zeros(3)
    tau        = np.zeros(3)
    u          = np.zeros(4)
    u_prev     = np.zeros(4)
    T_max = 2.0 * m * g
    u_max = 2.0 * m * g / 4.0

    control_effort_total = 0.0
    squared_error_total = 0.0
    overshoot_error_total = 0.0
    att_ref = np.zeros(3)
    att_ref_rate = np.zeros(3)
    att_ref_acc = np.zeros(3)

    for k in range(steps):
        t = k * dt

        
        if (not np.all(np.isfinite(state)) or abs(state[6]) >= 1.4
                or abs(state[7]) >= 1.4 or np.any(np.abs(state[3:6]) >= 15.0)
                or np.any(np.abs(state[9:12]) >= 20.0)):
            log[k:] = np.nan
            ref_log[k:] = np.nan
            return 1e6, -1e6, log, ref_log, t_log

        x_ref, y_ref, z_ref, psi_d = step_reference(t, traj, axis)
        pos_d_raw = (x_ref, y_ref, z_ref)

        for i in range(3):
            pos_d[i] = alpha * pos_d_raw[i] + (1.0 - alpha) * pos_d[i]
            if k == 0:
                vel_d[i] = 0.0
                acc_d[i] = 0.0
            else:
                vel_d[i] = (pos_d[i] - pos_d_prev[i]) / dt
                acc_d[i] = (vel_d[i] - vel_d_prev[i]) / dt
                acc_d[i] = min(max(acc_d[i], -10.0), 10.0)
                acc_d[i] = 0.20 * acc_d[i] + 0.80 * acc_d_prev[i]
            acc_d_prev[i] = acc_d[i]
            pos_d_prev[i] = pos_d[i]
            vel_d_prev[i] = vel_d[i]

            e[i]     = state[i] - pos_d[i]
            e_dot[i] = state[3 + i] - vel_d[i]
            s        = e_dot[i] + lam_p[i] * e[i]
            acc_cmd[i] = acc_d[i] - lam_p[i] * e_dot[i] - kp_smc[i] * np.tanh(s / max(phi_p[i], 1e-3))
            if (abs(vel_d[i]) < OVERSHOOT_VEL_THRESHOLD
                    and e[i] * state[3+i] > 0.0):
                overshoot_error_total += e[i] * e[i]

        az = acc_cmd[2] + g
        spsi_d, cpsi_d = np.sin(psi_d), np.cos(psi_d)
        phi_d   = (acc_cmd[0] * spsi_d - acc_cmd[1] * cpsi_d) / g
        theta_d = (acc_cmd[0] * cpsi_d + acc_cmd[1] * spsi_d) / g
        phi_d   = min(max(phi_d,   -0.6), 0.6)
        theta_d = min(max(theta_d, -0.6), 0.6)
        cos_tilt = max(0.5, np.cos(state[6]) * np.cos(state[7]))
        T = min(max(m * az / cos_tilt, 0.0), T_max)

        
        att_d = (phi_d, theta_d, psi_d)
        phi, theta = state[6], state[7]
        p, q, r = state[9], state[10], state[11]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
        rates = np.array([p + sf * st / ct * q + cf * st / ct * r,
                          cf * q - sf * r, (sf * q + cf * r) / ct])
        eta_acc = np.zeros(3)
        for i in range(3):
            att_ref_acc[i] = 100.0 * (att_d[i] - att_ref[i]) - 20.0 * att_ref_rate[i]
            err = state[6+i] - att_ref[i]
            if i == 2:
                err = (err + np.pi) % (2*np.pi) - np.pi
            err_rate = rates[i] - att_ref_rate[i]
            att_s = err_rate + lam_a[i] * err
            eta_acc[i] = (att_ref_acc[i] - lam_a[i] * err_rate
                          - ka_smc[i] * np.tanh(att_s / phi_a_in[i]))
            omega[i] = state[9+i]
       
        fd, td, yd = rates[0], rates[1], rates[2]
        pd_cmd = eta_acc[0] - st*eta_acc[2] - ct*td*yd
        qd_cmd = (cf*eta_acc[1] + sf*ct*eta_acc[2] - sf*fd*td
                  + cf*ct*fd*yd - sf*st*td*yd)
        rd_cmd = (-sf*eta_acc[1] + cf*ct*eta_acc[2] - cf*fd*td
                  - sf*ct*fd*yd - cf*st*td*yd)
        tau[0] = Ixx*pd_cmd + (Izz-Iyy)*q*r
        tau[1] = Iyy*qd_cmd + (Ixx-Izz)*p*r
        tau[2] = Izz*rd_cmd + (Iyy-Ixx)*p*q

        u, actual = allocate_motors(T, tau, u_max)

        squared_error_total += np.sum(e * e)
       
        control_effort_total += np.sum(np.abs(u - u_prev))
    
        tau[:] = actual[1:4]

        u_prev[:] = u

        phi, theta, psi = state[6], state[7], state[8]
        sphi, cphi = np.sin(phi), np.cos(phi)
        sth, cth_raw = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)
       
        U = np.sum(u)
        rotation = np.array([
            [cth_raw*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth_raw*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth, sphi*cth_raw, cphi*cth_raw]
        ])
        thrust_world = rotation @ np.array([0.0, 0.0, U])
        acc_lin = thrust_world / m - np.array([0.0, 0.0, g])
        p, q, r = omega[0], omega[1], omega[2]
        acc_ang = np.array([
            (tau[0] + (Iyy - Izz) * q * r) / Ixx,
            (tau[1] + (Izz - Ixx) * p * r) / Iyy,
            (tau[2] + (Ixx - Iyy) * p * q) / Izz
        ])
        cth = max(0.1, cth_raw)
        tth = np.tan(theta)
        att_dot = np.array([
            p + sphi * tth * q + cphi * tth * r,
            cphi * q - sphi * r,
            (sphi / cth) * q + (cphi / cth) * r
        ])

        
        log[k] = state
        ref_log[k] = pos_d
        t_log[k] = t
        att_ref += att_ref_rate * dt
        att_ref_rate += att_ref_acc * dt

        state[0:3]  += state[3:6] * dt
        state[3:6]  += acc_lin * dt
        state[6:9]  += att_dot * dt
        state[9:12] += acc_ang * dt

        if state[2] < 0:
            state[2] = 0
            state[5] = max(state[5], 0.0)

    
    pos_rmse = np.sqrt(squared_error_total / (3 * steps))
    avg_effort = control_effort_total / steps
    overshoot_rmse = np.sqrt(overshoot_error_total / (3 * steps))
    cost = (pos_rmse + effort_weight * avg_effort
            + overshoot_weight * overshoot_rmse)
    return cost, -cost, log, ref_log, t_log


def run_quad_simulation(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, kd_p=None, active_axis='x', traj_type='square', evaluate_reward=False, simulation_model=None):
   
    traj = TRAJ_SINGLE_STEP if traj_type == 'single_step' else TRAJ_SQUARE
    gains = [np.asarray(a, dtype=np.float64) for a in (lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in)]
    if any(a.shape != (3,) or not np.all(np.isfinite(a)) or np.any(a <= 0) for a in gains):
        raise ValueError("Gain vectors must have three finite positive entries.")
    if traj_type not in ('square', 'single_step') or active_axis not in (*AXES, 'xyz'):
        raise ValueError("Unknown trajectory or axis.")
    model = SIMULATION_MODEL if simulation_model is None else simulation_model
    if model not in ('comparison', 'physical'):
        raise ValueError('simulation_model must be comparison or physical')
    kernel = _simulate if model == 'comparison' else _simulate_physical
    return kernel(*gains, traj, AXES.get(active_axis, -1), evaluate_reward, EFFORT_WEIGHT, OVERSHOOT_WEIGHT)


def yaw_step_rmse(lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, kd_p=None):
    _, _, log, _, t_log = run_quad_simulation(
        lam_p, kp_smc, phi_p, lam_a, ka_smc, phi_a_in, kd_p,
        active_axis='yaw', traj_type='single_step', evaluate_reward=False
    )
    yaw_ref = np.array([step_reference(t, TRAJ_SINGLE_STEP, 3)[3] for t in t_log])
    yaw_error = (log[:, 8] - yaw_ref + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(np.mean(yaw_error ** 2))


YAW_TRAIN_WEIGHT = 0.2



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
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    def sample(self, batch_size=32):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
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



hp_lam_p  = np.array([2.0, 0.9, 1.0])
hp_kp_smc = np.array([1.5, 1.9, 5.0])
hp_phi_p  = np.array([0.25, 0.25, 0.25])  #hand picked gains
hp_lam_a  = np.array([8.0, 8.0, 8.0])
hp_ka_smc = np.array([12.0, 12.0, 9.0])
hp_phi_a  = np.array([0.5, 0.5, 0.55])
hp_kd_p   = np.array([0, 0.0, 0])

num_episodes = 400
batch_size = 32
gamma = 0.95
tau = 0.005
NUM_TUNING_PASSES = 3

def train_channel_3param(name, loop_type, index, g_mins, g_maxs, init_params, patience=80,
                         cur_lam_p=None, cur_kp_p=None, cur_phi_p=None,
                         cur_lam_a=None, cur_ka_a=None, cur_phi_a=None,
                         initialize_from_baseline=False):
    if cur_lam_p is None:  cur_lam_p  = hp_lam_p.copy()
    if cur_kp_p is None:   cur_kp_p   = hp_kp_smc.copy()
    if cur_phi_p is None:  cur_phi_p  = hp_phi_p.copy()
    if cur_lam_a is None:  cur_lam_a  = hp_lam_a.copy()
    if cur_ka_a is None:   cur_ka_a   = hp_ka_smc.copy()
    if cur_phi_a is None:  cur_phi_a  = hp_phi_a.copy()

    print(f"--- Training {name} ({loop_type}) [lambda, K_smc, delta] ---")
    actor = MultiParamActor(gain_mins=g_mins, gain_maxs=g_maxs)
    if initialize_from_baseline:
       
        with torch.no_grad():
            actor.net[-2].weight.mul_(0.15)
            center = np.clip(2*(np.asarray(init_params)-np.asarray(g_mins)) /
                             (np.asarray(g_maxs)-np.asarray(g_mins))-1, -0.98, 0.98)
            actor.net[-2].bias.copy_(torch.tensor(np.arctanh(center), dtype=torch.float32))
    critic = MultiParamCritic()
    t_actor = MultiParamActor(gain_mins=g_mins, gain_maxs=g_maxs)
    t_critic = MultiParamCritic()
    t_actor.load_state_dict(actor.state_dict())
    t_critic.load_state_dict(critic.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=5e-5)
    opt_c = optim.Adam(critic.parameters(), lr=5e-5)
    buf = ReplayBuffer()
    noise = OUNoise(3, sigma=0.4, theta=0.15)

    best_cost = float('inf')
    best_params = list(init_params)


    search_init = (np.asarray(init_params, dtype=float).copy()
                   if initialize_from_baseline else
                   np.random.uniform(np.asarray(g_mins), np.asarray(g_maxs)))
    lp0, kpp0, pp0 = cur_lam_p.copy(), cur_kp_p.copy(), cur_phi_p.copy()
    la0, kaa0, paa0 = cur_lam_a.copy(), cur_ka_a.copy(), cur_phi_a.copy()
    if loop_type == 'position':
        lp0[index], kpp0[index], pp0[index] = search_init
    else:
        la0[index], kaa0[index], paa0[index] = search_init
    base_cost, _, _, _, _ = run_quad_simulation(
        lp0, kpp0, pp0, la0, kaa0, paa0, hp_kd_p,
        active_axis='xyz', traj_type='square', evaluate_reward=False
    )
    is_yaw = (loop_type == 'attitude' and index == 2)
    if is_yaw:
        base_cost = base_cost + YAW_TRAIN_WEIGHT * yaw_step_rmse(
            lp0, kpp0, pp0, la0, kaa0, paa0, hp_kd_p)
    best_cost = base_cost
    best_params = np.array(search_init, dtype=float)
    current_action = np.array(search_init, dtype=float)
    current_cost = base_cost
    prev_cost = base_cost
    print(f"  Initial search objective for this channel: {base_cost:.6f}")

    no_improve_count = 0

    for ep in range(num_episodes):
        cost_trend = np.clip(current_cost - prev_cost, -10.0, 10.0)
        state = np.array([current_action[0], current_action[1], current_action[2],
                          min(current_cost, 10.0), cost_trend], dtype=np.float32)
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = actor(s_tensor).detach().numpy()[0]

        
        noise_scale = max(0.05, 1.0 - (ep / (num_episodes * 0.8)))
        action_noisy = np.clip(action + noise.sample() * noise_scale * 0.5 * (np.asarray(g_maxs) - np.asarray(g_mins)), g_mins, g_maxs)

        lp, kpp, pp = cur_lam_p.copy(), cur_kp_p.copy(), cur_phi_p.copy()
        la, kaa, paa = cur_lam_a.copy(), cur_ka_a.copy(), cur_phi_a.copy()

        if loop_type == 'position':
            lp[index]  = action_noisy[0]
            kpp[index] = action_noisy[1]
            pp[index]  = action_noisy[2]
        else:
            la[index]  = action_noisy[0]
            kaa[index] = action_noisy[1]
            paa[index] = action_noisy[2]

        try:
            cost, reward, _, _, _ = run_quad_simulation(
                lp, kpp, pp, la, kaa, paa, hp_kd_p,
                active_axis='xyz', traj_type='square', evaluate_reward=True
            )
            if not np.isfinite(cost) or not np.isfinite(reward):
                cost, reward = 1e6, -1e6
        except Exception as exc:
            cost, reward = 1e6, -1e6
            print(f"{name} episode {ep+1} failed: {type(exc).__name__}: {exc}")

        if is_yaw and cost < 1e5:
            cost = cost + YAW_TRAIN_WEIGHT * yaw_step_rmse(
                lp, kpp, pp, la, kaa, paa, hp_kd_p)
            reward = -cost

        prev_cost = current_cost
        current_action = action_noisy.copy()
        current_cost = cost

        cost_trend_next = np.clip(current_cost - prev_cost, -10.0, 10.0)
        next_state = np.array([current_action[0], current_action[1], current_action[2],
                               min(current_cost, 10.0), cost_trend_next], dtype=np.float32)
        
        state[3:] = np.clip(state[3:], -10.0, 10.0)
        next_state[3:] = np.clip(next_state[3:], -10.0, 10.0)
        buf.push(state, action_noisy, max(reward, -10.0), next_state)

        if cost < best_cost:
            best_cost = cost
            best_params = action_noisy.copy()
            no_improve_count = 0
            print(f"{name} Episode {ep+1} | NEW BEST filtered RMSE: {best_cost:.6f}")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"--> {name} early-stopped at Episode {ep+1}.\n")
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

            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            a_loss = -critic(st, actor(st)).mean()
            opt_a.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_a.step()
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

            for tp, p in zip(t_actor.parameters(), actor.parameters()): tp.data.copy_(tau*p.data + (1-tau)*tp.data)
            for tp, p in zip(t_critic.parameters(), critic.parameters()): tp.data.copy_(tau*p.data + (1-tau)*tp.data)

    return best_params

def main(argv=None):
    global num_episodes, SIMULATION_MODEL
    parser = argparse.ArgumentParser(description="Offline six-channel SMC gain tuning")
    parser.add_argument('--passes', type=int, default=NUM_TUNING_PASSES)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--patience', type=int, default=55)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: random/unfixed)')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--model', choices=['comparison', 'physical'], default='comparison')
    parser.add_argument('--init', choices=['random', 'baseline'], default='random',
                        help='DDPG initialization: blind bounded search (default) or baseline-seeded')
    parser.add_argument('--output', default=None,
                        help='Destination for selected gains and RMSE history')
    args = parser.parse_args(argv)
    if min(args.passes, args.episodes, args.patience) < 1:
        parser.error('passes, episodes and patience must be positive')
    num_episodes = args.episodes
    SIMULATION_MODEL = args.model
    if args.output is None:
        args.output = f'ddpg_smc_amir_{args.model}_results.npz'
    print(f'Simulation model: {SIMULATION_MODEL}')
    print(f'DDPG initialization: {args.init}')
    metric_name = ('3D position RMSE' if SIMULATION_MODEL == 'comparison'
                   else 'pooled XYZ position RMSE')
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Running with fixed seed: {args.seed}")
    else:
        print("Running with stochastic (unseeded) initialization")
    print("Evaluating Hand-Picked Baseline on Full Square Trajectory...")
    hp_cost, _, hp_log, hp_ref, _ = run_quad_simulation(
        hp_lam_p, hp_kp_smc, hp_phi_p, hp_lam_a, hp_ka_smc, hp_phi_a, hp_kd_p,
        active_axis='xyz', traj_type='square', evaluate_reward=False
    )
    print(f"Baseline filtered {metric_name}: {hp_cost:.6f} m\n")
    
    if hp_cost >= 1e6:
        raise RuntimeError("Baseline leaves the flight envelope; repair it before tuning.")
    best_lam_p  = hp_lam_p.copy()
    best_kp_smc = hp_kp_smc.copy()
    best_phi_p  = hp_phi_p.copy()
    best_lam_a  = hp_lam_a.copy()
    best_ka_smc = hp_ka_smc.copy()
    best_phi_a  = hp_phi_a.copy()
    
    baseline_gains = tuple(a.copy() for a in (
        hp_lam_p, hp_kp_smc, hp_phi_p, hp_lam_a, hp_ka_smc, hp_phi_a))
    
    channels = [
        ('X',     'position', 0, [1.0, 1.5, 0.1], [3.5, 6.0, 1.0]),
        ('Y',     'position', 1, [0.9, 1.9, 0.1], [4.5, 6.0, 1.0]),
        ('Z',     'position', 2, [1.0, 1.0, 0.1], [7.0, 7.0, 1.0]),
        ('Roll',  'attitude', 0, [3, 10.0, 0.1], [10.0, 13.0, 0.72]),
        ('Pitch', 'attitude', 1, [3.0, 10, 0.10], [10.0, 13.0, 0.72]),
        ('Yaw',   'attitude', 2, [3.0, 9.0, 0.10], [11.0, 12.0, 0.72]),
    ]
    
    
    
    
    
    
    rmse_history = [hp_cost]
    for tuning_pass in range(args.passes):
        print(f"\n{'='*55}")
        print(f"{'='*55}")
        print(f"--- FINE-TUNING PASS {tuning_pass+1} / {args.passes} ---")
        print(f"{'='*55}")
        for name, loop_type, idx, g_mins, g_maxs in channels:
            
            init = [g_mins[i] + 0.25 * (g_maxs[i] - g_mins[i]) for i in range(3)]
    
            result = train_channel_3param(
                name, loop_type, idx, g_mins, g_maxs, init,
                patience=args.patience,
                cur_lam_p=best_lam_p.copy(),
                cur_kp_p=best_kp_smc.copy(),
                cur_phi_p=best_phi_p.copy(),
                cur_lam_a=best_lam_a.copy(),
                cur_ka_a=best_ka_smc.copy(),
                cur_phi_a=best_phi_a.copy(),
                initialize_from_baseline=(args.init == 'baseline'),
            )
    
            if loop_type == 'position':
                cand_lam_p = best_lam_p.copy(); cand_lam_p[idx] = result[0]
                cand_kp_p = best_kp_smc.copy(); cand_kp_p[idx] = result[1]
                cand_phi_p = best_phi_p.copy(); cand_phi_p[idx] = result[2]
                cand_lam_a, cand_ka_a, cand_phi_a = best_lam_a, best_ka_smc, best_phi_a
            else:
                cand_lam_a = best_lam_a.copy(); cand_lam_a[idx] = result[0]
                cand_ka_a = best_ka_smc.copy(); cand_ka_a[idx] = result[1]
                cand_phi_a = best_phi_a.copy(); cand_phi_a[idx] = result[2]
                cand_lam_p, cand_kp_p, cand_phi_p = best_lam_p, best_kp_smc, best_phi_p

            cand_square_cost, _, _, _, _ = run_quad_simulation(
                cand_lam_p, cand_kp_p, cand_phi_p,
                cand_lam_a, cand_ka_a, cand_phi_a,
                hp_kd_p, active_axis='xyz', traj_type='square', evaluate_reward=False
            )
            if loop_type == 'attitude' and idx == 2:
                cand_cost = cand_square_cost + YAW_TRAIN_WEIGHT * yaw_step_rmse(
                    cand_lam_p, cand_kp_p, cand_phi_p,
                    cand_lam_a, cand_ka_a, cand_phi_a, hp_kd_p)
                ref_cost = rmse_history[-1] + YAW_TRAIN_WEIGHT * yaw_step_rmse(
                    best_lam_p, best_kp_smc, best_phi_p,
                    best_lam_a, best_ka_smc, best_phi_a, hp_kd_p)
            else:
                cand_cost = cand_square_cost
                ref_cost = rmse_history[-1]
            if cand_cost < ref_cost - 1e-10:
                if loop_type == 'position':
                    best_lam_p[idx]  = result[0]
                    best_kp_smc[idx] = result[1]
                    best_phi_p[idx]  = result[2]
                else:
                    best_lam_a[idx]  = result[0]
                    best_ka_smc[idx] = result[1]
                    best_phi_a[idx]  = result[2]
                cur_cost = cand_cost
                print(f"  -> Full-system objective after {name}: {cur_cost:.6f}\n")
            else:
                cur_cost = rmse_history[-1]
                print(f"  -> Full-system objective after {name}: {cur_cost:.6f} (retained previous gains)\n")
            rmse_history.append(cur_cost)
        print(f"Pass {tuning_pass+1} retained RMSE: {cur_cost:.6f} m")
    
    
    

    final_lam_p = best_lam_p
    final_kp_p  = best_kp_smc
    final_phi_p = best_phi_p
    
    final_lam_a = best_lam_a
    final_ka_a  = best_ka_smc
    final_phi_a = best_phi_a
    
   
    eval_cost, _, eval_log, eval_ref, t_log = run_quad_simulation(
        final_lam_p, final_kp_p, final_phi_p, final_lam_a, final_ka_a, final_phi_a, hp_kd_p,
        active_axis='xyz', traj_type='square', evaluate_reward=False
    )
    baseline_rmse = np.sqrt(np.mean(np.sum((hp_log[:, :3] - hp_ref)**2, axis=1)))
    candidate_rmse = (np.sqrt(np.mean(np.sum((eval_log[:, :3] - eval_ref)**2, axis=1)))
                      if np.isfinite(eval_cost) else np.inf)
    if candidate_rmse > baseline_rmse + 1e-10:
        print("DDPG result did not improve the protected baseline; restoring hand-tuned gains.")
        (final_lam_p, final_kp_p, final_phi_p,
         final_lam_a, final_ka_a, final_phi_a) = tuple(a.copy() for a in baseline_gains)
        eval_cost, _, eval_log, eval_ref, t_log = run_quad_simulation(
            final_lam_p, final_kp_p, final_phi_p, final_lam_a, final_ka_a, final_phi_a, hp_kd_p,
            active_axis='xyz', traj_type='square', evaluate_reward=False
        )

    def fmt(arr):
        return ", ".join(f"{v:.17g}" for v in arr)
    
    print("\n" + "=" * 60)
    print(f"--- SELECTED FIXED GAINS FOR {SIMULATION_MODEL.upper()} MODEL ---")
    print("=" * 60)
    print("SMC1_GAINS = dict(")
    print(f"    lam_p  = np.array([{fmt(final_lam_p)}]),")
    print(f"    kp_smc = np.array([{fmt(final_kp_p)}]),")
    print(f"    phi_p  = np.array([{fmt(final_phi_p)}]),")
    print(f"    lam_a  = np.array([{fmt(final_lam_a)}]),")
    print(f"    ka_smc = np.array([{fmt(final_ka_a)}]),")
    print(f"    phi_a  = np.array([{fmt(final_phi_a)}]),")
    print(")")
    print("=" * 60)
    
    print(f"\nFinal filtered {metric_name}: {eval_cost:.6f} m vs baseline {hp_cost:.6f} m")
    print("Retained RMSE history (baseline, then each channel):", rmse_history)
    for label, positions in [('Baseline', hp_log), ('Optimized', eval_log)]:
        errors = positions[:, :3] - eval_ref
        axis_rmse = np.sqrt(np.mean(errors**2, axis=0))
        print(f"{label} filtered X/Y/Z RMSE (m): {axis_rmse}")
        print(f"{label} filtered 3D RMSE (m): {np.sqrt(np.sum(axis_rmse**2)):.6f}")
        raw_ref = np.array([step_reference(t, TRAJ_SQUARE, -1)[:3] for t in t_log])
        print(f"{label} overall raw-reference RMSE (m): {np.sqrt(np.mean((positions[:, :3]-raw_ref)**2)):.6f}")
        ref_vel = np.gradient(eval_ref, t_log, axis=0)
        state_vel = positions[:, 3:6]
        overshoot_mask = ((np.abs(ref_vel) < OVERSHOOT_VEL_THRESHOLD)
                          & ((errors * state_vel) > 0.0))
        overshoot_rmse = np.sqrt(np.mean(errors[overshoot_mask]**2)) if np.any(overshoot_mask) else 0.0
        print(f"{label} corner overshoot RMSE (m): {overshoot_rmse:.6f}")
    
    for label, gains in [('Baseline', (hp_lam_p, hp_kp_smc, hp_phi_p, hp_lam_a, hp_ka_smc, hp_phi_a)),
                         ('Optimized', (final_lam_p, final_kp_p, final_phi_p, final_lam_a, final_ka_a, final_phi_a))]:
        check = run_quad_simulation(*gains, hp_kd_p, active_axis='yaw', traj_type='single_step')
        if check[0] >= 1e6:
            print(f"{label} yaw-step validation: FAILED flight envelope")
        else:
            yaw_ref = np.array([step_reference(t, TRAJ_SINGLE_STEP, 3)[3] for t in check[4]])
            yaw_error = (check[2][:, 8]-yaw_ref+np.pi) % (2*np.pi)-np.pi
            print(f"{label} independent raw yaw-step RMSE: {np.sqrt(np.mean(yaw_error**2)):.6f} rad")
    np.savez(args.output, lam_p=final_lam_p, kp_smc=final_kp_p, phi_p=final_phi_p,
             lam_a=final_lam_a, ka_smc=final_ka_a, phi_a=final_phi_a,
             rmse_history=np.asarray(rmse_history), seed=args.seed,
             passes=args.passes, max_episodes_per_channel=args.episodes,
             initialization=args.init,
             selection_metric=('3D_filtered_reference_position_RMSE_m'
                               if SIMULATION_MODEL == 'comparison' else
                               'pooled_filtered_reference_position_RMSE_m'),
             simulation_model=SIMULATION_MODEL)
    from pathlib import Path
    gain_path = Path(args.output).with_suffix('.gains.py')
    gain_names = ('lam_p', 'kp_smc', 'phi_p', 'lam_a', 'ka_smc', 'phi_a')
    gain_values = (final_lam_p, final_kp_p, final_phi_p, final_lam_a, final_ka_a, final_phi_a)
    gain_path.write_text(f'# Model: {SIMULATION_MODEL}; fixed gains, seed {args.seed}\n'
                         + 'import numpy as np\n\nSMC1_GAINS = dict(\n'
                         + ''.join(f'    {name} = np.array([{fmt(values)}]),\n'
                                   for name, values in zip(gain_names, gain_values))
                         + ')\n', encoding='utf-8')
    print(f'Copy the SMC1_GAINS block from {gain_path}')
    print(f"Saved selected gains and RMSE history to {args.output}")
    if args.no_plots:
        return rmse_history
    
    
   
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8))
    fig1.suptitle(f'Square Trajectory Evaluation - Position Loops (filtered RMSE: {eval_cost:.4f} m)', fontsize=12)
    pos_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
    for i in range(3):
        axes1[i].plot(t_log, eval_ref[:, i], 'k--', label='Filtered reference', alpha=0.7)
        axes1[i].plot(t_log, hp_log[:, i], 'b-', label=f'Baseline ({hp_cost:.6f} m)', linewidth=1.2)
        axes1[i].plot(t_log, eval_log[:, i], 'r-', label=f'DDPG-Optimized ({eval_cost:.6f} m)', linewidth=1.5)
        axes1[i].set_ylabel(pos_labels[i])
        axes1[i].grid(True)
        if i == 0: axes1[i].legend(loc='upper right', fontsize=8)
    axes1[2].set_xlabel('Time (s)')
    fig1.tight_layout()
    plt.show()

    
    baseline_errors = hp_log[:, :3] - hp_ref
    optimized_errors = eval_log[:, :3] - eval_ref
    fig_err, axes_err = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig_err.suptitle('Position Tracking Errors — Baseline vs DDPG-Optimized', fontsize=12)
    error_labels = ['X error (m)', 'Y error (m)', 'Z error (m)']
    for i in range(3):
        axes_err[i].plot(t_log, baseline_errors[:, i], 'b-',
                         label='Baseline', linewidth=1.2)
        axes_err[i].plot(t_log, optimized_errors[:, i], 'r-',
                         label='DDPG-Optimized', linewidth=1.5)
        axes_err[i].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
        axes_err[i].set_ylabel(error_labels[i])
        axes_err[i].grid(True)
        if i == 0:
            axes_err[i].legend(loc='upper right', fontsize=8)
    axes_err[2].set_xlabel('Time (s)')
    fig_err.tight_layout()
    plt.show()
    
    
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8))
    fig2.suptitle('Inner Attitude Loop Evaluation (Roll, Pitch, Yaw Angles)', fontsize=12)
    att_labels = ['Roll Angle (rad)', 'Pitch Angle (rad)', 'Yaw Angle (rad)']
    att_indices = [6, 7, 8]
    for i in range(3):
        axes2[i].plot(t_log, hp_log[:, att_indices[i]], 'b-', label='Baseline', linewidth=1.2)
        axes2[i].plot(t_log, eval_log[:, att_indices[i]], 'r-', label='DDPG-Optimized', linewidth=1.5)
        axes2[i].set_ylabel(att_labels[i])
        axes2[i].grid(True)
        if i == 0: axes2[i].legend(loc='upper right', fontsize=8)
    axes2[2].set_xlabel('Time (s)')
    fig2.tight_layout()
    # %%
    plt.show()
    
    return rmse_history


if __name__ == "__main__":
    main()
