# 3-D variable-gain DDPG-SMC

This folder contains the full-position adaptive controller. DDPG produces
bounded, smoothed residuals for the nine position gains

`[lambda_x, K_x, delta_x, lambda_y, K_y, delta_y, lambda_z, K_z, delta_z]`.

Roll, pitch, and yaw gains remain fixed at the values used in
`ddpg_smc_amir.py`. The plant uses a randomly sampled mass in the range
`0.076--0.084 kg` during training while the controller uses the nominal mass
`0.080 kg`.

The residual limits are ±30% for the three lambda gains and ±15% for each K
and delta gain. Residuals are smoothed at the 20-Hz policy update rate. The
hover test retains the fixed position gains; DDPG adaptation is enabled for
the square manoeuvre, where the horizontal transients provide a meaningful
gain-tuning objective. Rewards use the post-integration tracking error, the
same error used in the reported RMSE.

Run a short smoke test from the project root:

```powershell
python online_xyz_ddpg/online_xyz_ddpg.py --episodes 2 --train-scenario both --output online_xyz_ddpg/smoke
```

Run a full training experiment:

```powershell
python online_xyz_ddpg/online_xyz_ddpg.py --episodes 60 --train-scenario both --output online_xyz_ddpg/results
```

The controller saves adaptive and fixed-gain `.npz` trajectories, plots,
weights, training history, and an evaluation summary. Create additional
publication figures with:

```powershell
python online_xyz_ddpg/plot_xyz_ddpg_comparison.py `
  --results online_xyz_ddpg/results `
  --output online_xyz_ddpg/results/plots
```

For the validated run used in this folder, `results_final_aw` also contains
`mass_sweep.json`, which evaluates both controllers at 0.076, 0.080, and
0.084 kg.
