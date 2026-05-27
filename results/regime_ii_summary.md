# Regime II summary


Sample-complexity endpoint uses 20 seeds and 500 MC eval paths per seed. Baseline panel uses 2000 paired MC paths.

## Agent endpoint summary

| Agent | Premium @ 1e5 | Premium @ 1e6 | Clearance @ 1e6 | Value RMSE @ 1e6 | Wall @ 1e6 |
|---|---:|---:|---:|---:|---:|
| A. Tabular Q | +0.963 | +0.998 ± 0.565 | 1.000 | 0.450 | 4.7s |
| B. DDQN | +1.171 | +1.254 ± 0.640 | 0.432 | 1.974 | 728.7s |
| C. Hybrid PPO | +1.667 | +1.413 ± 0.668 | 0.307 | 0.707 | 381.2s |
| D. Plug-in MLE | +1.312 | +1.312 ± 0.396 | 0.856 | 0.014 | 0.6s |

## Extended baselines

| Strategy | Premium | 95% CI | Clearance | MO/path | LO/path |
|---|---:|---:|---:|---:|---:|
| TWAP | +0.000 | [+0.000, +0.000] | 1.000 | 100.00 | 0.00 |
| Almgren-Chriss | -0.576 | [-1.221, +0.092] | 1.000 | 100.00 | 0.00 |
| VWAP | -0.907 | [-1.759, -0.025] | 1.000 | 100.00 | 0.00 |
| POV 10pct | -0.499 | [-1.512, +0.557] | 1.000 | 50.00 | 0.00 |
| Pure passive | +0.785 | [-0.228, +1.751] | 0.000 | 0.00 | 67.38 |
| Pure aggressive | -0.495 | [-1.135, +0.139] | 1.000 | 92.70 | 7.30 |
| FD Optimal | +0.975 | [+0.273, +1.631] | 0.857 | 30.64 | 69.20 |
| Plug-in MLE (1000 fills) | +0.958 | [+0.239, +1.638] | 0.862 | 29.74 | 70.11 |
