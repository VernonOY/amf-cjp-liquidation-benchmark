# Forced-MO fairness ablation


20 seeds, 500 Monte-Carlo evaluation paths per seed, Regime I, n=100000 equivalent budget (826 training episodes).

| Agent | Exploration | Premium vs TWAP | Seed s.e. | Clearance | MO/path |
|---|---|---:|---:|---:|---:|
| Tabular Q | baseline | +0.140 | 0.004 | 0.969 | 0.055 |
| Tabular Q | forced MO 5% | +0.135 | 0.004 | 1.000 | 0.119 |
| DDQN | baseline | +0.133 | 0.012 | 0.540 | 0.028 |
| DDQN | forced MO 5% | +0.136 | 0.010 | 0.466 | 0.350 |
