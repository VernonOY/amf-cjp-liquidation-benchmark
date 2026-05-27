# Sample Complexity of Calibration versus Model-Free Learning in CJP Optimal Liquidation

This repository contains the code, data, figures, and LaTeX source accompanying the Applied Mathematical Finance submission:

**Sample Complexity of Calibration versus Model-Free Learning in Cartea--Jaimungal--Penalva Optimal Liquidation**

Wendi Ouyang

The project benchmarks model-free reinforcement learning, structure-aware RL, and a plug-in parametric model-based calibrator in the Cartea--Jaimungal--Penalva limit-order/market-order liquidation model. It includes finite-difference QVI solvers, a CIR stochastic-intensity extension, sample-complexity experiments, robustness sweeps, and the manuscript source.

## Repository Layout

```text
src/                 Core Python package
  common/            Parameters, simulator, statistics, estimators
  analytical/        Closed-form / analytical benchmark code
  numerical/         Constant- and stochastic-intensity finite-difference solvers
  rl/                Tabular Q, DDQN, Hybrid PPO, Plug-in MLE agents
  baselines/         TWAP, Almgren--Chriss, VWAP, POV, passive/aggressive policies
  experiments/       Experiment entry points used in the paper
scripts/             Figure and table rendering utilities
tests/               Unit tests and golden-snapshot regression tests
data/                Stored CSV intermediates used for the reported tables/figures
figures/             Generated paper figures
paper/               LaTeX manuscript, references, tables, and compiled PDF
configs/             Small experiment configuration files
results/             Lightweight text summaries
```

Large local caches, logs, course-project artifacts, and private submission-policy screenshots are intentionally excluded from version control.

## Reproducing the Results

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the fast verification suite:

```bash
bash run_all.sh phase1
```

Run the full cached/reproducibility pipeline:

```bash
bash run_all.sh all
```

Selected experiment targets can be run independently:

```bash
bash run_all.sh exp1       # Regime-I sample-complexity sweep
bash run_all.sh exp1_r2    # Regime-II sample-complexity stress test
bash run_all.sh exp2       # CIR stochastic-intensity validation
bash run_all.sh exp3       # Sensitivity and misspecification sweeps
bash run_all.sh exp4       # Failure-mode diagnostics
bash run_all.sh exp4_fair  # Forced-MO fairness ablation
bash run_all.sh exp5       # Regime-II baseline panel
```

Some targets accept `--smoke` for a small sanity-check run, for example:

```bash
bash run_all.sh exp1 --smoke
bash run_all.sh exp1_r2 --smoke
```

Render all paper figures from stored CSV intermediates:

```bash
python3 scripts/render_all_figures.py
```

Build the manuscript:

```bash
cd paper
tectonic -X compile main.tex
```

## Tests

```bash
python3 -m pytest tests/ -q -m "not slow"
```

The tests cover finite-difference solver behavior, analytical benchmarks, MLE estimators, bootstrap/statistical utilities, RL environment stepping, and migration shims for legacy import paths.

## Data Availability

The stored CSV intermediates under `data/` reproduce the figures and tables in the submitted manuscript. Random seeds are fixed in the experiment scripts. The repository is intended as the review-accessible project repository referenced in the manuscript's Data Availability Statement.

## Citation

If you use this repository, please cite the accompanying manuscript. A machine-readable citation file is provided in `CITATION.cff`.
