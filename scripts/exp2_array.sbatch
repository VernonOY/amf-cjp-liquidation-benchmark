#!/bin/bash
#SBATCH --job-name=amf-exp2
#SBATCH --output=logs/exp2_%A_%a.out
#SBATCH --error=logs/exp2_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-4

# Exp 2: CIR FD over 5 sigma_lam values.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

SIGMAS=(0.0 0.05 0.1 0.2 0.5)
SIG=${SIGMAS[$SLURM_ARRAY_TASK_ID]}
python3 -c "
from src.experiments.exp2_stochastic_lambda import fd_cir_surface
import csv, pathlib
rows = fd_cir_surface([float('$SIG')], n_lam=50, dt=0.05)
out = pathlib.Path(f'data/exp2/cells/sig_{$SIG}.csv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    w.writeheader()
    [w.writerow(r) for r in rows]
"
