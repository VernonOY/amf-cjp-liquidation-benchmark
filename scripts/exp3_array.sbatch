#!/bin/bash
#SBATCH --job-name=amf-exp3
#SBATCH --output=logs/exp3_%A_%a.out
#SBATCH --error=logs/exp3_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-2

# Exp 3 has three regimes (univariate, bivariate, misspec).
# Allocate one job per regime so they can run in parallel.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

case $SLURM_ARRAY_TASK_ID in
  0) python3 -c "from src.experiments.exp3_robustness import univariate_sensitivity; import json, pathlib;
rows = univariate_sensitivity(n_paths=2000)
out = pathlib.Path('data/exp3/univariate.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))" ;;
  1) python3 -c "from src.experiments.exp3_robustness import bivariate_sensitivity; import json, pathlib;
rows = bivariate_sensitivity(n_paths=1000)
out = pathlib.Path('data/exp3/bivariate.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))" ;;
  2) python3 -c "from src.experiments.exp3_robustness import misspecification_grid; import json, pathlib;
rows = misspecification_grid(n_paths=1000)
out = pathlib.Path('data/exp3/misspec.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))" ;;
esac
