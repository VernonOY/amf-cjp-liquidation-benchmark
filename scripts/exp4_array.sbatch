#!/bin/bash
#SBATCH --job-name=amf-exp4
#SBATCH --output=logs/exp4_%A_%a.out
#SBATCH --error=logs/exp4_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-0

# Failure-mode diagnostics: single-job (panels are small and independent).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

python3 -m src.experiments.exp4_failure_modes
