#!/bin/bash
#SBATCH --job-name=amf-exp1
#SBATCH --output=logs/exp1_%A_%a.out
#SBATCH --error=logs/exp1_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-559

# Slurm-array driver for Experiment 1 (sample complexity).
# 7 sample sizes x 20 seeds x 4 agents = 560 cells.
# Each task handles one (n, seed, agent) cell.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Decode SLURM_ARRAY_TASK_ID into (agent_idx, n_idx, seed)
AGENTS=("A_tabular" "B_ddqn" "C_hybrid" "D_plugin")
N_LEVELS=(100 316 1000 3162 10000 31623 100000)
N_SEEDS=20

idx=${SLURM_ARRAY_TASK_ID}
agent_idx=$(( idx / (${#N_LEVELS[@]} * N_SEEDS) ))
remaining=$(( idx % (${#N_LEVELS[@]} * N_SEEDS) ))
n_idx=$(( remaining / N_SEEDS ))
seed=$(( remaining % N_SEEDS ))

AGENT=${AGENTS[$agent_idx]}
N=${N_LEVELS[$n_idx]}

echo "Running exp1 cell: agent=$AGENT n=$N seed=$seed"

python3 -c "
from src.experiments.exp1_sample_complexity import _one_run, _fd_reference
from src.common.params import TASK2
import json
fd = _fd_reference(TASK2)
row = _one_run('$AGENT', $N, $seed, TASK2, fd, n_eval_paths=2000, eval_seed_offset=20000)
import pathlib
out = pathlib.Path('data/exp1/cells/' + f'{row[\"agent\"]}_n{row[\"n\"]}_s{row[\"seed\"]}.json')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as fp:
    json.dump(row, fp, indent=2)
"
