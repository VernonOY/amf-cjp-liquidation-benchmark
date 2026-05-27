#!/usr/bin/env bash
# AMF revision: incremental reproduction driver.
#
# Targets:
#   phase1      Phase 0/1 golden-lock + stats/estimators tests (~10 s)
#   exp1        Sample-complexity sweep (Fig 9 + Tab 5)
#   exp1_r2     Checkpointed Regime-II sample-complexity stress test
#   exp2        CIR FD + validation (Figs 11, 12, validation)
#   exp3        Univariate + bivariate + misspec robustness (Figs 15-16, Tab 6)
#   exp4        Failure-mode diagnostics (Fig 10)
#   exp4_fair   Forced-MO fairness ablation (Table in §5.4)
#   exp5        Regime-II extended baseline panel
#   legacy      Original course-project pipeline (still works via shims)
#   all         Run phase1 + exp1 + exp2 + exp3 + exp4 + exp4_fair + exp5
#
# Options:
#   --smoke     Run a tiny sweep that completes in <2 min, for CI sanity
#   --force    Re-run even if cached CSV exists
#   --slurm    Print Slurm-array job commands instead of running them
#
# Usage:
#   bash run_all.sh phase1
#   bash run_all.sh exp1 --smoke
#   bash run_all.sh all

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

target="${1:-help}"
shift || true

SMOKE=0
FORCE=0
SLURM=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --force) FORCE=1 ;;
        --slurm) SLURM=1 ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
run_phase1() {
    echo "[phase1] running golden-lock and stats/estimators tests"
    "$PYTHON" -m pytest tests/ -q -m "not slow"
}

run_exp1() {
    echo "[exp1] sample-complexity sweep"
    args=()
    if [[ $SMOKE -eq 1 ]]; then args+=(--smoke); fi
    "$PYTHON" -m src.experiments.exp1_sample_complexity "${args[@]}"
}

run_exp1_r2() {
    echo "[exp1_r2] checkpointed Regime-II sample-complexity sweep"
    if [[ $SMOKE -eq 1 ]]; then
        "$PYTHON" -m src.experiments.exp1_regime_cached \
            --regime ii --agents D_plugin,A_tabular --budgets 10000 \
            --seed-count 2 --n-eval-paths 50 \
            --cache-dir data/cache/exp1_regime_ii_smoke \
            --out data/smoke/regime_ii_cached_smoke.csv \
            --fig figures/fig_regime_ii_cached_smoke.pdf
    else
        "$PYTHON" -m src.experiments.exp1_regime_cached \
            --regime ii --seed-count 20 --n-eval-paths 500 \
            --cache-dir data/cache/exp1_regime_ii \
            --out data/exp1/regime_ii_sample_complexity_full.csv \
            --fig figures/fig_regime_ii_sample_complexity.pdf
    fi
}

run_exp2() {
    echo "[exp2] CIR FD + validation"
    "$PYTHON" -m src.experiments.exp2_stochastic_lambda
}

run_exp3() {
    echo "[exp3] robustness sweeps"
    "$PYTHON" -m src.experiments.exp3_robustness
}

run_exp4() {
    echo "[exp4] failure-mode diagnostics"
    "$PYTHON" -m src.experiments.exp4_failure_modes
}

run_exp4_fair() {
    echo "[exp4_fair] forced-MO fairness ablation"
    if [[ $SMOKE -eq 1 ]]; then
        "$PYTHON" -m src.experiments.exp4_failure_modes \
            --fairness-only --seed-count 3 --n-eval-paths 200 \
            --out data/smoke/forced_mo_fairness_cli_smoke.csv
    else
        "$PYTHON" -m src.experiments.exp4_failure_modes \
            --fairness-only --full --seed-count 20 --n-eval-paths 500 \
            --out data/exp4/forced_mo_fairness_full.csv
    fi
}

run_exp5() {
    echo "[exp5] Regime-II extended baselines"
    n_paths=2000
    if [[ $SMOKE -eq 1 ]]; then n_paths=200; fi
    "$PYTHON" -m src.experiments.exp5_regime_baselines \
        --regime ii --n-paths "$n_paths" \
        --out data/exp5/regime_ii_baselines.csv
}

run_legacy() {
    echo "[legacy] course-project pipeline (uses deprecation shims)"
    "$PYTHON" -m src.task1_lo_only.simulate --n-paths 10000 --dt 0.05 --seed 0
    "$PYTHON" -m src.task1_lo_only.figures
    "$PYTHON" -m src.task2_lo_mo.simulate --n-paths 10000 --sim-dt 0.05 --fd-dt 0.01 --seed 0
    "$PYTHON" -m src.task2_lo_mo.figures
    "$PYTHON" -m src.task3_rl.train --method tabular --tabular-episodes 80000 --dt 1.0 --seed 0
    "$PYTHON" -m src.task3_rl.train --method ddqn --ddqn-episodes 6000 --dt 1.0 --seed 0
    "$PYTHON" -m src.task3_rl.eval
}

case "$target" in
    phase1) run_phase1 ;;
    exp1)   run_exp1   ;;
    exp1_r2) run_exp1_r2 ;;
    exp2)   run_exp2   ;;
    exp3)   run_exp3   ;;
    exp4)   run_exp4   ;;
    exp4_fair) run_exp4_fair ;;
    exp5)   run_exp5 ;;
    legacy) run_legacy ;;
    all)
        run_phase1
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        run_exp4_fair
        run_exp5
        ;;
    help|*)
        cat <<'EOF'
AMF revision reproduction driver. Usage:
    bash run_all.sh phase1
    bash run_all.sh exp1 [--smoke]
    bash run_all.sh exp1_r2 [--smoke]
    bash run_all.sh exp2
    bash run_all.sh exp3
    bash run_all.sh exp4
    bash run_all.sh exp4_fair [--smoke]
    bash run_all.sh exp5 [--smoke]
    bash run_all.sh legacy
    bash run_all.sh all
EOF
        ;;
esac
