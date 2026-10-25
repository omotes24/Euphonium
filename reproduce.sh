#!/bin/bash
# reproduce.sh -- recomputes every result file and every generated table, figure and macro of the paper from the per-image
#   statistics of the GPU runs (inputs/, release asset inputs-v1: README.md) and compares them with the committed copies.
#   CPU only (environment/requirements-analysis.txt).  The recomputed files replace the committed ones in place; the committed
#   copies are kept in a temporary folder for the comparison.   usage: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
INP=${EUPHONIUM_INPUTS:-inputs}
[ -d "$INP/feat" ] || { echo "no per-image statistics in $INP/ (README.md: release asset inputs-v1)"; exit 1; }
KEEP=$(mktemp -d)
mkdir -p "$KEEP/protocol/history" "$KEEP/paper/figs" logs
cp -r results "$KEEP/"; cp protocol/history/explore_10nfe.json "$KEEP/protocol/history/"
cp paper/*.tex "$KEEP/paper/"; cp paper/figs/fig_*.pdf "$KEEP/paper/figs/"
run() {
  local t0=$SECONDS log="logs/reproduce_$(basename "$1" .py).log"
  python "$1" > "$log" 2>&1 || { echo "$1 failed, see $log"; exit 1; }
  printf '%-40s %5d s\n' "$1" $((SECONDS - t0))
}
run analysis/baselines_eval.py              # the eight comparison methods
run analysis/exact_ratio.py                 # Ablation: exact ratio (PF-ODE)
run analysis/select_rule.py                 # the selection of the Euphonium rule (checks rule.json)
run analysis/evaluate.py                    # Euphonium on the 63 pairs
run analysis/ablations/grid_candidates.py   # Ablations: grid candidates
run analysis/ablations/frontier.py          # Ablations: budgets, difference form, scale window, single noise
run analysis/ablations/robustness.py        # checks of the selection
run protocol/history/explore_10nfe.py       # the exploratory search before the protocol
run paper/gen/make_paper.py                 # tables, figures and macros of the paper
python analysis/compare_outputs.py "$KEEP" .
echo "committed copies: $KEEP;  typeset the paper with: bash paper/build.sh"
