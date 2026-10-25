# analysis (CPU)

From the per-image statistics of the GPU runs (`inputs/`, release asset `inputs-v1`) to the result files in `../results/`.
No network is evaluated here. Every script runs from anywhere (`python analysis/<script>.py`); `bash reproduce.sh` runs them
all in this order (with `protocol/history/explore_10nfe.py`, 7 min, and `paper/gen/make_paper.py`, 15 s; 19 min in all) and
compares the outputs with the committed files. `$EUPHONIUM_INPUTS` overrides the location of `inputs/`.

| script | writes | what | time (2 CPU cores) |
|---|---|---|---|
| `baselines_eval.py` | `results/baselines_pairs.json` | the eight comparison methods (NLL, IC-PNG, IC-J2K, TT, MSMA-type, DiffPath, GEPC, EigenScore) on the 63 pairs; every detector is fitted on ID development images only | 2 min |
| `exact_ratio.py` | `results/exact_ratio_pairs.json` | Ablation: the exact ratio Λ90 by the PF-ODE (four IDs) | 1 s |
| `select_rule.py` | `results/selection_10nfe.json` | **the selection of the Euphonium rule** (1285 candidates, selection data only, tie rule) and the check that the winner is `rule.json` | 45 s |
| `evaluate.py` | `results/euphonium_pairs.json` | **Euphonium on the 63 pairs** (main evaluation) and the comparison with the eight methods | 5 s |
| `ablations/grid_candidates.py` | `results/ablations/grid_selection.json` | Ablation: the grid candidates c1–c12, r60, r80 (one scale set for every rotation; c3 is the earlier 20-pass version) | 6 s |
| `ablations/frontier.py` | `results/ablations/frontier.json` | Ablation: budgets of 2–10 passes, the difference form, the scale window σ1–σ13, single noise | 8 min |
| `ablations/robustness.py` | `results/ablations/robustness.json` | checks of the selection: ranking, split halves of the selection OOD sets, noise seeds, per-pair AUROCs | 1 min |

`common.py` defines the benchmark: the six IDs with their models (`TIERS`), the OOD sets of the main evaluation (63 pairs;
"natural" = without the upside-down ID images and the synthetic sets), the selection OOD sets (`POOL`, eleven per ID, never
part of the main evaluation), the run names, and the loaders and metrics (AUROC by ranks, FPR at 95% TPR; ID = positive
class, higher score = more in-distribution).

Which images: the tables compare all methods on the images every method has (`benchmark/common_idx.json`: 1000 per set,
500 for ImageNet-64, 2000 for the synthetic sets); the selection uses the ID development images (runs `B_`, `P_`, `I2_`) and
the selection OOD sets (final-split images of the `F_` runs). `protocol/PROTOCOL_10NFE.md` describes the selection and
`protocol/history/explore_10nfe.py` the exploratory search before it.

Euphonium is computed here from the full-grid runs (`F_<tag>`: 13 scales, six transforms) by taking its three atoms. The
10-pass implementation (`euphonium/score.py`) evaluates exactly these quantities with the same noise images:
`experiments/check_euphonium.py` compares the two.
