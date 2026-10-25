# results

Result files of the analysis (`analysis/`), read by `paper/gen/make_paper.py`. AUROC and FPR95 are in %, ID = positive class;
"best" = Euphonium ties (within 0.05 points) or beats the best of the eight comparison methods on that pair.

| file | written by | contents |
|---|---|---|
| `euphonium_pairs.json` | `analysis/evaluate.py` | **Euphonium** (the rule of `rule.json`) on the 63 pairs: `pairs[tag][set]` = AUROC, FPR95, image counts, best comparison method, best / second; `ids[tag]` = mean over the natural OOD sets; `summary` (mean over the six IDs 86.9, best on 34 of 63 pairs, second on 8, best on 4 of 6 IDs) |
| `baselines_pairs.json` | `analysis/baselines_eval.py` | the eight comparison methods on the same images: `[tag]['common'][method][set]` |
| `selection_10nfe.json` | `analysis/select_rule.py` | the selection of the rule: all 1285 candidates with their per-ID selection AUROCs, the tie rule, the three tied candidates and the selected one |
| `exact_ratio_pairs.json` | `analysis/exact_ratio.py` | Ablation: the exact ratio Λ90 (PF-ODE), four IDs |
| `ablations/grid_selection.json` | `analysis/ablations/grid_candidates.py` | Ablation: grid candidates c1–c12 and the ten-scale configurations, selection data |
| `ablations/frontier.json` | `analysis/ablations/frontier.py` | Ablation: best atom sets per budget (2–10 passes) in the log-ratio and the difference form, the scale window, single noise, Euphonium's atoms in the difference form |
| `ablations/robustness.json` | `analysis/ablations/robustness.py` | checks of the selection: ranking of the pure-pair candidates, split halves, noise seeds, per-pair AUROCs |
| `bench_cpu.jsonl` | `experiments/bench_cpu.py` | Table A2: CPU seconds and network evaluations per image of every method, one session on 2 CPU cores |
| `external/knn_feature.json` | – | reference value quoted in Section 4: feature kNN of the same models (an experiment of the companion final-presentation resume, not reproduced here; see its `_note`) |
