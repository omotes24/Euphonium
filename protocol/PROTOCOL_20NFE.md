# Ablation: the grid candidates with at most 20 forward passes per image (no main-evaluation data). Fixed 2026-10-23 16:21 UTC

> These candidates (c1--c12, together with r60, r80 and dev) are **not** Euphonium: the paper reports them as **Ablations**
> (lower block of Table A3, and the c3 row of Table A2). Euphonium's rule (at most 10 passes, atom sets) is in
> [`PROTOCOL_10NFE.md`](PROTOCOL_10NFE.md). This protocol was fixed before the single-noise candidates were computed and before
> the main evaluation of the selected candidate. `analysis/ablations/grid_candidates.py` recomputes the selection values.

## Aim

Choose one configuration of the orientation statistic (rotations, scale set, noise draws) computable with at most 20 network
evaluations per image, without looking at the OOD datasets of the main evaluation (six ID datasets x 63 pairs). An earlier
20-pass configuration (R180, {sigma_1, sigma_3, sigma_4, sigma_6, sigma_8}) had been chosen by comparing candidates on
development copies of the main OOD datasets; it is not eligible (it is the reference row "dev" of Table A3).

## Candidates (all with shared noise on the grid sigma_k = 0.01 x 3162.3^((k-1)/15), k = 1..10; "pair" = antithetic pair (+eps, -eps), "single" = +eps only)

| name | rotations | scales | noise | NFE |
|---|---|---|---|---|
| c1 | R180 | odd {1,3,5,7,9} | pair | 20 |
| c2 | R180 | even {2,4,6,8,10} | pair | 20 |
| c3 | R90 | odd | pair | 20 |
| c4 | R90 | even | pair | 20 |
| c5 | R90+R180 | {1,5,9} | pair | 18 |
| c6 | R90+R180 | {2,6,10} | pair | 18 |
| c7 | R180 | all ten | single | 20 |
| c8 | R90 | all ten | single | 20 |
| c9 | R90+R180 | odd | single | 15 |
| c10 | R90+R180 | even | single | 15 |
| c11 | R90+R180+R270 | odd | single | 20 |
| c12 | R90+R180+R270 | even | single | 20 |

(odd / even: every other grid point; {1,5,9}, {2,6,10}: every fourth. Rotations: R90, R180 and both, which carry the evidence
of R270 and of the vertical flip when the ID distribution is left-right symmetric; R270 is added for comparison.)

## Selection data (not part of the main evaluation)

- ID side: the development images of every ID dataset (images other than the final-split images of the main evaluation; the
  same split as used to fit the comparison methods).
- OOD side: OOD datasets that are not among the 63 pairs ("extra OOD sets"):
  the four 32x32 models: STL-10, GTSRB, Flowers-102, EuroSAT, MNIST (RGB), Oxford-IIIT Pet, Caltech-101, Food-101,
  FGVC-Aircraft, Imagenette, Rendered SST-2; ImageNet-64: the 64x64 versions of the same sets (without Imagenette; ten sets);
  FashionMNIST: EMNIST letters, Kuzushiji-49, SEMEION, CIFAR-10 (grey), SVHN (grey), GTSRB (grey).
- Criterion: per candidate, the mean AUROC over the extra OOD sets of every ID, averaged over the six IDs; the largest is
  chosen; ties within 0.1 points go to the candidate with fewer passes.

## Procedure

1. Pair candidates from the existing runs (ID development images B_/P_/I2_, extra OOD sets F_).
2. Single candidates from new runs on the GPU server (ID development images and extra OOD sets; one superset run with four
   rotations x ten scales x a single draw, from which every subset is taken; job list `experiments/jobs/N20.txt`).
3. Choose one by the criterion. The main evaluation (63 pairs) of a single candidate would need one more GPU run; for a pair
   candidate it comes from the existing final-split runs.
4. Report every candidate's value on the selection data and on the main evaluation (afterwards) in the supplementary material.

## Note

- Among the subsets of the ten-point grid, c1, c2 (and the 60-, 40- and 30-pass configurations) had been seen on the main
  evaluation before this procedure was fixed. The procedure was applied to them mechanically all the same, and its result was
  to be adopted whatever it was.
- Result (selection data): c3 83.90 (best), c9 83.72, c5 83.67, c11 83.58 (`results/ablations/grid_selection.json`).
