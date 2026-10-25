# How the Euphonium rule was selected (at most 10 forward passes per image; no main-evaluation data). 2026-10-25

This is the protocol behind `rule.json`. `analysis/select_rule.py` recomputes the selection from the per-image statistics and
checks that it returns the atoms of `rule.json`.

## History (stated plainly)

The grid candidates of [`PROTOCOL_20NFE.md`](PROTOCOL_20NFE.md) (at most 20 passes, one scale set shared by every rotation) had
been selected and evaluated on the main evaluation (63 pairs) when the question arose whether **10 forward passes** could do as
well or better. This search therefore took place **after** the main evaluation of the grid candidates had been seen. The
candidates were nevertheless compared on the same **selection data** as the grid candidates only (ID development images vs
extra OOD sets that are not part of the main evaluation); the main evaluation (final-split images and the OOD sets of the 63
pairs) was not used to choose among them. The main evaluation of the selected rule was computed after the choice.

## Candidates

- Atom = (rotation T, scale k) with the log-ratio log e_k(Tx)/e_k(x). Candidate = a set of atoms (their sum). Unlike the grid
  candidates, every rotation may use its own scales.
- Noise: antithetic pair only. Pair atoms of the development images, of the extra OOD sets and of the final split all come from
  runs with the same noise images, whereas candidates with +eps-only atoms (or mixtures of both) would add values computed with
  different noise images. Single-noise candidates were evaluated in the exploratory search (`history/explore_10nfe.py`) but are
  not eligible.
- Forward passes per image: NFE = 2 x (number of scales used by the atoms + number of atoms) (the upright image is evaluated at
  +eps and -eps once per scale, every atom's rotated image twice).
- T in {R90, R180, R270}, k in {sigma_1, ..., sigma_10}, NFE <= 10 (which allows one or two scales and one to three atoms), all
  of them: **1285 candidates** (70 with one scale, 405 + 810 with two scales and two or three atoms).

## Selection data and criterion (as for the grid candidates)

- ID side: the development images of every ID dataset (runs B_edmc10, P_l6svhn, P_dpceleba, P_edmc100, I2_edm2xs, P_fmnist; all
  their images).
- OOD side: the extra OOD sets of every ID (the eleven sets of `PROTOCOL_20NFE.md`; ImageNet-64: ten, FashionMNIST: six).
- Criterion: per ID, the mean AUROC over its extra OOD sets; the mean over the six IDs is maximised. Candidates within 0.1 points
  of the best are tied; among tied candidates the one with fewer forward passes is taken, and among those the one with the
  largest worst-ID mean (minimum over the six IDs).

## Result

| configuration | NFE | selection (6-ID mean) | worst ID |
|---|---|---|---|
| R90 {sigma_1, sigma_7} + R180 {sigma_1} | 10 | 85.75 | 54.4 |
| R90 {sigma_1, sigma_6} + R180 {sigma_1} | 10 | 85.70 | 53.6 |
| R90 {sigma_1} + R180 {sigma_1} + R270 {sigma_6} | 10 | 85.65 | 52.6 |
| R90 {sigma_6} + R180 {sigma_1} (best with 8 passes, reference) | 8 | 84.12 | 53.6 |
| grid candidate c3: R90, odd five scales (`PROTOCOL_20NFE.md`, Ablation) | 20 | 83.90 | 55.8 |

The first three are tied within 0.1 points at the same cost, so the worst-ID rule selects **R90 {sigma_1, sigma_7} + R180
{sigma_1}** (`e10`, `rule.json`).

The tie rule was stated when it turned out that the order of the first two depends on which development images are used:
the exploratory search (`history/explore_10nfe.py`) used only the images that the pair and the single-noise runs have in
common, and there the sigma_6 configuration was 0.01 points ahead; on all development images (`analysis/select_rule.py`, the
values above) the sigma_7 configuration is 0.05 points ahead. The rule was therefore not fixed before the search. The choice
does not change the conclusions of the main evaluation: e10b 87.5% / 36 pairs, e10c 87.5% / 37 pairs, e10 86.9% / 34 pairs;
all three are above the best grid candidate c3 (85.4% / 36 pairs) on the mean.

## Checks (`analysis/ablations/robustness.py`, on the images common to the pair and single runs)

- Split-half selection: choosing on the odd- (even-) numbered extra OOD sets and checking on the other half, the chosen
  configuration beats c3 on the held-out half in both splits (88.0 vs 87.7; 83.0 vs 80.6).
- Noise seeds, CIFAR-10, the 12 pairs on all final-split images, three seeds (mean +- sd): e10 (Euphonium) 92.2 +- 0.6,
  e10b 92.2 +- 0.7, c3 91.7 +- 0.5.

## Further checks (`analysis/ablations/frontier.py`, `results/ablations/frontier.json`)

- The log-ratio form is a design choice (it removes the weight e_k(x) that lets the finest scales dominate), not a choice made
  on the selection data. Searching the same 1285 candidates in the difference form e_k(Tx) - e_k(x) (the first approximation
  of the discretised integral), the best on the selection data is R90 {sigma_9} + R180 {sigma_1} + R270 {sigma_9} (86.90,
  worst ID 64.6), above the best log-ratio candidate (85.75). Its main evaluation (afterwards) is 85.4% / 36 pairs, below
  Euphonium's 86.9% / 34 pairs on the mean (SVHN 86.3). Euphonium's three atoms in the difference form give 84.3% / 36 pairs
  (best of the eight methods on 5 of the 6 IDs; SVHN 73.5).
- The scale window: with candidates over sigma_1..sigma_13 (five IDs; the FashionMNIST development run has ten scales), no atom
  at sigma_11..sigma_13 enters the top ten, in either form.
- Initially the log-ratio form and the window sigma <= sigma_10 had been fixed on development copies of the main OOD sets, which
  the rule of this repository (choose on data outside the main evaluation only) does not allow; the two checks above repeat
  them on the selection data: the window is confirmed, the log-ratio form is not preferred by the selection data.
- Budgets (pair, best on the selection data): 8 passes R90 {sigma_6} + R180 {sigma_1} (main 86.4%), 6 passes R90 {sigma_1} +
  R180 {sigma_1} (82.3%), 4 passes R90 {sigma_1} (81.7%). Single noise (+eps only; N_ runs, images common to both runs): with 5
  passes R90 {sigma_1, sigma_6} + R180 {sigma_1}, selection 84.7, main 86.4% / 30 pairs (Euphonium on the same images: 85.7,
  86.9%).

## Main evaluation of the selected rule (afterwards; not used for the selection)

e10: mean over the six IDs 86.9% (best grid candidate c3: 85.4%); ties or beats the best of the eight comparison methods on
34 of 63 pairs (c3: 36). The tied runner-up R90 {sigma_1, sigma_6} + R180 {sigma_1}: 87.5%, 36 of 63.
