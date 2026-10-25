# <img src="paper/figs/euphonium_icon.svg" alt="" height="40" align="middle"> Euphonium

**Out-of-distribution detection with the self-rotation likelihood ratio of a diffusion model, in 10 forward passes per image.**

Code, result files and LaTeX source of the paper *Euphonium：拡散モデルの自己回転尤度比による分布外検知* (Japanese;
[`paper/main.pdf`](paper/main.pdf), [`paper/`](paper/README.md)).

Likelihoods of diffusion models are higher for simple out-of-distribution (OOD) images than for in-distribution (ID) images
(the likelihood paradox). Euphonium compares an image with its own rotation instead. Rotating an image does not change its
complexity, so in the ratio

  Λ_T(x) = log p(x) − log p(Tx)

the complexity cancels and only the model's knowledge "is the upright image more plausible?" remains; Λ_T is the log-odds of
the Bayes-optimal classifier between upright and rotated images. The ratio is the integral over noise scales of the difference
of two denoising errors, which we estimate on **shared noise**: the noised upright image x + σε is itself rotated, so both errors
are measured against the same ε. Euphonium adds the log-ratios of these errors over three (rotation, scale) **atoms**:

  Eu(x) = Σ_{(T,k)∈A} log ê_k(Tx) / ê_k(x),  A = {(R90, σ1), (R90, σ7), (R180, σ1)}

with an antithetic pair (+ε, −ε) per scale: 2 × (2 scales + 3 atoms) = **10 forward passes, no backward pass, no training**
(higher = more in-distribution). The rule is fixed in [`rule.json`](rule.json); it was selected among 1285 candidates on data
outside the main evaluation ([`protocol/PROTOCOL_10NFE.md`](protocol/PROTOCOL_10NFE.md)). Euphonium is a weighted statistic
built from the terms of the ratio, not an estimate of Λ itself (its rank correlation with the exact ratio is 0.35–0.78).

## Results

Six ID datasets with their pretrained (or our) diffusion models, 63 ID/OOD pairs, the same images for every method. Mean AUROC
(%) over the natural OOD sets of every ID ([`results/euphonium_pairs.json`](results/euphonium_pairs.json),
[`results/baselines_pairs.json`](results/baselines_pairs.json)):

| ID (model) | Euphonium (10 passes) | best of the eight comparison methods |
|---|---|---|
| CIFAR-10 (EDM, DDPM++) | **91.2** | 75.0 (EigenScore) |
| SVHN (VP-SDE U-Net) | 98.0 | **99.5** (DiffPath) |
| CelebA (iDDPM) | **100.0** | 99.8 (DiffPath) |
| CIFAR-100 (EDM-type DDPM++) | **72.8** | 67.1 (IC, PNG) |
| ImageNet-64 (EDM2-XS) | 60.5 | **62.4** (MSMA-type) |
| FashionMNIST (our U-Net) | **98.9** | 95.8 (MSMA-type) |

Mean over the six IDs 86.9%; Euphonium ties or beats the best of the eight methods (NLL, IC-PNG, IC-J2K, typicality test,
MSMA-type, DiffPath, GEPC, EigenScore) on 34 of the 63 pairs. The ratio is one-sided: pairs whose OOD data are more strongly
oriented than the ID data reverse (CIFAR-100 vs CIFAR-10, ImageNet-64 vs CelebA), and they do so for the exact ratio as well.
For reference, a feature kNN of the same CIFAR-10 model reaches 92.5% (not a comparison method here;
[`results/external/`](results/external/knn_feature.json)).

## Euphonium and the Ablations

Only the first row is the method. Everything else is an Ablation (or a reference): it lives under `analysis/ablations/`, in
the lower block of Table A3 or in the supplementary tables, and none of it is used by `euphonium/`.

| | configuration | passes / image | where |
|---|---|---|---|
| **Euphonium (Ours)** | R90 at σ1, σ7 and R180 at σ1, log-ratio form, antithetic pair | **10** | [`rule.json`](rule.json), [`euphonium/score.py`](euphonium/score.py), [`experiments/score_euphonium.py`](experiments/score_euphonium.py), [`analysis/select_rule.py`](analysis/select_rule.py), [`analysis/evaluate.py`](analysis/evaluate.py) |
| Ablation | the tied runners-up of the selection: e10b (R90 σ1, σ6 + R180 σ1), e10c (R90 σ1 + R180 σ1 + R270 σ6); the best with 8, 6, 4 passes | 4–10 | `analysis/select_rule.py`, `analysis/ablations/frontier.py` |
| Ablation | grid candidates c1–c12: one scale set for every rotation (c3 = R90 on the odd scales σ1, σ3, …, σ9 was the earlier 20-pass version) | 15–20 | [`protocol/PROTOCOL_20NFE.md`](protocol/PROTOCOL_20NFE.md), `analysis/ablations/grid_candidates.py` |
| Ablation | r60 / r80 (all ten scales, two / three rotations), Eu10 (R90, ten scales: the shared-noise estimate of Λ90) | 40–80 | Table A3, Figure 3, Tables A10–A11 |
| Ablation | difference form e_k(Tx) − e_k(x), scale window σ1–σ13, single noise (+ε only) | 2–10 | `analysis/ablations/frontier.py` |
| Ablation | noise controls: independent noise, no antithetic pair, per-image noise, three seeds | 10–20 | `paper/gen/make_paper.py` (Tables A4, A5), `analysis/ablations/robustness.py` |
| Ablation | the exact ratio Λ90 by the PF-ODE | 256 (+ 256 backward) | [`analysis/exact_ratio.py`](analysis/exact_ratio.py) |
| reference only | dev: chosen on development copies of the main OOD sets (not eligible under the protocol) | 20 | Table A3 |

## Quick start: score your images

```bash
pip install -r environment/requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
bash third_party/fetch.sh                                  # NVlabs edm / edm2, improved-diffusion into ext/ (loaders of the checkpoints)
mkdir -p ckpt && wget -P ckpt https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
python experiments/prep_data.py cifar10                    # data/npy/cifar10_test.npy (data/README.md)
python experiments/score_euphonium.py --model edm:edm-cifar10-32x32-uncond-vp.pkl --data cifar10_test \
       --n 3000 --complement --cap 5000 --out runs/eu/edmc10/cifar10_test.npz
```

From Python, with any denoiser `den(y, sigma) -> E[x0 | y]` (EDM convention `y = x0 + sigma * eps`) and images in [-1, 1]:

```python
import sys; sys.path.insert(0, 'experiments')
from core import load_model, to_t                           # loaders of the six models of the paper
from euphonium import euphonium
den, net = load_model('edm:edm-cifar10-32x32-uncond-vp.pkl', 'cuda')
out = euphonium(den, to_t(images_uint8_nhwc, 'cuda'))       # out['eu']: score per image; out['nfe'] == 10
```

`python tests/test_euphonium.py` checks the implementation on toy denoisers (a rotation-equivariant model gives Eu = 0 exactly;
the score equals a direct evaluation of the definition; 10 passes per image). `python experiments/check_euphonium.py --model ...
--data ... --export inputs/feat/<run>/<set>.npz` checks it against the code that produced the paper's statistics: on CPU with
the CIFAR-10 model the per-atom errors agree with the full-grid code to 2.4e-6 (relative), and Eu agrees with the paper's GPU
run (fp16) to 6e-4.

## Reproducing the paper

| level | command | needs | time |
|---|---|---|---|
| 0 | `bash paper/build.sh` | pLaTeX (TeX Live 2023) and the MIRU 2025 class files ([`paper/README.md`](paper/README.md)) | 1 min |
| 1 | `bash reproduce.sh` | Python 3.13 with [`environment/requirements-analysis.txt`](environment/requirements-analysis.txt) and the per-image statistics (release asset) | 19 min on 2 CPU cores |
| 2 | the GPU runs: [`experiments/README.md`](experiments/README.md) | GPUs, the datasets, the checkpoints | GPU-days |

The per-image statistics of the GPU runs are the release asset `inputs-v1` (`euphonium_inputs_v1.tar`, 447 MB): every file
the analysis reads, with the arrays it reads and the arguments of the run that wrote it. Extract it in the repository root and
check it:

```bash
gh release download inputs-v1 -R omotes24/Euphonium        # or download it from the Releases page
tar -xf euphonium_inputs_v1.tar && md5sum -c --quiet analysis/MD5_inputs.txt
bash reproduce.sh
```

`reproduce.sh` runs `analysis/baselines_eval.py` (the eight comparison methods), `analysis/exact_ratio.py`,
`analysis/select_rule.py` (the selection of the rule; it checks that the winner is `rule.json`), `analysis/evaluate.py`
(Euphonium on the 63 pairs), `analysis/ablations/{grid_candidates,frontier,robustness}.py`,
`protocol/history/explore_10nfe.py` and `paper/gen/make_paper.py`, and compares every recomputed file with the committed copy
(JSON by value, TeX line by line, figures by their rendered pixels).

## Layout

| folder | contents |
|---|---|
| [`euphonium/`](euphonium/) | the method: `score.py` (10 forward passes), `rule.py` (the rule and the scale grid), `noise.py` (shared noise), `transforms.py` |
| [`experiments/`](experiments/README.md) | GPU side: data preparation, training of our models, per-image statistics (`extract2.py`: full grid; `score_euphonium.py`: the 10 passes), the comparison methods, PF-ODE likelihoods, timing, the job lists we ran |
| [`analysis/`](analysis/README.md) | CPU side: per-image statistics → result files: rule selection, main evaluation, comparison methods, exact ratio, Ablations |
| [`results/`](results/README.md) | result files (JSON) written by `analysis/` and read by `paper/gen/`; CPU timing of Table A2 |
| [`paper/`](paper/README.md) | LaTeX source of the paper (Japanese), its generated tables / figures / macros, `gen/make_paper.py`, `main.pdf` |
| [`protocol/`](protocol/) | how the rule was selected (`PROTOCOL_10NFE.md`), the grid candidates (`PROTOCOL_20NFE.md`, Ablation), the exploratory search (`history/`) |
| [`benchmark/`](benchmark/) | `common_idx.json`: the images every method is evaluated on (per ID and OOD set) |
| [`checkpoints/`](checkpoints/README.md), [`data/`](data/README.md) | sources and checksums of the models and of the image arrays (not redistributed) |
| `tests/`, `environment/`, [`third_party/`](third_party/README.md) | checks with toy denoisers; package versions; external repositories (fetched by a script, not copied) |

## Licence

Our code and result files are released under the [MIT licence](LICENSE). The pretrained models, the datasets, the third-party
code that the experiments use and the MIRU 2025 LaTeX class files keep their own licences and are not included.
