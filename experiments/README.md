# experiments (GPU)

Everything that evaluates a network: data preparation, training of our models, the per-image statistics, the comparison
methods, PF-ODE likelihoods and timing. We ran the jobs on one server with four GeForce RTX 2080 Ti (11 GB); the per-image
statistics were exported (`export.py`) and analysed on a CPU (`analysis/`). Commands run from the repository root.

## Setup

```bash
conda create -n euphonium python=3.11 && conda activate euphonium
pip install -r environment/requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
bash third_party/fetch.sh            # NVlabs edm / edm2 and improved-diffusion into ext/ (loaders of the checkpoints)
# pretrained weights into ckpt/ (checkpoints/README.md); image arrays into data/npy/ (data/README.md)
```

Environment variables: `SR_ROOT` (repository root; default: the parent of `experiments/`), `SR_DATASETS` (datasets the
scripts do not download; default `data/external/`), `SR_FP16=0` (fp32 instead of fp16 autocast; used for the CIFAR-100 model
and required by `bl_check*.py`).

## Euphonium

| file | role |
|---|---|
| `score_euphonium.py` | **the method**: Euphonium of one image set under one model, 10 forward passes per image (`euphonium/score.py`, rule `rule.json`) → `runs/eu/<tag>/<set>.npz` |
| `check_euphonium.py` | checks the 10-pass implementation against `core.rails` (the code of the paper's runs) and, optionally, against an exported run |
| `bench_cpu.py` | Table A2: CPU seconds and network evaluations per image of Euphonium, the Ablations c3 / r60 and the comparison methods → `results/bench_cpu.jsonl` |
| `mk_gp_jobs.py` | jobs of the pending control of Section 4 (prediction difference from Euphonium's own 10 passes, `--pred_diff`) → `jobs/GP.txt` |

The paper's numbers come from full-grid runs (`extract2.py`: squared noise-prediction errors of every image and its
transforms at 13 scales on shared noise, `F_<tag>`); the analysis takes Euphonium's three atoms from them, and every Ablation
(other atom sets, grid candidates, ten-scale estimate) from the same runs. `score_euphonium.py` evaluates only the 10 passes
of the rule, with the same noise images and the same scale values, so it gives the same scores: on the CIFAR-10 model (CPU)
the per-atom errors agree with `core.rails` to 2.4e-6 (relative) and Eu agrees with the paper's GPU run (fp16) to 6e-4.

## Code of the runs

| file | role |
|---|---|
| `core.py` | data (`load_npy`, the fixed development / final selection `select`), model loading (`load_model('<kind>:<file>')`: `edm`, `edm2`, `edmsd`, `iddpm`, `l6`, `own`, …), the denoiser interface `den(y, sigma) = E[x0 | y]`, shared antithetic noise (`shared_noise`), transforms, per-scale statistics (`rails`) |
| `extract2.py` | per-scale statistics of each image and its transforms on shared noise (scales, antithetic pairs, DCT bands, transforms) → `runs/feat/<run>/<set>.npz` |
| `extract_bu.py` | `extract2.py` on a subset of the scale grid with the noise images of the full grid (the training snapshots of A.7: identity and R180 at σ1, σ3, σ4, σ6, σ8) |
| `extract.py` | clean-input score norms on a 20-scale grid (MSMA-type comparison method) → `runs/base/` |
| `nll.py` | PF-ODE log-likelihood (Heun, 64 steps, one Rademacher probe of the divergence), also of the R90 copies with the rotated probe (the exact ratio) → `runs/nll/` |
| `baselines.py` | DiffPath, GEPC, EigenScore re-implemented on the denoiser interface → `runs/base2/`; checks against the official code: `bl_check.py`, `bl_check_q.py`, `eig_conv.py` |
| `complexity.py` | PNG / JPEG 2000 code lengths of every image (IC comparison methods) → `data/complexity/` |
| `unet.py`, `train.py` | EDM-preconditioned U-Net and EDM training of our models (the FashionMNIST model of the main evaluation; the CIFAR-10 and Tiny-ImageNet models of the training-length experiment, A.7) |
| `prep_*.py` | image arrays (`data/README.md`) |
| `runq2.py` | job queue: `python experiments/runq2.py experiments/jobs/<list>.txt <gpu>`, one worker per GPU |
| `export.py` | the compact per-image statistics that the analysis reads: `runs/` → `inputs/` (the release asset `inputs-v1` holds the files the analysis reads) |

## Job lists (`jobs/`)

| list | jobs | contents | used for |
|---|---|---|---|
| `TRAIN.txt` | 4 | training of our models: FashionMNIST, the CIFAR-10 and Tiny-ImageNet snapshot models | FashionMNIST ID; A.7 |
| `DEV.txt` | 6 | full-grid statistics of the ID development images of the six IDs | selection data (ID side): the rule, the grid candidates |
| `POOL.txt` | 60 | full-grid statistics of the extra OOD sets (final-split images) | selection data (OOD side) |
| `FINAL.txt` | 299 | full-grid statistics of the 63 pairs, the CIFAR-10 noise controls, score norms (MSMA-type), PF-ODE likelihoods of the images and of their R90 copies | the main evaluation: Euphonium, the comparison methods, the exact ratio, the Ablations |
| `BASELINES.txt` | 225 | DiffPath, GEPC, EigenScore | the comparison methods |
| `N20.txt` | 135 | single-noise runs (+ε only), ten scales, four orientations (written by `mk_n20_jobs.py`) | Ablations with single noise |
| `D3.txt` | 18 | identity + R180 at five scales on the snapshots of our models | training length (A.7) |
| `COMPLEXITY.txt` | 23 | PNG / JPEG 2000 code lengths (CPU) | IC comparison methods |
| `GP.txt` | 69 | Euphonium's 10 passes with `--pred_diff` on the 63 pairs (written by `mk_gp_jobs.py`; **not run yet**) | the pending control of Section 4 |

Order: `TRAIN` and the data first; `DEV`, `POOL`, `FINAL`, `BASELINES`, `N20`, `D3` in any order (the `F_fmnist` jobs need
the FashionMNIST model, the `D3` jobs the snapshots); then `python experiments/export.py` and the analysis. `mkdir -p logs`
before the training jobs (they append to `logs/train_<name>.log`).

Each line is one shell command (`{gpu}` is replaced by the GPU index). A worker claims a line by creating a marker file in
`jobs/<list>_claims/`, so several workers can share a list; the extraction jobs skip an output that exists (`train.py` resumes
from `latest.pt`). The lists hold the jobs
whose outputs this paper uses, with the arguments we ran (checked against the arguments stored in every output file); the
scheduling prefixes of the original lists (waiting for a model file) are removed.

The scripts are those of our runs, with the paths of this repository and revised comments (`core.py` and `unet.py` had other
file names on the server); `prep_newid.py` and `export.py` are reduced to the parts this paper uses.
