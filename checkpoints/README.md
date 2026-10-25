# checkpoints

## Pretrained models (download into `ckpt/`)

The five released models of the main evaluation, used without modification. `experiments/core.py` (`load_model`) reads them
from `ckpt/<file>`; the model spec of the job lists is `<kind>:<file>`. Checksums of the files we used:
[`MD5_pretrained.txt`](MD5_pretrained.txt) (`cd ckpt && md5sum -c <(awk '{print $1"  "$2}' ../checkpoints/MD5_pretrained.txt)`).

| ID | `ckpt/` file | spec | model | source |
|---|---|---|---|---|
| CIFAR-10 | `edm-cifar10-32x32-uncond-vp.pkl` | `edm:` | EDM, VP, DDPM++ (Karras et al., 2022) | https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl |
| SVHN | `layer6_svhn_vpsde.pt` | `l6:` | VP-SDE U-Net (diffusers `UNet2DModel`) of Kamkari et al. (2024) | `layer6ai-labs/dgm_ood_detection`, checkpoint folder `diffusion_svhn_regretless_ixiwzx2l_final` (score-based VP diffusion), renamed |
| CelebA | `diffpath_celeba32.pt` | `iddpm:` | iDDPM of DiffPath (Heng et al., 2024), 4000-step cosine schedule | https://huggingface.co/ajrheng/diffpath/resolve/main/celeba_ema_0.9999_499999.pt, renamed |
| CIFAR-100 | `edm_cifar100_uncond_vp.pt` | `edmsd:` | unconditional DDPM++ with the EDM VP preconditioning (state dict of `EDMPrecond`) of Chen et al. (2024), *Robust Classification via a Single Diffusion Model* (code: `huanranchen/DiffusionClassifier`) | the authors' Google Drive folder of EDM checkpoints https://drive.google.com/drive/folders/1mQoH6WbnfItphYKehWVniZmq9iixRn8L, file `edm_cifar100_uncond_vp.pt` |
| ImageNet-64 | `edm2-img64-xs-uncond-2147483-0.100.pkl` | `edm2:` | EDM2-XS, unconditional (Karras et al., 2024): raw snapshot after 2147483 kimg, power-function EMA with σ_rel = 0.100 | https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img64-xs-uncond/edm2-img64-xs-uncond-2147483-0.100.pkl |

These files are under the licences of their authors and are not redistributed here.

## Models we trained (`runs/models/`, spec `own:models/...`)

| `runs/models/` | trained on | images seen | used for | command |
|---|---|---|---|---|
| `fmnist_s/final.pt` | FashionMNIST train (32×32, zero padded), 22M parameters | 10M | the FashionMNIST ID (all tables) | `experiments/jobs/TRAIN.txt` |
| `c10_m/snap_{0.125M … 32M}.pt` | CIFAR-10 train | 0.125M–32M | training length (A.7) | `experiments/jobs/TRAIN.txt` |
| `tin_m/snap_{0.125M … 8M}.pt` | Tiny-ImageNet train (100k images at 32×32) | 0.125M–8M | training length (A.7) | `experiments/jobs/TRAIN.txt` |
| `tin_m32/snap_16M.pt`, `snap_32M.pt` | continued training of `tin_m` | 16M, 32M | training length (A.7) | `experiments/jobs/TRAIN.txt` |

They are the EMA weights written by `experiments/train.py` (EDM preconditioning and loss; snapshots at the image counts of
`--snap`). `train.py` holds out 5000 training images (`RandomState(1234)` permutation, stored as `calib_idx` in every
checkpoint). The weights are not part of this repository; [`MD5_own_models.txt`](MD5_own_models.txt) lists the checksums of
the files we keep (named `<name>-<file>`; the training-time `net` weights removed, `load_model` reads only `ema`).
