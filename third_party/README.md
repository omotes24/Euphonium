# Third-party code

`experiments/` imports code from four public repositories, which are **not** copied into this repository (their licences
differ from ours). `bash third_party/fetch.sh` clones them into `ext/` at the commits we used; none of them is modified.

| folder (`ext/`) | repository | commit | used for | licence |
|---|---|---|---|---|
| `edm` | [NVlabs/edm](https://github.com/NVlabs/edm) | `008a4e5` | loading the EDM CIFAR-10 pickle, and the DDPM++ network class of the CIFAR-100 checkpoint | CC BY-NC-SA 4.0 |
| `edm2` | [NVlabs/edm2](https://github.com/NVlabs/edm2) | `4bf8162` | loading the EDM2 ImageNet-64 pickle | CC BY-NC-SA 4.0 |
| `improved-diffusion` | [openai/improved-diffusion](https://github.com/openai/improved-diffusion) | `1bc7bbb` | the iDDPM network of DiffPath's CelebA model | MIT |
| `gepc_repo` | [RouzAY/gepc-diffusion](https://github.com/RouzAY/gepc-diffusion) | `1b01cf2` | numerical check of our GEPC port against the official code (`experiments/bl_check*.py`) | see the repository |

The SVHN model is a `diffusers` `UNet2DModel` (installed with `environment/requirements-gpu.txt`). The 10-pass Euphonium
itself (`euphonium/`) needs only PyTorch and a denoiser.
