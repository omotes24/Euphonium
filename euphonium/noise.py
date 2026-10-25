# noise.py -- the shared noise images of the experiments (identical to experiments/core.py: shared_noise)
import torch


def shared_noise(shape, seed, k, j, device):
    """One Gaussian noise image [1, *shape] shared by every test image, for scale index k of the grid and draw j.
    Draws come in antithetic pairs: j = 2m and j = 2m + 1 use the same image with opposite signs.
    The generator runs on the CPU, so the noise does not depend on the device."""
    g = torch.Generator(device='cpu').manual_seed(int(seed) * 1000003 + int(k) * 1009 + (int(j) // 2))
    eps = torch.randn([1] + list(shape), generator=g).to(device)
    return -eps if (j % 2 == 1) else eps
