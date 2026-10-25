# score.py -- the Euphonium score of a diffusion model: 10 forward passes per image, no backward pass
#
#   Eu(x) = sum over the atoms (T, k) of  log e_k(T x) - log e_k(x)                     (higher = more in-distribution)
#
#   with the shared-noise estimates of the denoising errors at noise scale sigma_k
#       e_k(x)   = 1/2 sum_{s = +1, -1} || s eps_k - epshat(x + s sigma_k eps_k) ||^2
#       e_k(T x) = 1/2 sum_{s = +1, -1} || s eps_k - T^-1 epshat(T (x + s sigma_k eps_k)) ||^2
#   where eps_k is ONE noise image shared by every test image (noise.shared_noise), epshat(y) = (y - D(y, sigma)) / sigma and
#   D(y, sigma) = E[x0 | y] is the denoiser (EDM convention y = x0 + sigma eps).  Rotating the noised input is the same as
#   adding the rotated noise T eps_k to the rotated image, and T eps_k has the law of eps_k, so e_k(T x) is a valid estimate of
#   the error of the rotated image while sharing its noise with e_k(x) (this removes the ||eps||^2 fluctuation from the ratio).
#
#   Rule (rule.json): atoms A = {(R90, sigma_1), (R90, sigma_7), (R180, sigma_1)}, antithetic pair, seed 0.
#   Forward passes per image: 2 x (2 scales for the upright image + 3 atoms) = 10.
#
#   The computation is that of experiments/core.py: rails() restricted to the atoms (same noise images, same scale values,
#   same order of operations); experiments/check_euphonium.py compares the two on real images.
import torch
from .rule import SIGMA_GRID, load_rule, scales, nfe
from .noise import shared_noise
from .transforms import TF


def sigma_value(k):
    """the scale value the experiments pass to the denoiser for grid index k (job lists write sigma with 6 significant digits)"""
    return float('%.6g' % SIGMA_GRID[k])


@torch.no_grad()
def euphonium(den, x, rule=None, seed=None, fuse=False, pred_diff=False):
    """Euphonium of a batch.
    den  : callable den(y, sigma) -> E[x0 | y] for a batch y [B, C, H, W] and a python float sigma (experiments/core.py: Denoiser)
    x    : float tensor [B, C, H, W] with pixel values in [-1, 1]
    rule : dict from euphonium.rule.load_rule() (default: rule.json); seed overrides rule['seed']
    fuse : evaluate all inputs of one scale in a single batch (faster on a GPU; the default calls the denoiser once per
           sign and rotation with batch B, exactly as the experiments did)
    pred_diff : also return, from the same forward passes, the prediction difference of every atom
           pd[(T, k)] = 1/2 sum_s ||T^-1 epshat(T x_s) - epshat(x_s)||^2 and a_up[k] = 1/2 sum_s ||epshat(x_s)||^2 (the magnitude that
           GEPC-type detectors read; used for the sign-or-magnitude control of the paper)
    returns dict(eu [B], log_ratios [B, n_atoms] in the order of rule['atoms'], e_up {k: [B]}, e_rot {(T, k): [B]},
                 nfe = forward passes per image[, pd {(T, k): [B]}, a_up {k: [B]}])"""
    rule = rule or load_rule()
    atoms = list(rule['atoms']); seed = rule.get('seed', 0) if seed is None else seed
    signs = (1.0, -1.0) if rule['noise'] == 'pair' else (1.0,)
    B, shape, dev = x.shape[0], x.shape[1:], x.device
    e_up, e_rot, pd, a_up, images = {}, {}, {}, {}, 0
    for k in scales(atoms):
        sig = sigma_value(k)
        eps = shared_noise(shape, seed, k, 0, dev)
        tfs = ['id'] + [T for T, kk in atoms if kk == k]
        acc = {t: torch.zeros(B, dtype=torch.float64, device=dev) for t in tfs}
        dif = {t: torch.zeros(B, dtype=torch.float64, device=dev) for t in tfs}           # dif['id'] holds ||epshat(x_s)||^2
        noised = [(s * eps, x + sig * (s * eps)) for s in signs]
        if fuse:
            inp = torch.cat([TF[t][0](xn) for _, xn in noised for t in tfs])
            out = den(inp, sig).split(B); images += inp.shape[0]
        for i, (e, xn) in enumerate(noised):
            for j, t in enumerate(tfs):
                f, finv = TF[t]
                if fuse: d = finv(out[i * len(tfs) + j])
                else: d = finv(den(f(xn), sig)); images += B
                eh = (xn - d) / sig                                              # T^-1 epshat(T xn): predicted noise, rotated back
                acc[t] += (e - eh).double().square().flatten(1).sum(1) / len(signs)
                if pred_diff:
                    if t == 'id': eh_id = eh; dif[t] += eh.double().square().flatten(1).sum(1) / len(signs)
                    else: dif[t] += (eh - eh_id).double().square().flatten(1).sum(1) / len(signs)
        e_up[k] = acc['id']; a_up[k] = dif['id']
        for t in tfs[1:]: e_rot[(t, k)] = acc[t]; pd[(t, k)] = dif[t]
    lr = torch.stack([torch.log(e_rot[(T, k)]) - torch.log(e_up[k]) for T, k in atoms], 1)
    used = images / B
    assert used == nfe(atoms, rule['noise']), (used, nfe(atoms, rule['noise']))
    out = dict(eu=lr.sum(1), log_ratios=lr, e_up=e_up, e_rot=e_rot, nfe=used)
    if pred_diff: out.update(pd=pd, a_up=a_up)
    return out


class Euphonium:
    """score = Euphonium(den)(x): the Euphonium score [B] of a batch (higher = more in-distribution)"""
    def __init__(self, den, rule=None, seed=None, fuse=False):
        self.den, self.rule, self.seed, self.fuse = den, rule or load_rule(), seed, fuse

    def __call__(self, x):
        return euphonium(self.den, x, self.rule, self.seed, self.fuse)['eu']
