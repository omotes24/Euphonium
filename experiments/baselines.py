# baselines.py -- re-implementations of three diffusion-model OOD detectors on the shared denoiser API of core
#   den(y, sigma) = E[x0 | y],  y = x0 + sigma * eps   (EDM convention; VP models: y = x_t / sqrt(abar_t), sigma^2 = (1 - abar_t) / abar_t)
# Writes raw per-image statistics; calibration and scoring use ID images only and are done offline (analysis/baselines_eval.py).
#
# diffpath  DiffPath (Heng et al., NeurIPS 2024; github.com/clear-nus/diffpath, save_*_statistics.py, eval_1d.py, eval_6d.py).
#           Deterministic DDIM encoding x0 -> x_T with the 'ddim10' striding of the 4000-step cosine schedule of the official
#           models (timesteps 0, 400, ..., 3600) and clip_denoised=True (x0-prediction clipped to [-1, 1], eps re-derived).
#           In sigma coordinates one DDIM step is  eps = (y - clip(D(y))) / sigma,  y' = clip(D(y)) + sigma' * eps.
#           Saves per-step sums over pixels of eps^q and of (d eps / dt)^q = (diff(eps) * 10)^q, q = 1, 2, 3.        10 NFE
# gepc      GEPC (Alexis et al., 2026; github.com/RouzAY/gepc-diffusion, gepc/methods/gepc.py, _gepc_at_t).
#           Transported residual g^-1 s(g y) - s(y) for g in {hflip, vflip, rot90, rot180, 1-px circular x/y shifts} (the identity
#           element of the official group contributes zero), channel-averaged squared maps pooled by the mean or by the mean of the
#           top rho fraction (official topk_rho 0.3 / 0.37) and divided by the pooled score energy; also the cosine and pairwise
#           variants.  One fresh noise draw per image and level (mc_samples = 1).  Levels: union of the official configs'
#           snr_levels sqrt(abar) in {0.99997, 0.9999, 0.999, 0.9979, 0.9969, 0.99}, snapped to the cosine-4000 grid as in
#           _build_t_list.  The score s = -eps_hat / sigma_t enters only through ratios, so eps_hat is used.        7 NFE per level
# eigen     EigenScore (Shoushtari et al., ICLR 2026, Algorithms 1-2).  Top-K eigenvalues of the posterior covariance
#           sigma^2 J_D(y), y = x + sigma z, by Jacobian-free subspace iteration (central differences of D along K orthonormal
#           directions, QR); lambda_i = sigma^2 ||J v_i|| with v_i the directions entering the last iteration.  VP timesteps
#           t in {100, 150, 200, 250, 300} of 1000 (beta 0.1 -> 20), I repetitions.                     2 K iters I NFE per timestep
import os, sys, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *


def cosine_schedule(steps=4000):
    """abar and sigma of every timestep index of the improved-diffusion cosine schedule (betas clipped at 0.999)"""
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = np.array([min(1 - f((i + 1) / steps) / f(i / steps), 0.999) for i in range(steps)], dtype=np.float64)
    ac = np.cumprod(1.0 - betas)
    return ac, np.sqrt(1.0 / ac - 1.0)


AC_COS, SIG_COS = cosine_schedule()
DP_T = list(range(0, 4000, 400))                                  # space_timesteps(4000, 'ddim10')
DP_SIG = SIG_COS[DP_T]
GEPC_SNR = (0.99997, 0.9999, 0.999, 0.9979, 0.9969, 0.99)
GEPC_T = [int(np.argmin(np.abs(np.sqrt(AC_COS.astype(np.float32)) - s))) for s in GEPC_SNR]
GEPC_SIG = SIG_COS[GEPC_T]
GEPC_G = ('hflip', 'vflip', 'rot90', 'rot180', 'rollx1', 'rolly1')
POOLS = (0.0, 0.3, 0.37)                                          # 0 = mean pooling, else mean of the top rho fraction of |map|
EIG_T = (0.1, 0.15, 0.2, 0.25, 0.3)
EIG_SIG = np.array([math.sqrt(math.expm1(0.5 * 19.9 * t * t + 0.1 * t)) for t in EIG_T])


def load_iddpm_rawt(path, device, steps=4000):
    """core.load_iddpm with one change: the UNet gets the raw timestep index t instead of t * 1000 / steps.  This is how the
    official GEPC adapter calls the 4000-step CelebA model (gepc/adapters/improved.py: self.model(x_t, t), while the model was
    trained with rescale_timesteps=True); used only to reproduce the published GEPC numbers (runs gepc_celq_*)."""
    p = f'{ROOT}/ext/improved-diffusion'
    if p not in sys.path: sys.path.insert(0, p)
    from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
    a = model_and_diffusion_defaults()
    a.update(image_size=32, num_channels=128, num_res_blocks=3, learn_sigma=True, diffusion_steps=steps, noise_schedule='cosine', dropout=0.3)
    model, diffusion = create_model_and_diffusion(**a)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.to(device).eval().requires_grad_(False)
    ac = torch.tensor(np.asarray(diffusion.alphas_cumprod), dtype=torch.float64, device=device)
    logsig = 0.5 * (torch.log1p(-ac) - torch.log(ac))

    def fn(x, sigma):
        t = (logsig[None, :] - sigma.double().log()[:, None]).abs().argmin(1)
        a_t = ac[t].sqrt().float().view(-1, 1, 1, 1)
        s_t = logsig[t].exp().float().view(-1, 1, 1, 1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=FP16):
            out = model(a_t * x, t.float())                                # raw index (official GEPC adapter)
        return x - s_t * out[:, :x.shape[1]].float()
    return Denoiser(fn, 'rawt_' + os.path.basename(path)), model


def pool(v, rho):
    """official _pool_spatial: [B,C,H,W] -> channel average -> mean, or mean of the top-k |values| with k = int(rho * H * W)"""
    v = v.mean(1).flatten(1)
    if rho == 0: return v.mean(1)
    k = max(1, int(rho * v.shape[1]))
    return torch.topk(v.abs(), k, dim=1).values.mean(1)


@torch.no_grad()
def diffpath_eps(den, x):
    """eps of the 10 DDIM encoding steps, [B, 10, C, H, W]"""
    y, E = x, []
    for i, s in enumerate(DP_SIG):
        s = float(s)
        d = den(y, s).clamp(-1, 1)
        e = (y - d) / s
        E.append(e)
        if i + 1 < len(DP_SIG): y = d + float(DP_SIG[i + 1]) * e
    return torch.stack(E, 1)


@torch.no_grad()
def diffpath(den, x):
    E = diffpath_eps(den, x).double()                             # [B, 10, C, H, W]
    dE = (E[:, 1:] - E[:, :-1]) * len(DP_SIG)                     # np.diff(eps, axis=1) * n_ddim_steps
    out = {}
    for q in (1, 2, 3):
        out[f'e{q}'] = (E ** q).flatten(2).sum(2).cpu()           # [B, 10]
        out[f'd{q}'] = (dE ** q).flatten(2).sum(2).cpu()          # [B, 9]
    return out


@torch.no_grad()
def gepc(den, x, gen):
    B, dev = x.shape[0], x.device
    L, nG, nP = len(GEPC_SIG), len(GEPC_G), len(POOLS)
    o = dict(num=torch.zeros(B, L, nG, nP, dtype=torch.float64), den=torch.zeros(B, L, nP, dtype=torch.float64),
             cos=torch.zeros(B, L, nG, dtype=torch.float64), pair=torch.zeros(B, L, nP, dtype=torch.float64))
    for l, s in enumerate(GEPC_SIG):
        s = float(s)
        y = x + s * torch.randn(x.shape, generator=gen).to(dev)
        ys = torch.cat([y] + [TF[g][0](y) for g in GEPC_G])      # one batched pass over the base copy and the group copies
        e = (ys - den(ys, s)) / s
        S = [e[:B]] + [TF[g][1](e[(i + 1) * B:(i + 2) * B]) for i, g in enumerate(GEPC_G)]   # back to the canonical frame
        for j, rho in enumerate(POOLS): o['den'][:, l, j] = pool(S[0].square(), rho).double().cpu()
        b0 = S[0].flatten(1)
        for i in range(nG):
            d = (S[i + 1] - S[0]).square()
            for j, rho in enumerate(POOLS): o['num'][:, l, i, j] = pool(d, rho).double().cpu()
            a_ = S[i + 1].flatten(1)
            o['cos'][:, l, i] = ((a_ * b0).sum(1) / (a_.norm(dim=1).clamp_min(1e-6) * b0.norm(dim=1).clamp_min(1e-6))).double().cpu()
        acc, n = torch.zeros(B, nP, device=dev, dtype=torch.float64), 0
        for i in range(nG + 1):                                   # pairs of the 7 group elements (identity included)
            for k in range(i + 1, nG + 1):
                d = (S[i] - S[k]).square()
                for j, rho in enumerate(POOLS): acc[:, j] += pool(d, rho).double()
                n += 1
        o['pair'][:, l] = (acc / n).cpu()
    return o


@torch.no_grad()
def eigen(den, x, gen, reps, iters, K):
    B, dev = x.shape[0], x.device
    D = x[0].numel()
    lam = torch.zeros(B, len(EIG_SIG), reps, iters, K, dtype=torch.float64)     # estimate after every iteration
    for l, s in enumerate(EIG_SIG):
        s = float(s)
        c = 0.01 * s * math.sqrt(D)                               # finite-difference step: 1% of the noise norm
        for r in range(reps):
            y = x + s * torch.randn(x.shape, generator=gen).to(dev)
            Y = y.repeat_interleave(K, 0)                         # rows (b, k)
            V = torch.randn((B, K, D), generator=gen).to(dev)     # v_0 ~ N(0, s^2 I): only the direction matters
            V = V / V.norm(dim=2, keepdim=True)
            for it in range(iters):
                P = (c * V).reshape(B * K, *x.shape[1:])
                out = den(torch.cat([Y + P, Y - P]), s)
                JV = ((out[:B * K] - out[B * K:]) / (2 * c)).reshape(B, K, D)
                lam[:, l, r, it] = (s * s * JV.norm(dim=2)).double().sort(dim=1, descending=True).values.cpu()
                if it + 1 < iters:
                    Q, _ = torch.linalg.qr(JV.transpose(1, 2))   # [B, D, K]
                    V = Q.transpose(1, 2).contiguous()
    return dict(lam=lam[:, :, :, -1], lam_it=lam)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['diffpath', 'gepc', 'eigen'])
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--subseed', type=int, default=0)
    p.add_argument('--complement', action='store_true')
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--pre', default='none')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reps', type=int, default=2)                 # EigenScore repetitions I
    p.add_argument('--iters', type=int, default=3)                # EigenScore subspace iterations
    p.add_argument('--neig', type=int, default=3)                 # EigenScore number of eigenvalues K
    p.add_argument('--limit', type=int, default=0)                # smoke tests: first images only
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    if os.path.exists(a.out):
        print('exists', a.out); sys.exit(0)
    dev = torch.device(f'cuda:{a.gpu}')
    torch.backends.cudnn.benchmark = True
    if a.model.startswith('iddpmraw:'): den, net = load_iddpm_rawt(f"{ROOT}/ckpt/{a.model.split(':', 1)[1]}", dev)
    else: den, net = load_model(a.model, dev)
    xu, idx = select(a.data, a.n, a.subseed, a.complement, a.cap)
    if a.limit: xu, idx = xu[:a.limit], idx[:a.limit]
    res, t0 = {}, time.time()
    for i in range(0, len(xu), a.bs):
        x = to_t(xu[i:i + a.bs], dev)
        if a.pre != 'none': x = TF[a.pre][0](x)
        gen = torch.Generator(device='cpu').manual_seed(int(a.seed) * 1000003 + 15485863 + i)
        if a.method == 'diffpath': st = diffpath(den, x)
        elif a.method == 'gepc': st = gepc(den, x, gen)
        else: st = eigen(den, x, gen, a.reps, a.iters, a.neig)
        for k, v in st.items(): res.setdefault(k, []).append(v.numpy())
        if (i // a.bs) % 5 == 0 or i + a.bs >= len(xu):
            el = time.time() - t0
            print(f'{a.method} {a.data} {i + len(x)}/{len(xu)} {el:.0f}s eta {el / (i + len(x)) * (len(xu) - i - len(x)):.0f}s '
                  f'mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}G', flush=True)
    res = {k: np.concatenate(v) for k, v in res.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, idx=idx, method=np.array(a.method), meta=np.array(str(vars(a))), dp_sig=DP_SIG, gepc_sig=GEPC_SIG,
             gepc_t=np.array(GEPC_T), gepc_snr=np.array(GEPC_SNR), gepc_g=np.array(GEPC_G), pools=np.array(POOLS), eig_sig=EIG_SIG,
             dim=np.array(float(np.prod(xu.shape[1:]))), **res)
    print('saved', a.out, {k: v.shape for k, v in res.items()}, f'{time.time() - t0:.0f}s', flush=True)
