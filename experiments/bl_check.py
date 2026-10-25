# bl_check.py -- numerical checks of baselines.py against the official code paths, on the DiffPath CelebA model
#   (improved-diffusion, cosine schedule, 4000 steps; the model both official repositories use for 32x32 inputs)
#   (1) DiffPath : eps of our sigma-coordinate DDIM port vs improved-diffusion's SpacedDiffusion('ddim10') stepped with DiffPath's
#                  ddim_reverse_sample (clip_denoised=True, eps re-derived from the clipped x0-prediction)
#   (2) GEPC     : our features vs the official gepc.py (_gepc_at_t) on the same noise.  The official adapter interface is
#                  emulated; the UNet gets the training-time timestep scaling t * 1000 / 4000.
#   (3) EigenScore: eigenvalue estimates after 3 subspace iterations vs 30 iterations
# usage: SR_FP16=0 python experiments/bl_check.py <gpu>
import os, sys, importlib.util
import numpy as np
import torch
assert os.environ.get('SR_FP16') == '0', 'run with SR_FP16=0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import baselines as BL

dev = torch.device('cuda:%s' % (sys.argv[1] if len(sys.argv) > 1 else '2'))
den, model = load_model('iddpm:diffpath_celeba32.pt', dev)
xu, _ = select('celeba_test', 16)
x = to_t(xu, dev)
sys.path.insert(0, f'{ROOT}/ext/improved-diffusion')
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.gaussian_diffusion import _extract_into_tensor as ext

# ---------------------------------------------------------------- (1) DiffPath
sp = create_gaussian_diffusion(steps=4000, learn_sigma=True, noise_schedule='cosine', rescale_timesteps=True,
                               rescale_learned_sigmas=True, timestep_respacing='ddim10')
with torch.no_grad():
    img, E_off = x.clone(), []
    for i in range(sp.num_timesteps):
        t = torch.full((x.shape[0],), i, device=dev, dtype=torch.long)
        out = sp.p_mean_variance(model, img, t, clip_denoised=True)
        eps = (ext(sp.sqrt_recip_alphas_cumprod, t, img.shape) * img - out['pred_xstart']) / ext(sp.sqrt_recipm1_alphas_cumprod, t, img.shape)
        abn = ext(sp.alphas_cumprod_next, t, img.shape)
        img = out['pred_xstart'] * torch.sqrt(abn) + torch.sqrt(1 - abn) * eps
        E_off.append(eps)
    E_off = torch.stack(E_off, 1)
    E_our = BL.diffpath_eps(den, x)
print('timesteps official', sorted(sp.use_timesteps), ' ours', BL.DP_T)
rel = [float((E_our[:, k] - E_off[:, k]).norm() / E_off[:, k].norm()) for k in range(10)]
print('(1) DiffPath eps relative difference per step:', ' '.join('%.1e' % r for r in rel))
f_off = (E_off.double() ** 2).flatten(1).sum(1); f_our = (E_our.double() ** 2).flatten(1).sum(1)
dO = (E_off[:, 1:] - E_off[:, :-1]).double() * 10; dU = (E_our[:, 1:] - E_our[:, :-1]).double() * 10
print('    eps_sum_sq rel diff %.1e   deps_dt_sq_sqrt rel diff %.1e' % (float(((f_our - f_off) / f_off).abs().max()),
      float(((dU.flatten(1).norm(dim=1) - dO.flatten(1).norm(dim=1)) / dO.flatten(1).norm(dim=1)).abs().max())))

# ---------------------------------------------------------------- (2) GEPC
spec = importlib.util.spec_from_file_location('gepc_off', f'{ROOT}/ext/gepc_repo/gepc/methods/gepc.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
base = create_gaussian_diffusion(steps=4000, learn_sigma=True, noise_schedule='cosine', rescale_timesteps=True, rescale_learned_sigmas=True)


class Adapter:
    def __init__(self):
        self.alphas_cumprod = torch.tensor(np.cumprod(1.0 - np.asarray(base.betas, np.float32)), dtype=torch.float32, device=dev)
        self.model = lambda xt, t: model(xt, t.float() * (1000.0 / 4000))
    def sigma_t(self, t): return torch.sqrt(1.0 - self.alphas_cumprod[int(t)])


ad = Adapter()
print('    official snapped timesteps (cifar10 + celeba configs):',
      sorted({int(np.argmin(np.abs(np.sqrt(ad.alphas_cumprod.cpu().numpy()) - s))) for s in BL.GEPC_SNR}), ' ours', BL.GEPC_T)
gen = torch.Generator(device='cpu').manual_seed(123)
NOISE = {t: torch.randn(x.shape, generator=gen).to(dev) for t in BL.GEPC_T}
G._forward_noisy = lambda adapter, x0, t_idx, gen=None: (torch.sqrt(adapter.alphas_cumprod[int(t_idx)]) * x0
                                                         + torch.sqrt(1 - adapter.alphas_cumprod[int(t_idx)]) * NOISE[int(t_idx)])
ours = BL.gepc(den, x, torch.Generator(device='cpu').manual_seed(123))
for j, rho in enumerate(BL.POOLS):
    m = G.GEPC(features=('gepc_s', 'gepc_s_cos', 'gepc_s_pair'), spatial_pool='topk' if rho > 0 else 'mean', topk_rho=rho,
               group_shifts=True, shift_px=1, amp='fp32', verbose=False)
    worst = {}
    for l, t in enumerate(BL.GEPC_T):
        f = m._gepc_at_t(ad, x, t)
        mine = dict(gepc_s=ours['num'][:, l, :, j].sum(1) / 7 / ours['den'][:, l, j],
                    gepc_s_cos=(1 - ours['cos'][:, l]).sum(1) / 7,
                    gepc_s_pair=ours['pair'][:, l, j] / ours['den'][:, l, j])
        for k, v in mine.items():
            o = f[k].double().cpu()
            worst[k] = max(worst.get(k, 0), float(((v - o).abs() / o.abs().clamp_min(1e-12)).max()))
    print('(2) GEPC pool=%s  max relative difference over 6 levels x 16 images:' % (('top%.2f' % rho) if rho else 'mean'),
          ' '.join('%s %.1e' % kv for kv in worst.items()))
print('    gepc_s at the 6 levels (image 0, top0.3):', ' '.join('%.4f' % float(ours['num'][0, l, :, 1].sum() / 7 / ours['den'][0, l, 1]) for l in range(6)))

# ---------------------------------------------------------------- (3) EigenScore convergence
l3 = BL.eigen(den, x[:8], torch.Generator(device='cpu').manual_seed(7), 1, 3, 3)['lam'][:, :, 0]
l30 = BL.eigen(den, x[:8], torch.Generator(device='cpu').manual_seed(7), 1, 30, 3)['lam'][:, :, 0]
s2 = torch.tensor(BL.EIG_SIG ** 2)[None, :, None]
print('(3) EigenScore: sigma^2 =', np.round(BL.EIG_SIG ** 2, 3).tolist())
print('    mean lambda_k / sigma^2 (rows: timesteps), 3 iters :', np.round((l3 / s2).mean(0).numpy(), 3).tolist())
print('    mean lambda_k / sigma^2 (rows: timesteps), 30 iters:', np.round((l30 / s2).mean(0).numpy(), 3).tolist())
m3, m30 = l3.sum(2).numpy(), l30.sum(2).numpy()
rk = lambda v: np.argsort(np.argsort(v))
print('    top-3 sum, ratio 3 / 30 iters per timestep:', np.round((m3 / m30).mean(0), 3).tolist(),
      ' rank corr over 8 images:', [round(float(np.corrcoef(rk(m3[:, i]), rk(m30[:, i]))[0, 1]), 2) for i in range(5)])
print('BL_CHECK_DONE')
