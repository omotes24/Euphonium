# bl_check_q.py -- our GEPC with load_iddpm_rawt vs the official gepc.py with the official adapter call (UNet(x_t, t), raw index)
# usage: SR_FP16=0 python experiments/bl_check_q.py <gpu>
import os, sys, importlib.util
import numpy as np
import torch
assert os.environ.get('SR_FP16') == '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import baselines as BL
dev = torch.device('cuda:%s' % (sys.argv[1] if len(sys.argv) > 1 else '0'))
den, model = BL.load_iddpm_rawt(f'{ROOT}/ckpt/diffpath_celeba32.pt', dev)
xu, _ = select('cifar10_test', 16)
x = to_t(xu, dev)
sys.path.insert(0, f'{ROOT}/ext/improved-diffusion')
from improved_diffusion.script_util import create_gaussian_diffusion
spec = importlib.util.spec_from_file_location('gepc_off', f'{ROOT}/ext/gepc_repo/gepc/methods/gepc.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
base = create_gaussian_diffusion(steps=4000, learn_sigma=True, noise_schedule='cosine', rescale_timesteps=True, rescale_learned_sigmas=True)


class Adapter:                                         # exactly the official call: self.model(x_t, t) with the raw index
    def __init__(self):
        self.alphas_cumprod = torch.tensor(np.cumprod(1.0 - np.asarray(base.betas, np.float32)), dtype=torch.float32, device=dev)
        self.model = model
    def sigma_t(self, t): return torch.sqrt(1.0 - self.alphas_cumprod[int(t)])


ad = Adapter()
gen = torch.Generator(device='cpu').manual_seed(123)
NOISE = {t: torch.randn(x.shape, generator=gen).to(dev) for t in BL.GEPC_T}
G._forward_noisy = lambda adapter, x0, t_idx, gen=None: (torch.sqrt(adapter.alphas_cumprod[int(t_idx)]) * x0
                                                         + torch.sqrt(1 - adapter.alphas_cumprod[int(t_idx)]) * NOISE[int(t_idx)])
ours = BL.gepc(den, x, torch.Generator(device='cpu').manual_seed(123))
m = G.GEPC(features=('gepc_s', 'gepc_s_cos', 'gepc_s_pair'), spatial_pool='topk', topk_rho=0.3, group_shifts=True, shift_px=1, amp='fp32', verbose=False)
worst = {}
for l, t in enumerate(BL.GEPC_T):
    f = m._gepc_at_t(ad, x, t)
    mine = dict(gepc_s=ours['num'][:, l, :, 1].sum(1) / 7 / ours['den'][:, l, 1], gepc_s_cos=(1 - ours['cos'][:, l]).sum(1) / 7,
                gepc_s_pair=ours['pair'][:, l, 1] / ours['den'][:, l, 1])
    for k, v in mine.items():
        o = f[k].double().cpu(); worst[k] = max(worst.get(k, 0), float(((v - o).abs() / o.abs().clamp_min(1e-12)).max()))
print('raw-t GEPC vs official gepc.py (top0.30, 6 levels x 16 images), max relative difference:', ' '.join('%s %.1e' % kv for kv in worst.items()))
print('BL_CHECK_Q_DONE')
