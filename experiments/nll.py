# nll.py -- probability-flow ODE log-likelihood (Heun, Hutchinson trace) for an EDM-type denoiser
#   dx/dsigma = (x - D(x, sigma)) / sigma ,   log p_smin(x) = log N(x_T; 0, smax^2 I) + int_smin^smax div f dsigma
import os, sys, time, argparse, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import *

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=None)
p.add_argument('--subseed', type=int, default=0)
p.add_argument('--steps', type=int, default=64)
p.add_argument('--smin', type=float, default=0.002)
p.add_argument('--smax', type=float, default=80.0)
p.add_argument('--rho', type=float, default=7.0)
p.add_argument('--probes', type=int, default=1)
p.add_argument('--grid', default='rho')            # 'rho' (EDM schedule) or 'log' (uniform in log sigma)
p.add_argument('--bs', type=int, default=125)
p.add_argument('--pre', default='none')
p.add_argument('--pairprobe', action='store_true')   # with --pre T: use the transformed Hutchinson probe (paired with the run on the untransformed image)
p.add_argument('--complement', action='store_true')
p.add_argument('--cap', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
if os.path.exists(a.out): print('exists', a.out); sys.exit(0)
core.FP16 = False
dev = torch.device(f'cuda:{a.gpu}')
den, net = load_model(a.model, dev)
if hasattr(net, 'fp16'): net.fp16 = False
xu, idx = select(a.data, a.n, a.subseed, a.complement, a.cap)
i = torch.arange(a.steps + 1, dtype=torch.float64)
sig = (a.smin ** (1 / a.rho) + i / a.steps * (a.smax ** (1 / a.rho) - a.smin ** (1 / a.rho))) ** a.rho      # increasing


if a.grid == 'log':                                   # uniform in u = log(sigma):  dx/du = x - D(x, sigma)
    sig = torch.exp(torch.linspace(math.log(a.smin), math.log(a.smax), a.steps + 1, dtype=torch.float64))


def f_and_div(x, s, vs):
    """drift f=(x-D)/s (or x-D on the log grid) and Hutchinson estimate of div f (mean over probes)."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        f = (x - den(x, torch.full([x.shape[0]], float(s), device=x.device))) / (1.0 if a.grid == 'log' else float(s))
        div = torch.zeros(x.shape[0], device=x.device)
        for j, v in enumerate(vs):
            g, = torch.autograd.grad((f * v).sum(), x, retain_graph=(j + 1 < len(vs)))
            div += (g * v).flatten(1).sum(1) / len(vs)
    return f.detach(), div.detach()


res, t0 = [], time.time()
for b in range(0, len(xu), a.bs):
    x = to_t(xu[b:b + a.bs], dev)
    if a.pre != 'none': x = TF[a.pre][0](x)
    dim = float(np.prod(x.shape[1:]))
    g = torch.Generator(device='cpu').manual_seed(1234)
    vs = [(torch.randint(0, 2, x.shape[1:], generator=g).float() * 2 - 1).to(dev)[None] for _ in range(a.probes)]
    if a.pairprobe and a.pre != 'none': vs = [TF[a.pre][0](v) for v in vs]      # probe of the transformed copy = transformed probe
    vs = [v.expand_as(x) for v in vs]
    logp = torch.zeros(x.shape[0], device=dev, dtype=torch.float64)
    for k in range(a.steps):
        s0, s1 = float(sig[k]), float(sig[k + 1]); h = math.log(s1 / s0) if a.grid == 'log' else s1 - s0
        f0, d0 = f_and_div(x, s0, vs)
        f1, d1 = f_and_div(x + h * f0, s1, vs)
        x = x + h * 0.5 * (f0 + f1)
        logp += h * 0.5 * (d0 + d1).double()
    prior = -0.5 * (x.double() / a.smax).square().flatten(1).sum(1) - dim * math.log(a.smax) - 0.5 * dim * math.log(2 * math.pi)
    ll = (prior + logp).cpu().numpy()
    res.append(ll)
    bpd = -ll / dim / math.log(2) + math.log2(127.5)
    print(f'{a.data} {b + len(x)}/{len(xu)} bpd {bpd.mean():.3f} {time.time() - t0:.0f}s', flush=True)
ll = np.concatenate(res)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, ll=ll, idx=idx, bpd=-ll / dim / math.log(2) + math.log2(127.5), meta=np.array(str(vars(a))))
print('saved', a.out, 'mean bpd', float((-ll / dim / math.log(2) + math.log2(127.5)).mean()),
      'peak mem GB %.2f' % (torch.cuda.max_memory_allocated(dev) / 1e9), f'{time.time() - t0:.0f}s', flush=True)
