# extract2.py -- two-rail statistics (A, C per scale / pair / DCT band / transform) for one (model, dataset)
import os, sys, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=None)
p.add_argument('--subseed', type=int, default=0)
p.add_argument('--tfs', default='id')
p.add_argument('--K', type=int, default=16)
p.add_argument('--smin', type=float, default=0.01)
p.add_argument('--smax', type=float, default=31.6228)
p.add_argument('--sigmas', default='')            # explicit comma-separated list overrides K/smin/smax
p.add_argument('--pairs', type=int, default=2)
p.add_argument('--nobands', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--bs', type=int, default=250)
p.add_argument('--pre', default='none')
p.add_argument('--start', type=int, default=0)     # optional slice of the data set (for sharding across GPUs)
p.add_argument('--stop', type=int, default=None)
p.add_argument('--complement', action='store_true')   # use the images outside the development subset (see below)
p.add_argument('--cap', type=int, default=0)
p.add_argument('--alts', default='')               # e.g. ve=edm:xxx.pkl,mc1=edmmc:yyy.pkl  (extra denoisers on the same noised inputs)
p.add_argument('--gpairs', default='')             # e.g. id:ve,id:mc1,mc1:mc2          (disagreement energies)
p.add_argument('--noanti', action='store_true')    # ablation: single noise draw instead of the antithetic pair
p.add_argument('--perimage', action='store_true')  # ablation: independent noise for every test image instead of one shared noise image
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()

if os.path.exists(a.out):
    print('exists', a.out); sys.exit(0)
dev = torch.device(f'cuda:{a.gpu}')
torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
alts, keep = {}, [net]
for s in [s for s in a.alts.split(',') if s]:
    nm, spec = s.split('=', 1)
    alts[nm], n2 = load_model(spec, dev); keep.append(n2)
gpairs = [tuple(s.split(':')) for s in a.gpairs.split(',') if s]
xu, idx = select(a.data, a.n, a.subseed, a.complement, a.cap)
xu, idx = xu[a.start:a.stop], idx[a.start:a.stop]
sigmas = np.array([float(s) for s in a.sigmas.split(',')]) if a.sigmas else np.exp(np.linspace(np.log(a.smin), np.log(a.smax), a.K))
tfs = a.tfs.split(',')
res, t0, bands, Nn = {}, time.time(), None, None
for i in range(0, len(xu), a.bs):
    x = to_t(xu[i:i + a.bs], dev)
    if a.pre != 'none': x = TF[a.pre][0](x)
    if bands is None and not a.nobands: bands = Bands(x.shape[-1], dev)
    st = rails(den, x, sigmas, a.pairs, tfs, a.seed, bands, True, alts, gpairs, anti=not a.noanti, perimage=a.perimage, offset=i)
    Nn = st.pop('N').numpy()
    for k, v in st.items(): res.setdefault(k, []).append(v.numpy().astype(np.float32))
    if (i // a.bs) % 4 == 0:
        el = time.time() - t0
        print(f'{a.data} {i + len(x)}/{len(xu)} {el:.0f}s eta {el / (i + len(x)) * (len(xu) - i - len(x)):.0f}s', flush=True)
res = {k: np.concatenate(v) for k, v in res.items()}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, sigmas=sigmas, idx=idx, tfs=np.array(tfs), N=Nn, dim=np.array(float(np.prod(xu.shape[1:]))),
         meta=np.array(str(vars(a))), **res)
print('saved', a.out, {k: v.shape for k, v in res.items()}, f'{time.time() - t0:.0f}s', flush=True)
