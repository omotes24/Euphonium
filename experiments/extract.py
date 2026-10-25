# extract.py -- compute per-scale statistics for one (model, dataset) and save an npz
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
p.add_argument('--band_tfs', default='id')
p.add_argument('--K', type=int, default=16)
p.add_argument('--smin', type=float, default=0.01)
p.add_argument('--smax', type=float, default=31.6228)
p.add_argument('--draws', type=int, default=2)
p.add_argument('--hutch', type=int, default=0)
p.add_argument('--power', type=int, default=0)
p.add_argument('--noclean', action='store_true')
p.add_argument('--nobands', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--bs', type=int, default=500)
p.add_argument('--pre', default='none')          # test-time input transform of the data set itself (e.g. vflip as an OOD set)
p.add_argument('--complement', action='store_true')
p.add_argument('--cap', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()

if os.path.exists(a.out):
    print('exists', a.out); sys.exit(0)
dev = torch.device(f'cuda:{a.gpu}')
torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
xu, idx = select(a.data, a.n, a.subseed, a.complement, a.cap)
sigmas = np.exp(np.linspace(np.log(a.smin), np.log(a.smax), a.K))
tfs = a.tfs.split(','); band_tfs = [t for t in a.band_tfs.split(',') if t in tfs]
res, t0 = {}, time.time()
bands = None
for i in range(0, len(xu), a.bs):
    x = to_t(xu[i:i + a.bs], dev)
    if a.pre != 'none': x = TF[a.pre][0](x)
    if bands is None and not a.nobands: bands = Bands(x.shape[-1], dev)
    st = scale_stats(den, x, sigmas, a.draws, tfs, band_tfs, a.seed, bands, not a.noclean, a.hutch, a.power)
    for k, v in st.items(): res.setdefault(k, []).append(v.numpy())
    if i == 0 or (i // a.bs) % 5 == 0:
        el = time.time() - t0
        print(f'{a.data} {i + len(x)}/{len(xu)} {el:.0f}s eta {el / (i + len(x)) * (len(xu) - i - len(x)):.0f}s', flush=True)
res = {k: np.concatenate(v) for k, v in res.items()}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, sigmas=sigmas, idx=idx, tfs=np.array(tfs), meta=np.array(str(vars(a))), **res)
print('saved', a.out, {k: v.shape for k, v in res.items()}, f'{time.time() - t0:.0f}s', flush=True)
