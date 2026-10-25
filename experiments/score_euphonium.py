# score_euphonium.py -- Euphonium scores of one image set under one diffusion model (10 forward passes per image)
#   python experiments/score_euphonium.py --model edm:edm-cifar10-32x32-uncond-vp.pkl --data cifar10_test --n 3000 --complement --cap 5000 \
#          --out runs/eu/edmc10/cifar10_test.npz [--pre vflip] [--idx <json list of image indices>] [--bs 500] [--gpu 0 | --cpu] [--fuse]
#   Image selection flags are those of extract2.py (core.select: --n / --complement / --cap), so the same command line selects the
#   same images as the runs of the paper.  Output: eu [N], log_ratios [N, n_atoms], e_up [N, n_scales], e_rot [N, n_atoms], idx [N]
#   (with --pred_diff also pd [N, n_atoms] = ||T^-1 epshat(T x_s) - epshat(x_s)||^2 and a_up [N, n_scales] = ||epshat(x_s)||^2).
#   Environment: SR_FP16=0 evaluates the network in fp32 (the paper's runs used fp16 autocast except for CIFAR-100).
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from core import load_model, select, to_t, TF as CORE_TF, NPY
from euphonium import euphonium, load_rule, scales, describe

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=None)
p.add_argument('--subseed', type=int, default=0)
p.add_argument('--complement', action='store_true')
p.add_argument('--cap', type=int, default=0)
p.add_argument('--idx', default='', help='json file with a list of image indices of --data (overrides --n/--complement/--cap)')
p.add_argument('--pre', default='none', help='transform applied to the images first (vflip: the upside-down ID set)')
p.add_argument('--rule', default='', help='rule.json to use (default: the repository rule)')
p.add_argument('--bs', type=int, default=500)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--cpu', action='store_true')
p.add_argument('--fuse', action='store_true', help='one network call per scale (faster; numerically equal up to float rounding)')
p.add_argument('--pred_diff', action='store_true', help='also store the prediction differences of the atoms (same passes; sign-or-magnitude control)')
a = p.parse_args()

if os.path.exists(a.out):
    print('exists', a.out); sys.exit(0)
dev = torch.device('cpu' if a.cpu else f'cuda:{a.gpu}')
if not a.cpu: torch.backends.cudnn.benchmark = True
rule = load_rule(a.rule or None)
den, net = load_model(a.model, dev)
if a.idx:
    idx = np.array(sorted(int(i) for i in json.load(open(a.idx))), dtype=np.int64)
    xu = np.ascontiguousarray(np.load(f'{NPY}/{a.data}.npy', mmap_mode='r')[idx])
else:
    xu, idx = select(a.data, a.n, a.subseed, a.complement, a.cap)
print(f'{a.data}: {len(idx)} images, rule {describe(rule["atoms"])}, {rule["nfe"]} forward passes per image', flush=True)
ks, out, t0 = scales(rule['atoms']), {'eu': [], 'log_ratios': [], 'e_up': [], 'e_rot': []}, time.time()
if a.pred_diff: out.update(pd=[], a_up=[])
for i in range(0, len(xu), a.bs):
    x = to_t(xu[i:i + a.bs], dev)
    if a.pre != 'none': x = CORE_TF[a.pre][0](x)
    r = euphonium(den, x, rule, fuse=a.fuse, pred_diff=a.pred_diff)
    out['eu'].append(r['eu'].cpu().numpy()); out['log_ratios'].append(r['log_ratios'].cpu().numpy())
    out['e_up'].append(torch.stack([r['e_up'][k] for k in ks], 1).cpu().numpy())
    out['e_rot'].append(torch.stack([r['e_rot'][(T, k)] for T, k in rule['atoms']], 1).cpu().numpy())
    if a.pred_diff:
        out['pd'].append(torch.stack([r['pd'][(T, k)] for T, k in rule['atoms']], 1).cpu().numpy())
        out['a_up'].append(torch.stack([r['a_up'][k] for k in ks], 1).cpu().numpy())
    el = time.time() - t0
    print(f'{a.data} {i + len(x)}/{len(xu)} {el:.0f}s eta {el / (i + len(x)) * (len(xu) - i - len(x)):.0f}s', flush=True)
res = {k: np.concatenate(v) for k, v in out.items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, idx=idx, atoms=np.array(['%s:%d' % (T, k) for T, k in rule['atoms']]), scales=np.array(ks),
         sigmas=np.array([float('%.6g' % s) for s in np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))[ks]]),
         nfe=np.array(rule['nfe']), meta=np.array(str(vars(a))), **res)
print('saved', a.out, {k: v.shape for k, v in res.items()}, f'{time.time() - t0:.0f}s', flush=True)
