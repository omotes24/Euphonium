# check_euphonium.py -- check the 10-pass implementation (euphonium/score.py) against the code that produced the paper's statistics
#   (1) the same images through experiments/core.py: rails() on the scale grid of the runs (the full-grid statistics of extract2.py,
#       restricted afterwards to the atoms): the per-atom errors must agree to float rounding;
#   (2) optionally, the exported statistics of the GPU run of the paper for those images (inputs/feat/<run>/<set>.npz; fp16 on the
#       GPU there, so the agreement is to the fp16 rounding of the network).
#   python experiments/check_euphonium.py --model edm:edm-cifar10-32x32-uncond-vp.pkl --data cifar10_test --export inputs/feat/F_edmc10/cifar10_test.npz --n 4 [--cpu]
import os, sys, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import core
from euphonium import euphonium, load_rule, scales, sigma_value

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', required=True)
p.add_argument('--export', default='', help='exported statistics of the paper run for this set (inputs/feat/<run>/<set>.npz)')
p.add_argument('--n', type=int, default=4, help='number of images (the first n images of the export, or of the set)')
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--cpu', action='store_true')
a = p.parse_args()
dev = torch.device('cpu' if a.cpu else f'cuda:{a.gpu}')
if a.cpu: torch.set_num_threads(os.cpu_count())
rule = load_rule(); atoms = rule['atoms']
den0, net = core.load_model(a.model, dev)
calls = [0]
def den(y, s): calls[0] += y.shape[0]; return den0(y, s)
full = np.load(f'{core.NPY}/{a.data}.npy', mmap_mode='r')
if a.export:
    Z = np.load(a.export, allow_pickle=True); idx = Z['idx'][:a.n]
else:
    Z, idx = None, np.arange(a.n)
x = core.to_t(np.ascontiguousarray(full[idx]), dev)
# (1) the 10-pass implementation
calls[0] = 0
r = euphonium(den, x, rule)
nfe_img = calls[0] / len(idx)
# (2) the full-grid statistics of the runs (rails over sigma_1..sigma_kmax with every rotation the atoms use, bands as in extract2.py)
kmax = max(k for _, k in atoms)
sig = np.array([sigma_value(k) for k in range(kmax + 1)])
tfs = ['id'] + sorted(set(T for T, _ in atoms))
st = core.rails(den0, x, sig, 1, tfs, rule['seed'], core.Bands(x.shape[-1], dev), True, anti=rule['noise'] == 'pair')
e_grid = {t: (st['N'][None].double() + st[f'A_{t}'].double() - 2 * st[f'C_{t}'].double()).sum(-1)[:, :, 0] for t in tfs}   # [B, K]
print(f'{a.data}: {len(idx)} images (idx {idx.tolist()}), {nfe_img:g} forward passes per image (rule: {rule["nfe"]})')
worst = 0.0
for (T, k), j in zip(atoms, range(len(atoms))):
    mine_up, mine_rot = r['e_up'][k].cpu().double(), r['e_rot'][(T, k)].cpu().double()
    ref_up, ref_rot = e_grid['id'][:, k], e_grid[T][:, k]
    d = max(float((mine_up / ref_up - 1).abs().max()), float((mine_rot / ref_rot - 1).abs().max()))
    worst = max(worst, d)
    print(f'  atom ({T}, sigma_{k + 1}): max relative difference to rails() {d:.2e}')
eu_grid = sum(torch.log(e_grid[T][:, k] / e_grid['id'][:, k]) for T, k in atoms)
print(f'  Eu: implementation {np.round(r["eu"].cpu().numpy(), 5).tolist()}')
print(f'      rails()        {np.round(eu_grid.numpy(), 5).tolist()}   max |difference| {float((r["eu"].cpu().double() - eu_grid).abs().max()):.2e}')
if Z is not None:
    tz = [str(t) for t in Z['tfs']]; e = Z['e'][:a.n, :, 0, :].astype(np.float64)
    eu_run = sum(np.log(e[:, k, tz.index(T)] / e[:, k, tz.index('id')]) for T, k in atoms)
    lr_run = np.stack([np.log(e[:, k, tz.index(T)] / e[:, k, tz.index('id')]) for T, k in atoms], 1)
    print(f'      paper run      {np.round(eu_run, 5).tolist()}   max |difference| {float(np.abs(r["eu"].cpu().numpy() - eu_run).max()):.2e}'
          f'   (per atom {np.abs(r["log_ratios"].cpu().numpy() - lr_run).max(0).round(5).tolist()})')
assert nfe_img == rule['nfe'], nfe_img
assert worst < 1e-4, worst
print('OK')
