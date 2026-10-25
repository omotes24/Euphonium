# eig_conv.py -- convergence of the EigenScore subspace iteration: top-3 eigenvalue sums after n iterations vs 40 iterations
# usage: SR_FP16=0 python experiments/eig_conv.py <gpu> <model spec> <data>
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import baselines as BL
dev = torch.device('cuda:' + sys.argv[1])
den, net = load_model(sys.argv[2], dev)
xu, _ = select(sys.argv[3], 24)
x = to_t(xu, dev)
rk = lambda v: np.argsort(np.argsort(v))
ref = BL.eigen(den, x, torch.Generator(device='cpu').manual_seed(7), 1, 40, 3)['lam'][:, :, 0].sum(2).numpy()
for n in (2, 3, 4, 6, 8, 12):
    m = BL.eigen(den, x, torch.Generator(device='cpu').manual_seed(7), 1, n, 3)['lam'][:, :, 0].sum(2).numpy()
    print('%s iters %2d  ratio to 40 iters per timestep %s  rank corr %s' % (sys.argv[2][:12], n, np.round((m / ref).mean(0), 3).tolist(),
          [round(float(np.corrcoef(rk(m[:, i]), rk(ref[:, i]))[0, 1]), 2) for i in range(5)]), flush=True)
# noise-realisation spread: two repetitions at 40 iterations
r2 = BL.eigen(den, x, torch.Generator(device='cpu').manual_seed(8), 1, 40, 3)['lam'][:, :, 0].sum(2).numpy()
print('rank corr between two noise draws (40 iters):', [round(float(np.corrcoef(rk(r2[:, i]), rk(ref[:, i]))[0, 1]), 2) for i in range(5)])
