# extract_bu.py -- extract2.py restricted to a subset of the 16-scale grid (the training snapshots of A.7: identity + R180 at the grid
#   indices 0,2,3,5,7 = sigma_1, sigma_3, sigma_4, sigma_6, sigma_8, antithetic pair: 20 NFE/image)
#   while keeping the shared-noise draws of the full grid (shared_noise is keyed by the scale INDEX of the 16-scale grid), and
#   optionally restricted to an explicit list of image indices (the common images of the paper's tables).
#   usage: python experiments/extract_bu.py --kidx 0,2,3,5,7 [--idxjson <file> --idxkey <tier>:<set>] <extract2.py arguments except --sigmas>
#   With --idxjson/--idxkey the --n/--complement/--cap selection is ignored and the listed indices of --data are used (sorted).
import os, sys, json, runpy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core

args = sys.argv[1:]


def take(flag):
    if flag not in args: return None
    i = args.index(flag); v = args[i + 1]; del args[i:i + 2]; return v


kidx = [int(k) for k in (take('--kidx') or '0,2,3,5,7').split(',')]
idxjson, idxkey = take('--idxjson'), take('--idxkey')
S16 = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
sig = ','.join('%.6g' % S16[k] for k in kidx)

_shared_noise = core.shared_noise
def shared_noise(shape, seed, k, j, device):                      # k = position in the reduced list -> index of the 16-scale grid
    return _shared_noise(shape, seed, kidx[k], j, device)
core.shared_noise = shared_noise

if idxjson:
    tier, name = idxkey.split(':', 1)
    IDX = np.array(sorted(int(i) for i in json.load(open(idxjson))[tier][name]), dtype=np.int64)
    def select(name, n=None, seed=0, complement=False, cap=0):
        full = np.load(f'{core.NPY}/{name}.npy', mmap_mode='r')
        assert IDX.max() < len(full), (name, IDX.max(), len(full))
        return np.ascontiguousarray(full[IDX]), IDX
    core.select = select

sys.argv = ['extract2.py'] + args + ['--sigmas', sig]
runpy.run_path(os.path.join(HERE, 'extract2.py'), run_name='__main__')
