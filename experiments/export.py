# export.py -- the compact per-image statistics that the analysis reads: runs/ -> inputs/ (or $EUPHONIUM_INPUTS)
#   feat/<run>/<set>.npz : e[B,K,P,T] = ||eps - eps_hat(T)||^2 summed over the pixels (the DCT band sums of extract2.py added
#                          back) per image, scale, antithetic draw and transform; tfs, sigmas, idx, dim, meta (the run's arguments)
#   nll/, base/, base2/, eu/ : copied as they are;   complexity/<set>.npz : PNG / JPEG 2000 bits of every image (data/complexity/)
#   The release asset inputs-v1 holds the files of inputs/ that the analysis reads, with the arrays it reads.
#   usage (repository root): python experiments/export.py
import os, glob, shutil
import numpy as np

R = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('EUPHONIUM_INPUTS', f'{R}/inputs')
os.makedirs(OUT, exist_ok=True)


def e_from(z, t):
    A, C = z[f'A_{t}'].astype(np.float64), z[f'C_{t}'].astype(np.float64)          # [B,K,P,nb]
    N = z['Ni'].astype(np.float64) if 'Ni' in z.files else z['N'].astype(np.float64)[None]
    return (N + A - 2 * C).sum(-1)                                                  # [B,K,P]


sets = set()
for d in sorted(glob.glob(f'{R}/runs/feat/*')):
    tag = os.path.basename(d)
    os.makedirs(f'{OUT}/feat/{tag}', exist_ok=True)
    for f in sorted(glob.glob(f'{d}/*.npz')):
        out = f'{OUT}/feat/{tag}/{os.path.basename(f)}'
        if os.path.exists(out): continue
        z = np.load(f, allow_pickle=True)
        tfs = [k[2:] for k in z.files if k.startswith('A_') and not k.startswith('A_@')]
        e = np.stack([e_from(z, t) for t in tfs], -1).astype(np.float32)           # [B,K,P,T]
        o = dict(e=e, tfs=np.array(tfs), sigmas=z['sigmas'], idx=z['idx'], dim=z['dim'])
        if 'meta' in z.files: o['meta'] = z['meta']
        np.savez_compressed(out, **o)
        sets.add(os.path.basename(f)[:-4].split('__')[0])
    print(tag, len(glob.glob(f'{d}/*.npz')), flush=True)
for sub in ('nll', 'base', 'base2', 'eu'):
    if os.path.isdir(f'{R}/runs/{sub}'): shutil.copytree(f'{R}/runs/{sub}', f'{OUT}/{sub}', dirs_exist_ok=True)
os.makedirs(f'{OUT}/complexity', exist_ok=True)
for f in sorted(glob.glob(f'{R}/data/complexity/*.npz')): shutil.copy(f, f'{OUT}/complexity/')
print('sets', sorted(sets)); print('EXPORT_DONE', flush=True)
