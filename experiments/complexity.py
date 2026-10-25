# complexity.py -- lossless code lengths (bits) of every image: PNG (max compression) and JPEG2000 (reversible), for the input-complexity baseline
import os, sys, io, glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

NPY = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'npy')
OUT = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'complexity'); os.makedirs(OUT, exist_ok=True)


def bits(a):
    im = Image.fromarray(a if a.ndim == 3 else a)
    b = io.BytesIO(); im.save(b, format='PNG', optimize=True, compress_level=9); png = 8 * b.tell()
    try:
        b = io.BytesIO(); im.save(b, format='JPEG2000', irreversible=False); j2k = 8 * b.tell()
    except Exception:
        j2k = -1
    return png, j2k


if __name__ == '__main__':
    names = sys.argv[1:] or [os.path.basename(f)[:-4] for f in sorted(glob.glob(f'{NPY}/*.npy'))
                             if not f.endswith('_y.npy') and 'imagenet32_train' not in f]
    for n in names:
        if os.path.exists(f'{OUT}/{n}.npz'): continue
        x = np.load(f'{NPY}/{n}.npy')
        if x.ndim == 4 and x.shape[-1] == 1: x = x[..., 0]
        with Pool(20) as p: r = np.array(p.map(bits, list(x), chunksize=512))
        np.savez(f'{OUT}/{n}.npz', png=r[:, 0], j2k=r[:, 1])
        print(n, x.shape, 'png bits/dim %.3f' % (r[:, 0].mean() / x[0].size), 'j2k bits/dim %.3f' % (r[:, 1].mean() / x[0].size), flush=True)
    print('COMPLEXITY_DONE')
