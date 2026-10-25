# prep_gray.py -- 28x28 grey-scale sets for the FashionMNIST tier (uint8 [N,28,28])
import os, gzip, subprocess, zipfile, tarfile, glob
import numpy as np
from PIL import Image

ROOT = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'); RAW, OUT = f'{ROOT}/raw/gray', f'{ROOT}/npy'
os.makedirs(RAW, exist_ok=True)


def get(url, dst):
    if not os.path.exists(dst):
        subprocess.run(['curl', '-sL', '--fail', '--retry', '5', '-o', dst + '.part', url], check=True)
        os.rename(dst + '.part', dst)
    return dst


def idx_images(path):
    with gzip.open(path, 'rb') as f:
        b = f.read()
    return np.frombuffer(b, np.uint8, offset=16).reshape(-1, 28, 28)


def idx_labels(path):
    with gzip.open(path, 'rb') as f:
        b = f.read()
    return np.frombuffer(b, np.uint8, offset=8)


SRC = {
    'fmnist': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion',
    'kmnist': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist',
    'mnist': 'https://ossci-datasets.s3.amazonaws.com/mnist',
}
for name, base in SRC.items():
    for s, pre in [('train', 'train'), ('test', 't10k')]:
        try:
            if not os.path.exists(f'{OUT}/g_{name}_{s}.npy'):
                x = idx_images(get(f'{base}/{pre}-images-idx3-ubyte.gz', f'{RAW}/{name}_{pre}_x.gz'))
                y = idx_labels(get(f'{base}/{pre}-labels-idx1-ubyte.gz', f'{RAW}/{name}_{pre}_y.gz'))
                np.save(f'{OUT}/g_{name}_{s}.npy', x); np.save(f'{OUT}/g_{name}_{s}_y.npy', y)
            print(name, s, np.load(f'{OUT}/g_{name}_{s}.npy').shape, flush=True)
        except Exception as e:
            print('FAILED', name, s, e, flush=True)
try:
    if not os.path.exists(f'{OUT}/g_omniglot_eval.npy'):
        z = get('https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip', f'{RAW}/omniglot_eval.zip')
        with zipfile.ZipFile(z) as zf: zf.extractall(f'{RAW}/omniglot')
        ps = sorted(glob.glob(f'{RAW}/omniglot/images_evaluation/*/*/*.png'))
        xs = [255 - np.asarray(Image.open(q).convert('L').resize((28, 28), Image.BILINEAR)) for q in ps]   # white strokes on black
        np.save(f'{OUT}/g_omniglot_eval.npy', np.stack(xs).astype(np.uint8))
    print('omniglot', np.load(f'{OUT}/g_omniglot_eval.npy').shape, flush=True)
except Exception as e:
    print('FAILED omniglot', e, flush=True)
try:
    if not os.path.exists(f'{OUT}/g_notmnist.npy'):
        z = get('http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz', f'{RAW}/notMNIST_small.tar.gz')
        with tarfile.open(z) as t: t.extractall(f'{RAW}/notmnist')
        xs = []
        for q in sorted(glob.glob(f'{RAW}/notmnist/notMNIST_small/*/*.png')):
            try: xs.append(np.asarray(Image.open(q).convert('L')))
            except Exception: pass
        np.save(f'{OUT}/g_notmnist.npy', np.stack(xs).astype(np.uint8))
    print('notmnist', np.load(f'{OUT}/g_notmnist.npy').shape, flush=True)
except Exception as e:
    print('FAILED notmnist', e, flush=True)
print('GRAY_DONE')
