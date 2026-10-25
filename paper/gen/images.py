# images.py -- test images for the illustrations (CIFAR-10 / SVHN test sets in the order used by the experiments).
#   The per-image statistics refer to images by their index `idx` in these arrays.  data/npy/<name>.npy (experiments/prep_data.py)
#   is used when present; otherwise the array is rebuilt from the Hugging Face parquet copy and checked against the PNG code lengths
#   of the experiments (inputs/complexity/<name>.npz).  Needs pyarrow and pandas for the download.
import os, io
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS = os.environ.get('EUPHONIUM_INPUTS', os.path.join(ROOT, 'inputs'))
HF = {'cifar10_test': 'https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/test/0.parquet',
      'svhn_test': 'https://huggingface.co/api/datasets/ufldl-stanford/svhn/parquet/cropped_digits/test/0.parquet'}


def _png_bits(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format='PNG', optimize=True, compress_level=9); return 8 * b.tell()


def test_images(name):
    """uint8 [N, 32, 32, 3] in the order used by the experiments"""
    for p in (os.path.join(ROOT, 'data', 'npy', f'{name}.npy'), os.path.join(ROOT, 'data', 'raw', 'hf', f'{name}.npy')):
        if os.path.exists(p): return np.load(p)
    import urllib.request, pyarrow.parquet as pq
    p = os.path.join(ROOT, 'data', 'raw', 'hf', f'{name}.npy'); os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.parquet'; urllib.request.urlretrieve(HF[name], tmp)
    t = pq.read_table(tmp).to_pandas(); col = [c for c in t.columns if 'im' in c.lower()][0]
    X = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB')) for d in t[col]])
    ref = np.load(os.path.join(INPUTS, 'complexity', f'{name}.npz'))['png']
    chk = range(0, len(X), max(1, len(X) // 200))
    assert all(_png_bits(X[i]) == ref[i] for i in chk), 'image order differs from the experiments'
    np.save(p, X); os.remove(tmp)
    return X


def rot90(img, k=1):
    """the experiments' rotation: torch x.rot90(k, (2, 3)) on NCHW == np.rot90 on (H, W), counter-clockwise"""
    return np.rot90(img, k, axes=(0, 1))
