# prep_newid.py -- Tiny-ImageNet train (100,000 images, 200 classes) at 32x32 (prep_data.resize_cc), the training data of the
#   Tiny-ImageNet models of the training-length experiment (A.7; experiments/jobs/TRAIN.txt).  The ID test set of these models is
#   tin_val = the official validation images (prep_data.py).  Needs data/raw/tiny-imagenet-200.zip (prep_data.py downloads it).
import os, sys, glob, zipfile
import numpy as np
from PIL import Image
from multiprocessing import Pool
ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RAW, OUT = f'{ROOT}/data/raw', f'{ROOT}/data/npy'
os.chdir(ROOT)


def resize_cc(img, size=32):
    img = img.convert('RGB'); w, h = img.size; s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    if (nw, nh) != (w, h): img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)))


def _load(path):
    try:
        with Image.open(path) as im: return resize_cc(im)
    except Exception as e:
        print('ERR', path, e, flush=True); return np.zeros((32, 32, 3), np.uint8)


def save(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8); assert arr.shape[1:] == (32, 32, 3), (name, arr.shape)
    np.save(f'{OUT}/{name}.npy', arr); print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


# ---------------------------------------------------------------- Tiny-ImageNet train, 32x32
if not os.path.exists(f'{OUT}/tin_train.npy'):
    z = f'{RAW}/tiny-imagenet-200.zip'
    if not os.path.exists(f'{RAW}/tiny-imagenet-200/train'):
        with zipfile.ZipFile(z) as zf: zf.extractall(RAW)
    paths = sorted(glob.glob(f'{RAW}/tiny-imagenet-200/train/*/images/*.JPEG'))
    print('tin train images', len(paths), flush=True); assert len(paths) == 100000
    with Pool(20) as p: arr = np.stack(p.map(_load, paths, chunksize=256))
    save('tin_train', arr)
    wn = sorted({os.path.basename(os.path.dirname(os.path.dirname(q))) for q in paths}); lab = {w: i for i, w in enumerate(wn)}
    np.save(f'{OUT}/tin_train_y.npy', np.array([lab[os.path.basename(os.path.dirname(os.path.dirname(q)))] for q in paths]))

print('PREP_NEWID_DONE', flush=True)
