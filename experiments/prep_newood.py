# prep_newood.py -- additional out-of-distribution sets.  In this paper they are selection data only (protocol/PROTOCOL_10NFE.md),
#   never part of the main evaluation.  Arrays follow the conventions of the existing tiers:
#     32x32 RGB  (CIFAR-10 / SVHN / CelebA / CIFAR-100 tiers)  prep_data.resize_cc : shorter side -> 32 (bilinear), centre crop
#       stl10_test (8000), gtsrb_test (12630), flowers102_test (6149), eurosat (27000); mnist_rgb32 already exists
#     64x64 RGB  (ImageNet-64 tier)  prep_in64.crop64 : EDM2 / ADM centre crop ("center-crop-dhariwal")
#       in64_stl10_test, in64_gtsrb_test, in64_flowers102_test, in64_eurosat, in64_mnist_rgb (MNIST test, bilinear to 64)
#     32x32 grey (FashionMNIST tier)  28x28 zero padded by 2 pixels, uint8 [N,32,32] like g32_*
#       g32_emnist_letters (20800), g32_cifar10_gray, g32_svhn_gray (test sets, bilinear 32->28, luminance), g32_gtsrb_gray
#   usage: python experiments/prep_newood.py [stl10 gtsrb flowers eurosat mnist64 emnist gray]
import os, sys, traceback
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets

ROOT = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'); TV, OUT = f'{ROOT}/raw/tv', f'{ROOT}/npy'
os.makedirs(TV, exist_ok=True)


def resize_cc(img, size=32):
    """prep_data.py: shorter side -> size (PIL bilinear, antialiased) then centre crop"""
    img = img.convert('RGB'); w, h = img.size; s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    if (nw, nh) != (w, h): img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)))


def crop64(im, size=64):
    """prep_in64.py: EDM2 / ADM preprocessing (box halvings while >= 2x, bicubic to the shorter side, centre crop)"""
    im = im.convert('RGB')
    while min(*im.size) >= 2 * size: im = im.resize(tuple(x // 2 for x in im.size), resample=Image.BOX)
    s = size / min(*im.size); im = im.resize(tuple(round(x * s) for x in im.size), resample=Image.BICUBIC)
    a = np.asarray(im); y, x = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[y:y + size, x:x + size]


def gray_pad(img):
    """FashionMNIST-tier format: 28x28 luminance, zero padded to 32x32"""
    g = np.asarray(Image.fromarray(resize_cc(img, 28)).convert('L'))
    return np.pad(g, 2)


def done(name): return os.path.exists(f'{OUT}/{name}.npy')


def save(name, arr, shape):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    assert arr.shape[1:] == shape, (name, arr.shape)
    np.save(f'{OUT}/{name}.npy', arr)
    print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)
    montage(name, arr)


def montage(name, arr, n=64):
    """first 64 images as an 8x8 sheet (data/raw/tv/check_<name>.png) for a visual check"""
    a = arr[:n]
    if a.ndim == 3: a = np.repeat(a[..., None], 3, -1)
    s = a.shape[1]; rows = [np.concatenate(list(a[i:i + 8]), 1) for i in range(0, len(a), 8)]
    Image.fromarray(np.concatenate(rows, 0)).resize((8 * s * 2, len(rows) * s * 2), Image.NEAREST).save(f'{TV}/check_{name}.png')


def pil_set(ds, name32, name64, name_g=None):
    """three formats from one pass over a torchvision data set of PIL images"""
    need32, need64, needg = not done(name32), not done(name64), name_g is not None and not done(name_g)
    if not (need32 or need64 or needg): print('exists', name32, name64, name_g); return
    x32, x64, xg = [], [], []
    for i in range(len(ds)):
        im = ds[i][0]
        if need32: x32.append(resize_cc(im))
        if need64: x64.append(crop64(im))
        if needg: xg.append(gray_pad(im))
        if i % 2000 == 0: print(name32, i, len(ds), flush=True)
    if need32: save(name32, np.stack(x32), (32, 32, 3))
    if need64: save(name64, np.stack(x64), (64, 64, 3))
    if needg: save(name_g, np.stack(xg), (32, 32))


def stl10():
    ds = datasets.STL10(TV, split='test', download=True)            # 8000 images, 96x96, uint8 [N,3,96,96]
    ims = [Image.fromarray(a.transpose(1, 2, 0)) for a in ds.data]
    class L:
        def __len__(self): return len(ims)
        def __getitem__(self, i): return ims[i], 0
    pil_set(L(), 'stl10_test', 'in64_stl10_test')


def gtsrb():
    ds = datasets.GTSRB(TV, split='test', download=True)            # 12630 traffic-sign crops of various sizes
    pil_set(ds, 'gtsrb_test', 'in64_gtsrb_test', 'g32_gtsrb_gray')


def flowers():
    ds = datasets.Flowers102(TV, split='test', download=True)       # 6149 images
    pil_set(ds, 'flowers102_test', 'in64_flowers102_test')


def eurosat():
    ds = datasets.EuroSAT(TV, download=True)                        # 27000 Sentinel-2 RGB patches, 64x64
    pil_set(ds, 'eurosat', 'in64_eurosat')


def mnist64():
    if done('in64_mnist_rgb'): print('exists in64_mnist_rgb'); return
    x = np.load(f'{OUT}/g_mnist_test.npy')
    save('in64_mnist_rgb', np.stack([np.asarray(Image.fromarray(a).convert('RGB').resize((64, 64), Image.BILINEAR)) for a in x]), (64, 64, 3))


def emnist():
    if done('g32_emnist_letters'): print('exists g32_emnist_letters'); return
    ds = datasets.EMNIST(TV, split='letters', train=False, download=True)      # 20800 images; the raw files are transposed w.r.t. MNIST
    x = ds.data.numpy().astype(np.uint8).transpose(0, 2, 1)
    save('g32_emnist_letters', np.pad(x, ((0, 0), (2, 2), (2, 2))), (32, 32))
    np.save(f'{OUT}/g32_emnist_letters_y.npy', ds.targets.numpy())


def gray():
    for src, name in [('cifar10_test', 'g32_cifar10_gray'), ('svhn_test', 'g32_svhn_gray')]:
        if done(name): print('exists', name); continue
        x = np.load(f'{OUT}/{src}.npy')
        save(name, np.stack([gray_pad(Image.fromarray(a)) for a in x]), (32, 32))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['stl10', 'gtsrb', 'flowers', 'eurosat', 'mnist64', 'emnist', 'gray']
    for t in todo:
        try:
            print('==', t, flush=True); globals()[t]()
        except Exception:
            traceback.print_exc(); print('FAILED', t, flush=True)
    print('PREP_NEWOOD_DONE', flush=True)
