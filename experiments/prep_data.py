# prep_data.py -- build uint8 arrays [N,32,32,3] under <repo>/data/npy
import os, sys, io, pickle, tarfile, zipfile, glob, subprocess, traceback
import numpy as np
from PIL import Image
from multiprocessing import Pool

ROOT = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
RAW, OUT = f'{ROOT}/raw', f'{ROOT}/npy'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
HOME = os.path.expanduser('~'); EXT_DATA = os.environ.get('SR_DATASETS', os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'external'))


def done(name):
    return os.path.exists(f'{OUT}/{name}.npy')


def save(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    assert arr.ndim == 4 and arr.shape[1:] == (32, 32, 3), arr.shape
    np.save(f'{OUT}/{name}.npy', arr)
    print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def wget(url, dst):
    if not os.path.exists(dst):
        subprocess.run(['curl', '-sL', '--fail', '--retry', '5', '-o', dst + '.part', url], check=True)
        os.rename(dst + '.part', dst)
    return dst


def gdrive(fid, dst):
    if not os.path.exists(dst):
        url = f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'
        subprocess.run(['curl', '-sL', '-o', dst + '.part', url], check=True)
        os.rename(dst + '.part', dst)
    return dst


def resize_cc(img, size=32):
    """shorter side -> size (PIL bilinear, antialiased) then center crop (OpenOOD / torchvision style)."""
    img = img.convert('RGB')
    w, h = img.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)))


def _load_cc(path):
    try:
        with Image.open(path) as im:
            return resize_cc(im)
    except Exception as e:
        print('ERR', path, e, flush=True)
        return np.zeros((32, 32, 3), np.uint8)


def _load_celeba(path):
    with Image.open(path) as im:
        im = im.convert('RGB')                      # 178 x 218
        l, t = round((178 - 140) / 2), round((218 - 140) / 2)
        im = im.crop((l, t, l + 140, t + 140)).resize((32, 32), Image.BILINEAR)
        return np.asarray(im)


def pmap(fn, paths, procs=20):
    with Pool(procs) as p:
        return np.stack(p.map(fn, paths, chunksize=256))


def cifar10():
    if done('cifar10_train') and done('cifar10_test'): return
    f = wget('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', f'{RAW}/cifar-10-python.tar.gz')
    with tarfile.open(f) as t: t.extractall(RAW)
    def rd(n):
        d = pickle.load(open(f'{RAW}/cifar-10-batches-py/{n}', 'rb'), encoding='bytes')
        return d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b'labels'])
    xs, ys = zip(*[rd(f'data_batch_{i}') for i in range(1, 6)])
    save('cifar10_train', np.concatenate(xs)); np.save(f'{OUT}/cifar10_train_y.npy', np.concatenate(ys))
    x, y = rd('test_batch'); save('cifar10_test', x); np.save(f'{OUT}/cifar10_test_y.npy', y)


def cifar100():
    if done('cifar100_train') and done('cifar100_test'): return
    src = f'{EXT_DATA}/cifar100/cifar-100-python'
    for s in ['train', 'test']:
        d = pickle.load(open(f'{src}/{s}', 'rb'), encoding='bytes')
        save(f'cifar100_{s}', d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        np.save(f'{OUT}/cifar100_{s}_y.npy', np.array(d[b'fine_labels']))


def svhn():
    import scipy.io as sio
    for s in ['train', 'test']:
        if done(f'svhn_{s}'): continue
        f = wget(f'http://ufldl.stanford.edu/housenumbers/{s}_32x32.mat', f'{RAW}/svhn_{s}_32x32.mat')
        m = sio.loadmat(f)
        save(f'svhn_{s}', m['X'].transpose(3, 0, 1, 2)); np.save(f'{OUT}/svhn_{s}_y.npy', m['y'].ravel() % 10)


def celeba():
    if all(done(f'celeba_{s}') for s in ['train', 'valid', 'test']): return
    B = 'https://huggingface.co/datasets/Yuehao/celeba/resolve/main'
    z = wget(f'{B}/img_align_celeba.zip', f'{RAW}/img_align_celeba.zip')
    p = wget(f'{B}/list_eval_partition.txt', f'{RAW}/list_eval_partition.txt')
    d = f'{RAW}/celeba'
    if not os.path.exists(f'{d}/img_align_celeba/202599.jpg'):
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
    part = [l.split() for l in open(p).read().strip().split('\n')]
    for i, s in enumerate(['train', 'valid', 'test']):
        paths = [f'{d}/img_align_celeba/{n}' for n, k in part if int(k) == i]
        save(f'celeba_{s}', pmap(_load_celeba, paths))


def textures():
    if done('textures'): return
    root = f'{EXT_DATA}/extra_ood/dtd'
    paths = sorted(glob.glob(f'{root}/images/*/*.jpg'))
    print('dtd images', len(paths), paths[:2], flush=True)
    save('textures', pmap(_load_cc, paths))
    with open(f'{OUT}/textures_paths.txt', 'w') as f: f.write('\n'.join(paths))


def tin():
    if done('tin_val') and done('tin_train10k'): return
    z = wget('https://image-net.org/data/tiny-imagenet-200.zip', f'{RAW}/tiny-imagenet-200.zip')
    if not os.path.exists(f'{RAW}/tiny-imagenet-200/val/images'):
        with zipfile.ZipFile(z) as zf: zf.extractall(RAW)
    paths = sorted(glob.glob(f'{RAW}/tiny-imagenet-200/val/images/*.JPEG'))
    save('tin_val', pmap(_load_cc, paths))
    tr = sorted(glob.glob(f'{RAW}/tiny-imagenet-200/train/*/images/*.JPEG'))
    rng = np.random.RandomState(0); sel = [tr[i] for i in sorted(rng.choice(len(tr), 10000, replace=False))]
    save('tin_train10k', pmap(_load_cc, sel))


def imagenet32():
    if not done('imagenet32_val'):
        z = wget('https://image-net.org/data/downsample/Imagenet32_val.zip', f'{RAW}/Imagenet32_val.zip')
        with zipfile.ZipFile(z) as zf: zf.extractall(f'{RAW}/imagenet32')
        fs = glob.glob(f'{RAW}/imagenet32/**/val_data', recursive=True)
        d = pickle.load(open(fs[0], 'rb'))
        save('imagenet32_val', np.asarray(d['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        np.save(f'{OUT}/imagenet32_val_y.npy', np.array(d['labels']) - 1)
    if not done('imagenet32_train') and os.environ.get('IN32_TRAIN', '0') == '1':
        z = wget('https://image-net.org/data/downsample/Imagenet32_train.zip', f'{RAW}/Imagenet32_train.zip')
        with zipfile.ZipFile(z) as zf: zf.extractall(f'{RAW}/imagenet32')
        xs, ys = [], []
        for i in range(1, 11):
            fs = glob.glob(f'{RAW}/imagenet32/**/train_data_batch_{i}', recursive=True)
            d = pickle.load(open(fs[0], 'rb'))
            xs.append(np.asarray(d['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)); ys.append(np.array(d['labels']) - 1)
        save('imagenet32_train', np.concatenate(xs)); np.save(f'{OUT}/imagenet32_train_y.npy', np.concatenate(ys))


def csi_fix():
    for name, fid in [('lsun_fix', '1KVWj9xpHfVwGcErH5huVujk9snhEGOxE'),
                      ('imagenet_fix', '1sO_-noq10mmziB1ECDyNhD5T4u5otyKA')]:
        if done(name): continue
        f = gdrive(fid, f'{RAW}/{name}.tar.gz')
        with tarfile.open(f) as t: t.extractall(f'{RAW}/{name}')
        paths = sorted(glob.glob(f'{RAW}/{name}/**/*.png', recursive=True))
        print(name, len(paths), flush=True)
        save(name, pmap(_load_cc, paths))


def places():
    if done('places365_val'): return
    paths = sorted(glob.glob(f'{EXT_DATA}/places365_val256/val_256/*.jpg'))
    print('places', len(paths), flush=True)
    save('places365_val', pmap(_load_cc, paths))


def gray_sets():
    import torchvision
    tv = f'{RAW}/tv'
    for name, cls in [('mnist', torchvision.datasets.MNIST), ('fmnist', torchvision.datasets.FashionMNIST),
                      ('kmnist', torchvision.datasets.KMNIST)]:
        for train in [True, False]:
            s = 'train' if train else 'test'
            if os.path.exists(f'{OUT}/g_{name}_{s}.npy'): continue
            ds = cls(tv, train=train, download=True)
            x = ds.data.numpy().astype(np.uint8)                       # [N,28,28]
            np.save(f'{OUT}/g_{name}_{s}.npy', x); np.save(f'{OUT}/g_{name}_{s}_y.npy', ds.targets.numpy())
            print('[saved]', f'g_{name}_{s}', x.shape, flush=True)
    if not os.path.exists(f'{OUT}/g_omniglot_eval.npy'):
        ds = torchvision.datasets.Omniglot(tv, background=False, download=True)
        xs = [255 - np.asarray(ds[i][0].convert('L').resize((28, 28), Image.BILINEAR)) for i in range(len(ds))]
        np.save(f'{OUT}/g_omniglot_eval.npy', np.stack(xs).astype(np.uint8)); print('[saved] g_omniglot_eval', len(xs), flush=True)
    if not done('mnist_rgb32'):
        x = np.load(f'{OUT}/g_mnist_test.npy')
        xs = [np.asarray(Image.fromarray(a).convert('RGB').resize((32, 32), Image.BILINEAR)) for a in x]
        save('mnist_rgb32', np.stack(xs))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['cifar10', 'cifar100', 'svhn', 'textures', 'tin', 'imagenet32', 'csi_fix', 'places', 'celeba', 'gray_sets']
    for t in todo:
        try:
            print('==', t, flush=True); globals()[t]()
        except Exception:
            traceback.print_exc(); print('FAILED', t, flush=True)
    print('PREP_DONE', flush=True)
