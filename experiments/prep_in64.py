# prep_in64.py -- 64x64 arrays for the ImageNet tier, using the EDM2 / ADM preprocessing ("center-crop-dhariwal")
import os, sys, glob, zipfile
import numpy as np
from PIL import Image
from multiprocessing import Pool

HOME = os.path.expanduser('~')
ROOT = os.path.join(os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'); OUT = f'{ROOT}/npy'; OO = f'{ROOT}/openood'
if not os.path.isdir(f'{OO}/benchmark_imglist'):
    with zipfile.ZipFile(f'{OO}/benchmark_imglist.zip') as z: z.extractall(OO)
L = f'{OO}/benchmark_imglist/imagenet'
D = f'{EXT_DATA}/openood_official'


def crop64(im, size=64):
    im = im.convert('RGB')
    while min(*im.size) >= 2 * size:
        im = im.resize(tuple(x // 2 for x in im.size), resample=Image.BOX)
    s = size / min(*im.size)
    im = im.resize(tuple(round(x * s) for x in im.size), resample=Image.BICUBIC)
    a = np.asarray(im)
    y, x = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[y:y + size, x:x + size]


def _one(path):
    try:
        with Image.open(path) as im:
            return crop64(im)
    except Exception as e:
        print('ERR', path, e, flush=True)
        return np.zeros((64, 64, 3), np.uint8)


def _celeba(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        l, t = round((178 - 140) / 2), round((218 - 140) / 2)
        return crop64(im.crop((l, t, l + 140, t + 140)))


def build(name, paths, fn=_one):
    if os.path.exists(f'{OUT}/{name}.npy'):
        print('exists', name, flush=True); return
    missing = [p for p in paths if not os.path.exists(p)]
    print(name, len(paths), 'missing', len(missing), missing[:2], flush=True)
    assert not missing
    with Pool(22) as pool:
        arr = np.stack(pool.map(fn, paths, chunksize=64))
    np.save(f'{OUT}/{name}.npy', arr)
    print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def read_list(f):
    rows = [l.rsplit(' ', 1) for l in open(f'{L}/{f}').read().strip().split('\n')]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows])


for split in ['val', 'test']:
    rel, y = read_list(f'{split}_imagenet.txt')
    paths = [f'{D}/images_largescale/imagenet_1k/val/{os.path.basename(r)}' for r in rel]
    build(f'in64_imagenet_{split}', paths); np.save(f'{OUT}/in64_imagenet_{split}_y.npy', y)
rel, _ = read_list('test_ssb_hard.txt')
build('in64_ssb_hard', [f'{EXT_DATA}/ood_benchmarks/' + r for r in rel])
rel, _ = read_list('test_ninco.txt')
nroot = f'{EXT_DATA}/ood_benchmarks/ninco/NINCO/NINCO_OOD_classes'
nidx = {os.path.relpath(q, nroot).replace(' ', '_'): q for q in glob.glob(f'{nroot}/*/*')}      # OpenOOD replaced spaces by underscores
build('in64_ninco', [nidx.get(r.split('/', 1)[1], f'{nroot}/' + r.split('/', 1)[1]) for r in rel])
rel, _ = read_list('test_inaturalist.txt')
build('in64_inaturalist', [f'{D}/images_largescale/inaturalist/extracted/' + r.split('/', 1)[1] for r in rel])
rel, _ = read_list('test_openimage_o.txt')
build('in64_openimage_o', [f'{D}/images_largescale/openimage_o/extracted/' + r.split('/', 1)[1] for r in rel])
rel, _ = read_list('test_textures.txt')
build('in64_textures', [f'{D}/images_classic/' + r for r in rel])
# extra far-domain sets at 64x64
pl = sorted(glob.glob(f'{EXT_DATA}/places365_val256/val_256/*.jpg'))
rng = np.random.RandomState(0)
build('in64_places365', [pl[i] for i in sorted(rng.choice(len(pl), 10000, replace=False))])
part = [l.split() for l in open(f'{ROOT}/raw/list_eval_partition.txt').read().strip().split('\n')]
build('in64_celeba_test', [f'{ROOT}/raw/celeba/img_align_celeba/{n}' for n, k in part if int(k) == 2], _celeba)
# a slice of ImageNet train (to check the train / held-out gap of the statistics)
wn = sorted(os.listdir(f'{D}/images_largescale/imagenet_1k/train'))
tr = []
for w in wn:
    if not os.path.isdir(f'{D}/images_largescale/imagenet_1k/train/{w}'): continue
    fs = sorted(os.listdir(f'{D}/images_largescale/imagenet_1k/train/{w}'))
    if fs: tr += [f'{D}/images_largescale/imagenet_1k/train/{w}/{f}' for f in fs[:10]]
build('in64_imagenet_train10', tr)
print('IN64_DONE', flush=True)
