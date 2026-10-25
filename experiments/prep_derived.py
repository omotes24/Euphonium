# prep_derived.py -- arrays derived from the downloaded sets (run after prep_data.py, prep_gray.py):
#   <set>_calib          5000 training images held out with the permutation of train.py (RandomState(1234), first 5000, sorted):
#                        cifar10_calib, cifar100_calib, svhn_calib, g32_fmnist_calib
#   g32_<set>            28x28 grey-scale sets zero padded to 32x32 (2 pixels on every side): fmnist / mnist / kmnist train+test,
#                        notmnist, omniglot_eval
#   synth_* / g32_synth_*         uniform noise and constant images for the final split (5000 each), one RandomState(0) stream
#   synthdev_* / g32_synthdev_*   the same for the development split (1000 each), one RandomState(1) stream
#   usage: python experiments/prep_derived.py [--out DIR]   (default: data/npy)
import os, argparse
import numpy as np
ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPY = os.path.join(ROOT, 'data', 'npy')
p = argparse.ArgumentParser(); p.add_argument('--out', default=NPY); a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
L = lambda n: np.load(f'{NPY}/{n}.npy', mmap_mode='r')


def save(name, x):
    np.save(f'{a.out}/{name}.npy', np.ascontiguousarray(x)); print('[saved]', name, x.shape, flush=True)


for g in ('fmnist_train', 'fmnist_test', 'mnist_train', 'mnist_test', 'kmnist_train', 'kmnist_test', 'notmnist', 'omniglot_eval'):
    save(f'g32_{g}', np.pad(L(f'g_{g}'), ((0, 0), (2, 2), (2, 2))))
for c, tr in (('cifar10_calib', 'cifar10_train'), ('cifar100_calib', 'cifar100_train'), ('svhn_calib', 'svhn_train'),
              ('g32_fmnist_calib', 'g32_fmnist_train')):
    t = np.load(f'{a.out}/{tr}.npy', mmap_mode='r') if os.path.exists(f'{a.out}/{tr}.npy') else L(tr)
    save(c, t[np.sort(np.random.RandomState(1234).permutation(len(t))[:5000])])
for seed, n, pre in ((0, 5000, 'synth'), (1, 1000, 'synthdev')):
    rng = np.random.RandomState(seed)                   # draw order: RGB noise, RGB constant, grey noise, grey constant
    save(f'{pre}_noise', rng.randint(0, 256, (n, 32, 32, 3)).astype(np.uint8))
    save(f'{pre}_const', np.broadcast_to(rng.randint(0, 256, (n, 1, 1, 3)), (n, 32, 32, 3)).astype(np.uint8))
    save(f'g32_{pre}_noise', rng.randint(0, 256, (n, 32, 32)).astype(np.uint8))
    save(f'g32_{pre}_const', np.broadcast_to(rng.randint(0, 256, (n, 1, 1)), (n, 32, 32)).astype(np.uint8))
