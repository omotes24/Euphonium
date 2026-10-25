# common.py -- the benchmark (six ID datasets, their OOD sets, the selection data) and loaders of the per-image statistics
#   The per-image statistics of the GPU runs live in inputs/ (release asset inputs-v1, or $EUPHONIUM_INPUTS):
#     inputs/feat/<run>/<set>.npz        e[B,K,P,T] = squared noise-prediction error per image, scale, draw pair and transform
#                                        (shared noise; experiments/extract2.py + export.py), idx, tfs, sigmas, dim
#     inputs/nll/<run>/<set>.npz         PF-ODE log-likelihood (ll, bpd) per image; runs *_rot90 hold the R90 copies (nll.py)
#     inputs/base/<run>/<set>.npz        clean-input score norms (MSMA-type baseline; extract.py)
#     inputs/base2/<method>_<tag>/...    raw statistics of DiffPath, GEPC, EigenScore (baselines.py)
#     inputs/complexity/<set>.npz        PNG / JPEG 2000 code lengths (IC baselines; complexity.py)
#   Run names: F_<tag> final split, B_/P_/I2_ development split (ID development images), N_<tag> single-noise runs (+eps only),
#   F_edmc10_s1/_s2 other noise seeds, F_edmc10_noanti / _perimg noise controls, D3d_* training snapshots of our own models.
import os, json
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.environ.get('EUPHONIUM_INPUTS', os.path.join(ROOT, 'inputs'))
RESULTS = os.path.join(ROOT, 'results')
BENCH = os.path.join(ROOT, 'benchmark')

# ----------------------------------------------------------------------------- the benchmark
RGB = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TIERS = {   # ID set and OOD sets of every ID model (63 pairs): natural sets, the upside-down ID set (__vflip) and two synthetic sets
    'edmc10':   dict(id='cifar10_test', ood=[s for s in RGB if s != 'cifar10_test'] + ['cifar10_test__vflip', 'synth_noise', 'synth_const']),
    'l6svhn':   dict(id='svhn_test', ood=[s for s in RGB if s != 'svhn_test'] + ['svhn_test__vflip', 'synth_noise', 'synth_const']),
    'dpceleba': dict(id='celeba_test', ood=[s for s in RGB if s != 'celeba_test'] + ['celeba_test__vflip', 'synth_noise', 'synth_const']),
    'edmc100':  dict(id='cifar100_test', ood=[s for s in RGB if s != 'cifar100_test'] + ['cifar100_test__vflip', 'synth_noise', 'synth_const']),
    'in64':     dict(id='in64_imagenet_test', ood=['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test', 'in64_imagenet_test__vflip']),
    'fmnist':   dict(id='g32_fmnist_test', ood=['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist', 'g32_fmnist_test__vflip', 'g32_synth_noise', 'g32_synth_const']),
}
IDS = list(TIERS)
ID_NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
MODEL = {'edmc10': 'edm:edm-cifar10-32x32-uncond-vp.pkl', 'l6svhn': 'l6:layer6_svhn_vpsde.pt', 'dpceleba': 'iddpm:diffpath_celeba32.pt',
         'edmc100': 'edmsd:edm_cifar100_uncond_vp.pt', 'in64': 'edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', 'fmnist': 'own:models/fmnist_s/final.pt'}
NAT = {t: [s for s in T['ood'] if not s.endswith('__vflip') and 'synth' not in s] for t, T in TIERS.items()}   # natural OOD sets
SET_NAME = {'cifar10_test': 'CIFAR-10', 'cifar100_test': 'CIFAR-100', 'svhn_test': 'SVHN', 'celeba_test': 'CelebA', 'textures': 'Textures',
            'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet', 'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365',
            'synth_noise': 'uniform noise', 'synth_const': 'constant', 'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO',
            'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O', 'in64_places365': 'Places365',
            'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
            'g32_notmnist': 'notMNIST', 'g32_synth_noise': 'uniform noise', 'g32_synth_const': 'constant'}


def set_name(s): return SET_NAME.get(s, 'upside-down ID' if s.endswith('__vflip') else s)


# runs of the per-image statistics: final split (pair = antithetic pair, single = +eps only) and ID development images
RUN = {'pair': {'edmc10': 'F_edmc10', 'l6svhn': 'F_l6svhn', 'dpceleba': 'F_dpceleba', 'edmc100': 'F_edmc100', 'in64': 'F2_in64', 'fmnist': 'F_fmnist'},
       'single': {t: f'N_{t}' for t in TIERS}}
DEVRUN = {'edmc10': 'B_edmc10', 'l6svhn': 'P_l6svhn', 'dpceleba': 'P_dpceleba', 'edmc100': 'P_edmc100', 'in64': 'I2_edm2xs', 'fmnist': 'P_fmnist'}
NLLRUN = {'edmc10': 'F_edmc10', 'l6svhn': 'F_l6svhn', 'edmc100': 'F_edmc100', 'in64': 'F_in64'}   # IDs with the exact ratio (PF-ODE)
# selection data (never part of the main evaluation): ID development images vs eleven extra OOD sets per ID
RGBPOOL = ['stl10_test', 'gtsrb_test', 'flowers102_test', 'eurosat', 'mnist_rgb32', 'pets_test', 'caltech101', 'food101_test', 'aircraft_test', 'imagenette_val', 'sst2_test']
POOL = {'edmc10': RGBPOOL, 'l6svhn': RGBPOOL, 'dpceleba': RGBPOOL, 'edmc100': RGBPOOL,
        'in64': ['in64_' + s for s in RGBPOOL if s not in ('imagenette_val', 'mnist_rgb32')] + ['in64_mnist_rgb'],
        'fmnist': ['g32_emnist_letters', 'g32_k49', 'g32_semeion', 'g32_cifar10_gray', 'g32_svhn_gray', 'g32_gtsrb_gray']}
BASELINES = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']     # the eight comparison methods


def common_idx():
    """{tag: {'__id__' | set: [image indices]}}: the images every compared method has (benchmark/common_idx.json); the tables
    compare all methods on exactly these images"""
    with open(os.path.join(BENCH, 'common_idx.json')) as f: return json.load(f)


# ----------------------------------------------------------------------------- loaders
def feat(run, name):
    z = np.load(f'{INPUTS}/feat/{run}/{name}.npz', allow_pickle=True)
    return dict(e=z['e'].astype(np.float64), tfs=[str(t) for t in z['tfs']], sigmas=z['sigmas'].astype(np.float64),
                idx=z['idx'], dim=float(z['dim']))


def has_feat(run, name): return os.path.exists(f'{INPUTS}/feat/{run}/{name}.npz')


def nll(run, name):
    p = f'{INPUTS}/nll/{run}/{name}.npz'
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    return dict(ll=z['ll'].astype(np.float64), idx=z['idx'], bpd=z['bpd'].astype(np.float64))


def base(run, name):
    p = f'{INPUTS}/base/{run}/{name}.npz'
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files if k != 'meta'}


def complexity(name, idx):
    p = f'{INPUTS}/complexity/{name.split("__")[0]}.npz'
    if not os.path.exists(p): return None
    z = np.load(p)
    return dict(png=z['png'][idx].astype(np.float64), j2k=z['j2k'][idx].astype(np.float64))


def first_pair(F):
    """keep the first antithetic pair (runs with two pairs were evaluated with the first one, like the single-pair runs)"""
    if F['e'].shape[2] > 1: F['e'] = F['e'][:, :, :1, :]
    return F


def sub(score, idx, keep):
    """the entries of score (indexed like idx) for the image indices in keep, in the order of keep"""
    pos = {int(i): j for j, i in enumerate(idx.tolist())}
    return score[[pos[i] for i in keep if i in pos]]


# ----------------------------------------------------------------------------- metrics (ID = positive class, higher score = more ID)
def auroc(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def fpr95(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    thr = np.percentile(a[np.isfinite(a)], 5)
    return float(np.mean(b[np.isfinite(b)] >= thr))


# ----------------------------------------------------------------------------- sums of log-ratios over atoms
def atom_score(F, atoms, mode='logr'):
    """sum over atoms (T, k) of log e_k(Tx)/e_k(x) (logr, Euphonium) or of e_k(Tx) - e_k(x) (abs, the difference form)  [N]"""
    e = F['e'][:, :, 0, :]; i0 = F['tfs'].index('id'); s = np.zeros(e.shape[0])
    for T, k in atoms:
        a, b = e[:, k, F['tfs'].index(T)], e[:, k, i0]
        s += np.log(a / b) if mode == 'logr' else (a - b)
    return s


def dg(F, tfs=('rot90', 'rot180', 'rot270'), ks=range(10), mode='logr', pair=0, ebar=None, ref='id'):
    """sum over transforms T and scales k of the contrast between e_k(Tx) and e_k(x) (grid configurations of the ablations)"""
    e = F['e'][:, :, pair, :]
    i0 = F['tfs'].index(ref)
    s = np.zeros(e.shape[0])
    for t in tfs:
        j = F['tfs'].index(t)
        for k in ks:
            a, b = e[:, k, j], e[:, k, i0]
            if mode == 'logr': s += np.log(a / b)
            elif mode == 'abs': s += a - b
            elif mode == 'max': s += (a - b) / np.maximum(b, ebar[k])
            else: raise ValueError(mode)
    return s


def dg_cross(F1, F2, tfs=('rot90', 'rot180', 'rot270'), ks=range(10), pair=0):
    """independent-noise control: e_k(Tx) from run F1, e_k(x) from run F2 (another noise draw)"""
    e1, e2 = F1['e'][:, :, pair, :], F2['e'][:, :, pair, :]
    i0 = F2['tfs'].index('id')
    s = np.zeros(e1.shape[0])
    for t in tfs:
        j = F1['tfs'].index(t)
        for k in ks: s += np.log(e1[:, k, j] / e2[:, k, i0])
    return s
