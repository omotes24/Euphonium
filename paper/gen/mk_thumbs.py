# mk_thumbs.py -- thumbnails of Figure 2 (figs/sh_*.pdf): x (CIFAR-10 test 3535, the median dog of Figure 1), eps, x_sigma = x + sigma eps
#   and T x_sigma, with T the rotation of the coarsest atom of rule.json (R90) and sigma = 0.3 for visibility.
import os, sys, json
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')                  # reproducible PDF metadata
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE); ROOT = os.path.dirname(PAPER)
sys.path.insert(0, HERE)
from images import test_images, rot90
RULE = json.load(open(os.path.join(ROOT, 'rule.json')))
T = max(RULE['atoms'], key=lambda a: a[1])[0]                     # rotation of the coarsest atom (the example of Figure 2)
x = test_images('cifar10_test')[3535].astype(np.float64) / 127.5 - 1
eps = np.random.RandomState(0).randn(*x.shape); xs = x + 0.3 * eps
def tf(a):
    if T == 'rot90': return rot90(a)
    if T == 'rot180': return rot90(rot90(a))
    if T == 'rot270': return rot90(rot90(rot90(a)))
    raise ValueError(T)
def save(name, a01):
    fig = plt.figure(figsize=(1, 1), dpi=128); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(np.clip(a01, 0, 1), interpolation='nearest'); ax.axis('off')
    fig.savefig(os.path.join(PAPER, 'figs', f'sh_{name}.pdf')); plt.close(fig)
save('x', (x + 1) / 2); save('eps', 0.5 + 0.18 * eps); save('xs', (xs + 1) / 2); save('txs', (tf(xs) + 1) / 2)
print('thumbnails for T =', T)
