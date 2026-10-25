# rule.py -- the Euphonium rule (rule.json at the repository root) and the noise-scale grid it refers to
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_PATH = os.path.join(ROOT, 'rule.json')

# log-spaced grid sigma_k = 0.01 * 3162.3^(k/15), k = 0..15 (sigma_1 = 0.01 ... sigma_16 = 31.62 in the paper's 1-based notation).
# The experiments use the first 13 points; the noise image of a scale is keyed by its index in this grid (noise.shared_noise).
SIGMA_GRID = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
ROTATIONS = ('rot90', 'rot180', 'rot270')


def load_rule(path=None):
    """{'atoms': [(rotation, k), ...], 'noise': 'pair' | 'single', 'seed': int, 'nfe': int, ...} with k an index into SIGMA_GRID"""
    with open(path or RULE_PATH) as f:
        r = json.load(f)
    r['atoms'] = [(str(T), int(k)) for T, k in r['atoms']]
    assert all(T in ROTATIONS for T, _ in r['atoms']), r['atoms']
    assert r['nfe'] == nfe(r['atoms'], r['noise']), 'rule.json: nfe does not match the atoms'
    return r


def scales(atoms):
    """distinct scale indices used by a set of atoms (the upright image is evaluated once per scale)"""
    return sorted(set(k for _, k in atoms))


def nfe(atoms, noise='pair'):
    """forward passes per image: one per draw for the upright image at every used scale, and one per draw for every atom"""
    draws = 2 if noise == 'pair' else 1
    return draws * (len(scales(atoms)) + len(atoms))


def describe(atoms):
    """'R90: sigma_1, sigma_7; R180: sigma_1' (1-based scale numbers, as in the paper)"""
    out = []
    for T in ROTATIONS:
        ks = sorted(k for TT, k in atoms if TT == T)
        if ks: out.append('R%s: %s' % (T[3:], ', '.join('sigma_%d' % (k + 1) for k in ks)))
    return '; '.join(out)
