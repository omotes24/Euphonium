# grid_candidates.py -- Ablation: the grid configurations of protocol/PROTOCOL_20NFE.md (one scale set shared by every rotation,
#   at most 20 forward passes per image), scored on the same selection data as Euphonium (ID development images vs the extra OOD
#   sets).  These candidates are not Euphonium; the paper reports them in the lower block of Table A3.
#     c1-c12  rotations R90 / R180 / both / three, scales every other grid point (odd {1,3,5,7,9} or even {2,...,10}), every fourth
#             or all ten, antithetic pair ("pair", runs B_/P_/I2_ and F_) or +eps only ("single", runs N_<tag>)
#     R2x10pair, R3x10pair, R2x10single  ten scales with two or three rotations (60, 80 and 30 forward passes)
#   Writes results/ablations/grid_selection.json = {candidate: {tfs, ks, noise, nfe, per: {tag: {set: AUROC %}}}}.
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common import TIERS, DEVRUN, RUN, POOL, RESULTS, feat, has_feat, first_pair, dg, auroc

ODD, EVEN, ALL = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9], list(range(10))
CAND = {   # name: (rotations, scales, noise, forward passes per image)
    'c1': (('rot180',), ODD, 'pair', 20), 'c2': (('rot180',), EVEN, 'pair', 20),
    'c3': (('rot90',), ODD, 'pair', 20), 'c4': (('rot90',), EVEN, 'pair', 20),
    'c5': (('rot90', 'rot180'), [0, 4, 8], 'pair', 18), 'c6': (('rot90', 'rot180'), [1, 5, 9], 'pair', 18),
    'c7': (('rot180',), ALL, 'single', 20), 'c8': (('rot90',), ALL, 'single', 20),
    'c9': (('rot90', 'rot180'), ODD, 'single', 15), 'c10': (('rot90', 'rot180'), EVEN, 'single', 15),
    'c11': (('rot90', 'rot180', 'rot270'), ODD, 'single', 20), 'c12': (('rot90', 'rot180', 'rot270'), EVEN, 'single', 20),
}
REF = {'R2x10pair': (('rot90', 'rot180'), ALL, 'pair', 60), 'R3x10pair': (('rot90', 'rot180', 'rot270'), ALL, 'pair', 80),
       'R2x10single': (('rot90', 'rot180'), ALL, 'single', 30)}


def load(run, name):
    return first_pair(feat(run, name)) if has_feat(run, name) else None


if __name__ == '__main__':
    out = {}
    for cid, (tfs, ks, noise, nfe) in list(CAND.items()) + list(REF.items()):
        per = {}
        for t in TIERS:
            if noise == 'pair':
                Fid = load(DEVRUN[t], TIERS[t]['id']); pool = {s: load(RUN['pair'][t], s) for s in POOL[t]}
            else:                                   # single-noise runs: ID development images are stored as <id>__dev
                Fid = load(RUN['single'][t], TIERS[t]['id'] + '__dev'); pool = {s: load(RUN['single'][t], s) for s in POOL[t]}
            if Fid is None or any(F is None for F in pool.values()) or any(x not in Fid['tfs'] for x in tfs):
                per[t] = None; continue
            g0 = dg(Fid, tfs=tfs, ks=ks)
            per[t] = {s: 100 * auroc(g0, dg(F, tfs=tfs, ks=ks)) for s, F in pool.items()}
        out[cid] = dict(tfs=tfs, ks=ks, noise=noise, nfe=nfe, per=per)
    os.makedirs(os.path.join(RESULTS, 'ablations'), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, 'ablations', 'grid_selection.json'), 'w'), indent=1)
    print('%-11s NFE  ' % 'candidate', ' '.join('%8s' % t for t in TIERS), '   mean')
    for cid, r in out.items():
        m = {t: (np.mean(list(v.values())) if v else None) for t, v in r['per'].items()}
        ok = all(v is not None for v in m.values())
        print('%-11s %3d  ' % (cid, r['nfe']), ' '.join('%8s' % ('--' if m[t] is None else '%.1f' % m[t]) for t in TIERS),
              '   %s' % ('%.2f' % np.mean(list(m.values())) if ok else '(incomplete)'))
    print('saved results/ablations/grid_selection.json')
