# select_rule.py -- how the Euphonium rule (rule.json) was selected (protocol/PROTOCOL_10NFE.md), recomputed from the inputs
#   Candidates: every set of 1-3 atoms (T, k), T in {R90, R180, R270}, k in sigma_1..sigma_10, antithetic pair, whose cost
#   2 x (number of distinct scales + number of atoms) is at most 10 forward passes per image: 1285 candidates.
#   Selection data only (never part of the main evaluation): ID development images (runs B_/P_/I2_) vs the eleven extra OOD sets of
#   every ID (common.POOL; final-split images in the F_ runs).  Criterion: mean over the six IDs of the per-ID mean AUROC.
#   Tie rule: candidates within 0.1 points of the best are tied; among them the fewest forward passes, then the largest worst-ID mean.
#   Writes results/selection_10nfe.json and checks that the selected atoms are those of rule.json.
import os, sys, json, itertools
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import TIERS, DEVRUN, RUN, POOL, RESULTS, ID_NAME, feat, first_pair
from euphonium import load_rule, nfe, describe

TFS, NK = ('rot90', 'rot180', 'rot270'), 10
ATOMS = [(T, k) for T in TFS for k in range(NK)]


def logratios(F):
    """[N, 3, 10]: log e_k(Tx)/e_k(x) for the three rotations and the first ten scales (first antithetic pair)"""
    e = first_pair(F)['e'][:, :NK, 0, :]; i0 = F['tfs'].index('id')
    return np.stack([np.log(e[:, :, F['tfs'].index(T)] / e[:, :, i0]) for T in TFS], 1)


def auroc_fast(a, b):
    n0, n1 = len(a), len(b); r = rankdata(np.concatenate([a, b]))
    return (r[:n0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)


def candidates(budget=10):
    return [list(c) for r in range(1, 4) for c in itertools.combinations(ATOMS, r) if nfe(list(c), 'pair') <= budget]


def selection_scores(cands):
    """{candidate: {tag: mean AUROC over the tag's selection OOD sets}}"""
    D = {}
    for t in TIERS:
        D[t] = (logratios(feat(DEVRUN[t], TIERS[t]['id'])), [logratios(feat(RUN['pair'][t], s)) for s in POOL[t]])
    out = {}
    for c in cands:
        ti = [TFS.index(T) for T, _ in c]; ki = [k for _, k in c]
        per = {}
        for t, (L0, Ls) in D.items():
            g0 = L0[:, ti, ki].sum(1)
            per[t] = float(100 * np.mean([auroc_fast(g0, L[:, ti, ki].sum(1)) for L in Ls]))
        out[tuple(c)] = per
    return out


def pick(scores, tie=0.1):
    """the selection rule of PROTOCOL_10NFE.md"""
    mean = {c: float(np.mean(list(v.values()))) for c, v in scores.items()}
    top = max(mean.values()); tied = [c for c in scores if mean[c] >= top - tie]
    best = min(tied, key=lambda c: (nfe(list(c), 'pair'), -min(scores[c].values())))
    return best, tied, mean


if __name__ == '__main__':
    rule = load_rule()
    cands = candidates(rule['nfe'])
    print(len(cands), 'candidates with at most', rule['nfe'], 'forward passes per image', flush=True)
    S = selection_scores(cands)
    best, tied, mean = pick(S)
    ranked = sorted(S, key=lambda c: -mean[c])
    print('rank  candidate                              NFE   mean   worst ID')
    for i, c in enumerate(ranked[:10]):
        print('%4d  %-38s %3d  %6.2f  %6.1f%s' % (i + 1, describe(list(c)), nfe(list(c), 'pair'), mean[c], min(S[c].values()), '   <- selected' if c == best else ('   (tied)' if c in tied else '')))
    json.dump(dict(n_candidates=len(cands), criterion='mean over the six IDs of the mean AUROC (%) over the selection OOD sets',
                   tie_rule='within 0.1 points of the best: fewest forward passes, then the largest worst-ID mean',
                   selected=[list(a) for a in best], tied=[[list(a) for a in c] for c in tied],
                   candidates=[dict(atoms=[list(a) for a in c], nfe=nfe(list(c), 'pair'), mean=round(mean[c], 4),
                                    per_id={t: round(v, 4) for t, v in S[c].items()}) for c in ranked]),
              open(os.path.join(RESULTS, 'selection_10nfe.json'), 'w'), indent=0)
    assert sorted(best) == sorted(rule['atoms']), (best, rule['atoms'])
    print('selected', describe(list(best)), '== rule.json; saved results/selection_10nfe.json')
