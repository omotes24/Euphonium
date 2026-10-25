# robustness.py -- robustness checks of the rule selection (protocol/PROTOCOL_10NFE.md, "checks"), on the images that the
#   antithetic-pair runs and the single-noise runs have in common (as in the exploratory search protocol/history/explore_10nfe.py):
#   (1) the pure-pair candidates (<= 10 forward passes) ranked on the selection data, the top twelve also on the main evaluation;
#   (2) split-half selection: choose on the odd- (even-) numbered selection OOD sets, check on the other half, against the best grid
#       candidate c3 (20 forward passes);
#   (3) noise seeds: CIFAR-10, the 12 pairs on all final-split images, three noise seeds (runs F_edmc10, F_edmc10_s1, F_edmc10_s2);
#   (4) per-pair AUROCs of c3, e10b and Euphonium (e10).
#   Writes results/ablations/robustness.json.
import os, sys, json, itertools
import numpy as np
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common import TIERS, RUN, DEVRUN, POOL, RESULTS, BASELINES, ID_NAME, feat, has_feat, common_idx

IDS = list(TIERS); TFS = ['rot90', 'rot180', 'rot270']; NK = 10
COM = common_idx(); BL = json.load(open(os.path.join(RESULTS, 'baselines_pairs.json')))
CACHE = {}


def FT(run, s):
    if (run, s) not in CACHE:
        if not has_feat(run, s): CACHE[(run, s)] = None
        else:
            F = feat(run, s)
            if F['e'].shape[2] > 1: F['e'] = F['e'][:, :, :1, :]
            CACHE[(run, s)] = F
    return CACHE[(run, s)]


def atoms(F):
    """per-atom log-ratios [N, 3 (rotation), 10 (scale)] and idx"""
    e = F['e'][:, :NK, 0, :]; i0 = F['tfs'].index('id')
    return F['idx'].astype(int), np.stack([np.log(e[:, :, F['tfs'].index(t)] / e[:, :, i0]) for t in TFS], 1)


def aligned(Fp, Fs, keep=None):
    """pair (Fp) and single (Fs) runs on their common images: [N, 2 (pair, single), 3, 10]"""
    ip, Lp = atoms(Fp); is_, Ls = atoms(Fs)
    common = set(ip.tolist()) & set(is_.tolist())
    if keep is not None: common &= set(int(i) for i in keep)
    order = sorted(common); pp = {int(i): j for j, i in enumerate(ip.tolist())}; ps = {int(i): j for j, i in enumerate(is_.tolist())}
    return np.stack([Lp[[pp[i] for i in order]], Ls[[ps[i] for i in order]]], 1)


def auroc_fast(a, b):
    n0, n1 = len(a), len(b); r = rankdata(np.concatenate([a, b]))
    return (r[:n0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)


def nfe_of(cand):
    up, rot = {}, 0
    for n, T, k in cand:
        up[k] = max(up.get(k, 0), 2 if n == 0 else 1); rot += 2 if n == 0 else 1
    return sum(up.values()) + rot


def score(D, cand):
    s = np.zeros(D.shape[0])
    for n, T, k in cand: s += D[:, n, T, k]
    return s


def fmt(cand):
    parts = []
    for n in (0, 1):
        for T in range(3):
            ks = sorted(k + 1 for nn, TT, k in cand if nn == n and TT == T)
            if ks: parts.append('%s %s{%s}' % ('pair' if n == 0 else 'single', TFS[T].replace('rot', 'R'), ','.join(map(str, ks))))
    return ' + '.join(parts)


SELD, MAIND = {}, {}
for t in IDS:
    SELD[t] = dict(id=aligned(FT(DEVRUN[t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id'] + '__dev')),
                   pool={s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s)) for s in POOL[t]})
    MAIND[t] = dict(id=aligned(FT(RUN['pair'][t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id']), keep=COM[t]['__id__']),
                    ood={s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s), keep=COM[t][s]) for s in TIERS[t]['ood']})
    print(t, 'selection ID images', SELD[t]['id'].shape[0], 'main ID images', MAIND[t]['id'].shape[0], flush=True)


def sel_mean_sets(cand, sets_of):
    v = {}
    for t in IDS:
        g0 = score(SELD[t]['id'], cand)
        v[t] = 100 * np.mean([auroc_fast(g0, score(SELD[t]['pool'][s], cand)) for s in sets_of(t)])
    return float(np.mean(list(v.values()))), v


def main_eval(cand):
    out = dict(main={}, wins=0, npairs=0)
    for t in IDS:
        g0 = score(MAIND[t]['id'], cand); nat = []
        for s, D in MAIND[t]['ood'].items():
            a = 100 * auroc_fast(g0, score(D, cand))
            if not s.endswith('__vflip') and 'synth' not in s: nat.append(a)
            best = max(100 * BL[t]['common'][b][s]['auroc'] for b in BASELINES if s in BL[t]['common'][b])
            out['npairs'] += 1; out['wins'] += int(a >= best - 0.05)
        out['main'][t] = float(np.mean(nat))
    out['main_mean'] = float(np.mean(list(out['main'].values())))
    return out


C3 = [(0, 0, k) for k in (0, 2, 4, 6, 8)]                  # best grid candidate (20 forward passes)
E10 = [(0, 0, 0), (0, 0, 6), (0, 1, 0)]                    # Euphonium: pair R90{sigma_1, sigma_7} + pair R180{sigma_1}
E10B = [(0, 0, 0), (0, 0, 5), (0, 1, 0)]                   # tied runner-up: pair R90{sigma_1, sigma_6} + pair R180{sigma_1}
REPORT = {}
# ---- (1) pure-pair candidates (same run and noise), per-rotation scale sets, <= 10 forward passes
pure = {}
for r in range(1, 6):
    for cand in itertools.combinations([(0, T, k) for T in range(3) for k in range(NK)], r):
        cand = list(cand)
        if nfe_of(cand) > 10: continue
        pure[tuple(cand)] = sel_mean_sets(cand, lambda t: POOL[t])[0]
ranked = sorted(pure.items(), key=lambda kv: -kv[1])
print('pure-pair candidates:', len(pure)); print('c3 selection:', round(sel_mean_sets(C3, lambda t: POOL[t])[0], 2))
REPORT['n_pure'] = len(pure); REPORT['c3_selection'] = sel_mean_sets(C3, lambda t: POOL[t])[0]
REPORT['pure_top'] = []
for key, v in ranked[:12]:
    m = main_eval(list(key)); REPORT['pure_top'].append(dict(cand=fmt(list(key)), nfe=nfe_of(list(key)), sel=v, main_mean=m['main_mean'], wins=m['wins'], main=m['main']))
    print('%-40s NFE %2d  sel %.2f  main %.1f  wins %d  ' % (fmt(list(key)), nfe_of(list(key)), v, m['main_mean'], m['wins']) + ' '.join('%.1f' % m['main'][t] for t in IDS))
# ---- (2) split-half selection: choose on one half of the selection OOD sets, check on the other half
def half(t, parity): return [s for i, s in enumerate(POOL[t]) if i % 2 == parity]
REPORT['split'] = []
for parity in (0, 1):
    rk = sorted(pure.keys(), key=lambda key: -sel_mean_sets(list(key), lambda t: half(t, parity))[0])
    best = list(rk[0]); other = sel_mean_sets(best, lambda t: half(t, 1 - parity))[0]; c3o = sel_mean_sets(C3, lambda t: half(t, 1 - parity))[0]
    m = main_eval(best)
    REPORT['split'].append(dict(choose_on='odd' if parity == 0 else 'even', chosen=fmt(best), heldout_sel=other, heldout_sel_c3=c3o, main_mean=m['main_mean'], wins=m['wins']))
    print('split %d: chosen %-34s held-out selection %.1f (c3 %.1f)  main %.1f wins %d' % (parity, fmt(best), other, c3o, m['main_mean'], m['wins']))
# ---- (3) noise seeds: CIFAR-10, 12 pairs, all final-split images
T0 = TIERS['edmc10']; REPORT['seeds'] = {}
for name, cand in (('c3', C3), ('e10b', E10B), ('e10', E10)):
    vals = []
    for run in ('F_edmc10', 'F_edmc10_s1', 'F_edmc10_s2'):
        Fid = FT(run, T0['id']); g0 = score(aligned(Fid, Fid), cand)
        a = [100 * auroc_fast(g0, score(aligned(FT(run, s), FT(run, s)), cand)) for s in T0['ood'] if FT(run, s) is not None]
        vals.append(float(np.mean(a)))
    REPORT['seeds'][name] = dict(values=vals, mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)))
    print('%-5s CIFAR-10 12 pairs, 3 seeds: %s  mean %.2f sd %.2f' % (name, ['%.1f' % v for v in vals], np.mean(vals), np.std(vals, ddof=1)))
# ---- (4) per-pair AUROCs on the main evaluation
rows = []
for t in IDS:
    g = {n: score(MAIND[t]['id'], c) for n, c in (('c3', C3), ('e10b', E10B), ('e10', E10))}
    for s, D in MAIND[t]['ood'].items():
        best = max(100 * BL[t]['common'][b][s]['auroc'] for b in BASELINES if s in BL[t]['common'][b])
        rows.append(dict(id=ID_NAME[t], ood=s, best_of_eight=best, **{n: 100 * auroc_fast(g[n], score(D, c)) for n, c in (('c3', C3), ('e10b', E10B), ('e10', E10))}))
REPORT['per_pair'] = rows
os.makedirs(os.path.join(RESULTS, 'ablations'), exist_ok=True)
json.dump(REPORT, open(os.path.join(RESULTS, 'ablations', 'robustness.json'), 'w'), indent=1)
print('saved results/ablations/robustness.json')
