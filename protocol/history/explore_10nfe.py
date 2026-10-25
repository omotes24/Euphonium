# explore_10nfe.py -- the exploratory search that preceded protocol/PROTOCOL_10NFE.md (kept for the record; not the selection itself)
#   Question: is there a configuration with at most 10 forward passes per image that is as good as the best 20-pass grid candidate (c3)?
#   Atoms = (noise draw: pair / single, rotation T, scale k) log-ratios; a candidate is a set of atoms (sum).  Forward passes:
#   upright image once per used scale (2 for a pair, 1 for a single draw) + one per draw for every atom.  Families searched:
#   (a) single noise, one rotation, 1-5 scales; (b) single noise, two rotations, 1-2 scales each; (c) pair on 1-2 scales of one
#   rotation plus 0-2 single atoms; (d) a beam search over all atoms.  Ranked on the selection data (ID development images vs the
#   extra OOD sets), on the images that the pair and the single runs have in common; the top 25 also on the main evaluation.
#   Mixed pair + single candidates add values from two runs with different noise images; the protocol therefore restricts the
#   candidates to pure-pair atom sets (analysis/select_rule.py).  Writes protocol/history/explore_10nfe.json.
import os, sys, json, itertools, time
import numpy as np
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'analysis'))
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
    e = F['e'][:, :NK, 0, :]; i0 = F['tfs'].index('id')
    return F['idx'].astype(int), np.stack([np.log(e[:, :, F['tfs'].index(t)] / e[:, :, i0]) for t in TFS], 1)


def aligned(Fp, Fs, keep=None):
    ip, Lp = atoms(Fp); is_, Ls = atoms(Fs)
    common = set(ip.tolist()) & set(is_.tolist())
    if keep is not None: common &= set(int(i) for i in keep)
    order = sorted(common); pp = {int(i): j for j, i in enumerate(ip.tolist())}; ps = {int(i): j for j, i in enumerate(is_.tolist())}
    return np.stack([Lp[[pp[i] for i in order]], Ls[[ps[i] for i in order]]], 1)   # [N, 2 (pair, single), 3, 10]


def auroc_fast(a, b):
    n0, n1 = len(a), len(b); r = rankdata(np.concatenate([a, b]))
    return (r[:n0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)


t0 = time.time()
SELD, MAIND = {}, {}
for t in IDS:
    SELD[t] = dict(id=aligned(FT(DEVRUN[t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id'] + '__dev')),
                   pool={s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s)) for s in POOL[t]})
    MAIND[t] = dict(id=aligned(FT(RUN['pair'][t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id']), keep=COM[t]['__id__']),
                    ood={s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s), keep=COM[t][s]) for s in TIERS[t]['ood']})
    print(t, 'sel id', SELD[t]['id'].shape[0], 'main id', MAIND[t]['id'].shape[0], flush=True)


def nfe_of(cand):
    up, rot = {}, 0
    for n, T, k in cand:
        up[k] = max(up.get(k, 0), 2 if n == 0 else 1); rot += 2 if n == 0 else 1
    return sum(up.values()) + rot


def score(D, cand):
    s = np.zeros(D.shape[0])
    for n, T, k in cand: s += D[:, n, T, k]
    return s


def sel_mean(cand, per_id=False):
    v = {}
    for t in IDS:
        g0 = score(SELD[t]['id'], cand)
        v[t] = 100 * np.mean([auroc_fast(g0, score(D, cand)) for D in SELD[t]['pool'].values()])
    return (float(np.mean(list(v.values()))), v) if per_id else float(np.mean(list(v.values())))


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


def fmt(cand):
    parts = []
    for n in (0, 1):
        for T in range(3):
            ks = sorted(k + 1 for nn, TT, k in cand if nn == n and TT == T)
            if ks: parts.append('%s %s{%s}' % ('pair' if n == 0 else 'single', TFS[T].replace('rot', 'R'), ','.join(map(str, ks))))
    return ' + '.join(parts)


results = {}
def consider(cand, fam):
    key = tuple(sorted(cand))
    if key in results or nfe_of(cand) > 10: return
    results[key] = dict(fam=fam, nfe=nfe_of(cand), sel=sel_mean(cand))
# (a) single noise, one rotation, 1-5 scales (all subsets)
for T in range(3):
    for r in range(1, 6):
        for ks in itertools.combinations(range(NK), r): consider([(1, T, k) for k in ks], 'a:single-1rot')
print('(a) done', len(results), '%.0fs' % (time.time() - t0), flush=True)
# (b) single noise, two rotations, 1-2 scales each
for T1, T2 in itertools.combinations(range(3), 2):
    for r1 in (1, 2):
        for r2 in (1, 2):
            for k1 in itertools.combinations(range(NK), r1):
                for k2 in itertools.combinations(range(NK), r2):
                    consider([(1, T1, k) for k in k1] + [(1, T2, k) for k in k2], 'b:single-2rot')
print('(b) done', len(results), '%.0fs' % (time.time() - t0), flush=True)
# (c) pair on 1-2 scales of one rotation (+ 1-2 single atoms)
for T in range(3):
    for r in (1, 2):
        for ks in itertools.combinations(range(NK), r):
            base = [(0, T, k) for k in ks]; consider(base, 'c:pair')
            for T2 in range(3):
                for k2 in range(NK):
                    if (T2, k2) in [(T, k) for k in ks]: continue
                    consider(base + [(1, T2, k2)], 'c:pair+single1')
                    for k3 in range(k2 + 1, NK):
                        if (T2, k3) in [(T, k) for k in ks]: continue
                        consider(base + [(1, T2, k2), (1, T2, k3)], 'c:pair+single2')
print('(c) done', len(results), '%.0fs' % (time.time() - t0), flush=True)
# (d) beam search (width 8): add one atom at a time, maximising the selection mean
ATOMS = [(n, T, k) for n in (0, 1) for T in range(3) for k in range(NK)]
beam = [[]]
for step in range(10):
    nxt = {}
    for cand in beam:
        for a in ATOMS:
            if a in cand: continue
            c = cand + [a]
            if nfe_of(c) > 10: continue
            key = tuple(sorted(c))
            if key in nxt: continue
            if key not in results: results[key] = dict(fam='d:beam', nfe=nfe_of(c), sel=sel_mean(c))
            nxt[key] = results[key]['sel']
    if not nxt: break
    beam = [list(k) for k, _ in sorted(nxt.items(), key=lambda kv: -kv[1])[:8]]
print('(d) done', len(results), '%.0fs' % (time.time() - t0), flush=True)
REF = {'c3 (grid, 20 passes)': [(0, 0, k) for k in (0, 2, 4, 6, 8)], 'single R90 odd 5 (10 passes)': [(1, 0, k) for k in (0, 2, 4, 6, 8)]}
ranked = sorted(results.items(), key=lambda kv: -kv[1]['sel'])
rows = []
for name, cand in REF.items():
    m = main_eval(cand); sm, per = sel_mean(cand, True)
    rows.append(dict(name=name, cand=cand, nfe=nfe_of(cand), sel=sm, sel_per=per, **m))
for key, r in ranked[:25]:
    cand = list(key); m = main_eval(cand); sm, per = sel_mean(cand, True)
    rows.append(dict(name=fmt(cand), cand=cand, nfe=r['nfe'], fam=r['fam'], sel=sm, sel_per=per, **m))
json.dump(dict(n_candidates=len(results), rows=rows), open(os.path.join(HERE, 'explore_10nfe.json'), 'w'), indent=1)
print('candidates evaluated:', len(results), '%.0fs' % (time.time() - t0))
print('%-44s NFE   sel   main  wins   ' % 'candidate' + ' '.join('%8s' % t for t in IDS))                # columns: the six ID tags
for r in rows:
    print('%-44s %3d %5.1f  %5.1f  %2d/%d  ' % (r['name'][:44], r['nfe'], r['sel'], r['main_mean'], r['wins'], r['npairs']) + ' '.join('%8.1f' % r['main'][t] for t in IDS))
