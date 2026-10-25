# frontier.py -- Ablations around the Euphonium rule: forward-pass budget, the difference form, the scale window, single noise
#   (A) antithetic-pair atom sets over 10 scales x 3 rotations with at most 2, 4, 6, 8, 10 forward passes, in the log-ratio form
#       (logr, Euphonium) and in the difference form e_k(Tx) - e_k(x) (diff, the first approximation of the discretised integral):
#       the best of each budget on the selection data (same data and tie rule as analysis/select_rule.py), main evaluation afterwards.
#   (B) the scale window: candidates over 13 scales (sigma_1..sigma_13; five IDs, the FashionMNIST development run has 10 scales):
#       do atoms at sigma_11..sigma_13 enter the top ten of the selection data?
#   (C) +eps-only ("single") atom sets from the N_<tag> runs, at most 6 forward passes, on the images that the pair and the single
#       runs have in common (and, for reference, Euphonium on the same images).
#   (D) Euphonium's three atoms in both forms on the main evaluation (Table A9, column "差Eu").
#   Writes results/ablations/frontier.json (read by paper/gen/make_paper.py).
import os, sys, json, itertools, time
import numpy as np
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common import TIERS, RUN, DEVRUN, POOL, RESULTS, BASELINES, ID_NAME, feat, has_feat, common_idx

IDS = list(TIERS); TFS = ['rot90', 'rot180', 'rot270']
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


def auroc_fast(a, b):
    n0, n1 = len(a), len(b); r = rankdata(np.concatenate([a, b]))
    return (r[:n0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)


def parts(F, nk):
    """per-atom values [N, 3 (rotation), nk (scale), 2 (logr, diff)]"""
    e = F['e'][:, :nk, 0, :]; i0 = F['tfs'].index('id')
    out = np.zeros((e.shape[0], 3, nk, 2))
    for j, t in enumerate(TFS):
        a, b = e[:, :, F['tfs'].index(t)], e[:, :, i0]
        out[:, j, :, 0] = np.log(a / b); out[:, j, :, 1] = a - b
    return out


def sub(P, idx, keep):
    pos = {int(i): j for j, i in enumerate(idx.tolist())}; rows = [pos[int(i)] for i in keep if int(i) in pos]
    return P[rows]


def score(P, cand, form):
    s = np.zeros(P.shape[0])
    for T, k in cand: s += P[:, T, k, form]
    return s


def nfe_pair(cand): return 2 * (len(set(k for _, k in cand)) + len(cand))
def nfe_single(cand): return len(set(k for _, k in cand)) + len(cand)
def fmt(cand):
    return ' + '.join('%s{%s}' % (TFS[T].replace('rot', 'R'), ','.join(str(k + 1) for TT, k in sorted(cand) if TT == T)) for T in range(3) if any(TT == T for TT, _ in cand))
def nat_sets(t): return [s for s in TIERS[t]['ood'] if not s.endswith('__vflip') and 'synth' not in s]
def best_of_eight(t, s): return max(100 * BL[t]['common'][b][s]['auroc'] for b in BASELINES if s in BL[t]['common'][b])
def best_mean_of_eight(t): return max(np.mean([100 * BL[t]['common'][b][s]['auroc'] for s in nat_sets(t)]) for b in BASELINES)


def pick(items):
    """the selection rule: largest mean; within 0.1 points tied -> fewest forward passes, then the largest worst-ID mean"""
    top = max(v['sel'] for _, v in items); tied = [(k, v) for k, v in items if v['sel'] >= top - 0.1]
    return min(tied, key=lambda kv: (kv[1]['nfe'], -min(kv[1]['per'].values())))


t0 = time.time()
# ---------------- (A) antithetic pair: selection data (all development images) and main evaluation
NK = 13
SEL, MAIN = {}, {}
for t in IDS:
    nk = min(NK, FT(DEVRUN[t], TIERS[t]['id'])['e'].shape[1])
    SEL[t] = dict(nk=nk, id=parts(FT(DEVRUN[t], TIERS[t]['id']), nk), pool={s: parts(FT(RUN['pair'][t], s), nk) for s in POOL[t]})
    Fi = FT(RUN['pair'][t], TIERS[t]['id'])
    MAIN[t] = dict(id=sub(parts(Fi, NK), Fi['idx'], COM[t]['__id__']), ood={})
    for s in TIERS[t]['ood']:
        F = FT(RUN['pair'][t], s); MAIN[t]['ood'][s] = sub(parts(F, NK), F['idx'], COM[t][s])
    print(t, 'dev nk', nk, 'sel id', SEL[t]['id'].shape[0], 'main id', MAIN[t]['id'].shape[0], '%.0fs' % (time.time() - t0), flush=True)


def sel_mean(cand, form, ids=IDS):
    v = {}
    for t in ids:
        g0 = score(SEL[t]['id'], cand, form)
        v[t] = 100 * np.mean([auroc_fast(g0, score(P, cand, form)) for P in SEL[t]['pool'].values()])
    return float(np.mean(list(v.values()))), v


def main_eval(cand, form):
    out = dict(main={}, wins=0, npairs=0, per={}, nIDbest=0)
    for t in IDS:
        g0 = score(MAIN[t]['id'], cand, form); nat = []
        for s, P in MAIN[t]['ood'].items():
            a = 100 * auroc_fast(g0, score(P, cand, form)); out['per'][(t, s)] = a
            if not s.endswith('__vflip') and 'synth' not in s: nat.append(a)
            out['npairs'] += 1; out['wins'] += int(a >= best_of_eight(t, s) - 0.05)
        out['main'][t] = float(np.mean(nat))
        out['nIDbest'] += int(out['main'][t] >= best_mean_of_eight(t) - 0.05)
    out['main_mean'] = float(np.mean(list(out['main'].values())))
    return out


ATOMS10 = [(T, k) for T in range(3) for k in range(10)]
cands = [list(c) for r in range(1, 4) for c in itertools.combinations(ATOMS10, r) if nfe_pair(list(c)) <= 10]
print('pair candidates (10 scales, NFE<=10):', len(cands), flush=True)
RES = {}
for form, fname in ((0, 'logr'), (1, 'diff')):
    for c in cands:
        m, per = sel_mean(c, form); RES[(fname, tuple(sorted(c)))] = dict(sel=m, per=per, nfe=nfe_pair(c))
    print(fname, 'done %.0fs' % (time.time() - t0), flush=True)
REPORT = dict(A={}, B={}, C={}, D={})
print('\n(A) pair atom sets: best of each budget on the selection data (main evaluation afterwards)')
for fname in ('logr', 'diff'):
    for budget in (2, 4, 6, 8, 10):
        items = [(k, v) for (f, k), v in RES.items() if f == fname and v['nfe'] <= budget]
        if not items: continue
        key, v = pick(items); m = main_eval(list(key), 0 if fname == 'logr' else 1)
        REPORT['A']['%s_nfe%d' % (fname, budget)] = dict(cand=fmt(list(key)), nfe=v['nfe'], sel=v['sel'], sel_min=min(v['per'].values()), main=m['main'], main_mean=m['main_mean'], wins=m['wins'], nIDbest=m['nIDbest'])
        print('%-4s NFE<=%2d  %-28s nfe %2d  sel %.2f (min %.1f)  main %.1f  wins %d/63  IDbest %d  ' % (fname, budget, fmt(list(key)), v['nfe'], v['sel'], min(v['per'].values()), m['main_mean'], m['wins'], m['nIDbest']) + ' '.join('%.1f' % m['main'][t] for t in IDS), flush=True)
    ranked = sorted([(k, v) for (f, k), v in RES.items() if f == fname], key=lambda kv: -kv[1]['sel'])[:5]
    REPORT['A'][fname + '_top5'] = [(fmt(list(k)), v['nfe'], round(v['sel'], 2)) for k, v in ranked]
    print('   top5', REPORT['A'][fname + '_top5'])
# ---------------- (D) Euphonium's atoms in both forms (Table A9, "差Eu")
E10 = [(0, 0), (0, 6), (1, 0)]
for fname, form in (('logr', 0), ('diff', 1)):
    m = main_eval(E10, form); s, per = sel_mean(E10, form)
    REPORT['D'][fname] = dict(sel=s, sel_per=per, main=m['main'], main_mean=m['main_mean'], wins=m['wins'], nIDbest=m['nIDbest'], per=[(t, s_, round(a, 1)) for (t, s_), a in m['per'].items()])
    print('(D) e10 atoms, %s: sel %.2f main %.1f wins %d IDbest %d ' % (fname, s, m['main_mean'], m['wins'], m['nIDbest']) + ' '.join('%.1f' % m['main'][t] for t in IDS))
# ---------------- (B) 13 scales (five IDs): do sigma_11..sigma_13 enter the top ten?
IDS5 = [t for t in IDS if SEL[t]['nk'] >= 13]
ATOMS13 = [(T, k) for T in range(3) for k in range(13)]
cands13 = [list(c) for r in range(1, 4) for c in itertools.combinations(ATOMS13, r) if nfe_pair(list(c)) <= 10]
print('\n(B) 13 scales, five IDs (%s): %d candidates' % (', '.join(ID_NAME[t] for t in IDS5), len(cands13)), flush=True)
for form, fname in ((0, 'logr'), (1, 'diff')):
    R = []
    for c in cands13:
        m, per = sel_mean(c, form, IDS5); R.append((tuple(sorted(c)), m, per))
    R.sort(key=lambda x: -x[1])
    top = R[:10]
    coarse_in_top = [fmt(list(k)) for k, m, per in top if any(kk >= 10 for _, kk in k)]
    best10 = max((x for x in R if all(kk < 10 for _, kk in x[0])), key=lambda x: x[1])
    REPORT['B'][fname] = dict(top10=[(fmt(list(k)), round(m, 2)) for k, m, per in top], coarse_in_top10=coarse_in_top, best_within10=(fmt(list(best10[0])), round(best10[1], 2)))
    print(fname, 'top10:', REPORT['B'][fname]['top10']); print('   coarse atoms in top10:', coarse_in_top, ' best within sigma_1-sigma_10:', REPORT['B'][fname]['best_within10'], '%.0fs' % (time.time() - t0), flush=True)
# ---------------- (C) single-noise atom sets (N_ runs) on the images common to the pair and single runs, NFE <= 6
SELS, MAINS = {}, {}


def aligned(Fp, Fs, nk, keep=None):
    Pp, Ps = parts(Fp, nk), parts(Fs, nk)
    common = set(Fp['idx'].astype(int).tolist()) & set(Fs['idx'].astype(int).tolist())
    if keep is not None: common &= set(int(i) for i in keep)
    order = sorted(common); pp = {int(i): j for j, i in enumerate(Fp['idx'].tolist())}; ps = {int(i): j for j, i in enumerate(Fs['idx'].tolist())}
    return Pp[[pp[i] for i in order]], Ps[[ps[i] for i in order]]


for t in IDS:
    pid, sid = aligned(FT(DEVRUN[t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id'] + '__dev'), 10)
    pool = {s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s), 10) for s in POOL[t]}
    SELS[t] = dict(id=(pid, sid), pool=pool)
    mid = aligned(FT(RUN['pair'][t], TIERS[t]['id']), FT(RUN['single'][t], TIERS[t]['id']), 10, COM[t]['__id__'])
    MAINS[t] = dict(id=mid, ood={s: aligned(FT(RUN['pair'][t], s), FT(RUN['single'][t], s), 10, COM[t][s]) for s in TIERS[t]['ood']})
    print('(C)', t, 'sel id common', pid.shape[0], 'main id common', mid[0].shape[0], flush=True)


def sel_mean_s(cand, form, which):
    v = {}
    for t in IDS:
        g0 = score(SELS[t]['id'][which], cand, form)
        v[t] = 100 * np.mean([auroc_fast(g0, score(P[which], cand, form)) for P in SELS[t]['pool'].values()])
    return float(np.mean(list(v.values()))), v


def main_eval_s(cand, form, which):
    out = dict(main={}, wins=0, npairs=0, nIDbest=0)
    for t in IDS:
        g0 = score(MAINS[t]['id'][which], cand, form); nat = []
        for s, P in MAINS[t]['ood'].items():
            a = 100 * auroc_fast(g0, score(P[which], cand, form))
            if not s.endswith('__vflip') and 'synth' not in s: nat.append(a)
            out['npairs'] += 1; out['wins'] += int(a >= best_of_eight(t, s) - 0.05)
        out['main'][t] = float(np.mean(nat))
        out['nIDbest'] += int(out['main'][t] >= best_mean_of_eight(t) - 0.05)
    out['main_mean'] = float(np.mean(list(out['main'].values())))
    return out


cands_s = [list(c) for r in range(1, 5) for c in itertools.combinations(ATOMS10, r) if nfe_single(list(c)) <= 6]
print('\n(C) single-noise candidates (NFE<=6):', len(cands_s), flush=True)
RESS = {}
for form, fname in ((0, 'logr'), (1, 'diff')):
    for c in cands_s:
        m, per = sel_mean_s(c, form, 1); RESS[(fname, tuple(sorted(c)))] = dict(sel=m, per=per, nfe=nfe_single(c))
    print(fname, 'single done %.0fs' % (time.time() - t0), flush=True)
    for budget in (2, 3, 4, 5, 6):
        items = [(k, v) for (f, k), v in RESS.items() if f == fname and v['nfe'] <= budget]
        if not items: continue
        key, v = pick(items); m = main_eval_s(list(key), form, 1)
        REPORT['C']['%s_single_nfe%d' % (fname, budget)] = dict(cand=fmt(list(key)), nfe=v['nfe'], sel=v['sel'], sel_min=min(v['per'].values()), main=m['main'], main_mean=m['main_mean'], wins=m['wins'], nIDbest=m['nIDbest'])
        print('%-4s single NFE<=%d  %-28s nfe %d  sel %.2f (min %.1f)  main %.1f  wins %d/63  IDbest %d  ' % (fname, budget, fmt(list(key)), v['nfe'], v['sel'], min(v['per'].values()), m['main_mean'], m['wins'], m['nIDbest']) + ' '.join('%.1f' % m['main'][t] for t in IDS), flush=True)
    s, per = sel_mean_s(E10, form, 0); m = main_eval_s(E10, form, 0)      # Euphonium (pair) on the same images, for reference
    REPORT['C']['%s_pair_e10_on_common' % fname] = dict(sel=s, main_mean=m['main_mean'], wins=m['wins'], nIDbest=m['nIDbest'])
    print('%-4s pair e10 on the common images  sel %.2f main %.1f wins %d IDbest %d' % (fname, s, m['main_mean'], m['wins'], m['nIDbest']))
os.makedirs(os.path.join(RESULTS, 'ablations'), exist_ok=True)
json.dump(REPORT, open(os.path.join(RESULTS, 'ablations', 'frontier.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print('saved results/ablations/frontier.json %.0fs' % (time.time() - t0))
