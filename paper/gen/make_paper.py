# make_paper.py -- tables, figures and number macros of the paper (paper/main.tex, Japanese) from the per-image statistics
#   (inputs/, release asset inputs-v1) and the result files of the analysis (results/):
#     figs/fig_concept.pdf   Figure 1: NLL vs Lambda (likelihood axis and orientation axis)
#     figs/fig_verify.pdf    Figure 3: (a) shared-noise estimate (40 passes) vs exact ratio (PF-ODE), (b) per-scale log-ratio AUROC (13 scales)
#     figs/fig_G.pdf         Figure A1: integrand, per-scale log-ratios of R90 / R180 with the atoms of Euphonium, distribution of the score
#     figs/fig_gode.pdf      Figure A2: Euphonium (10 passes) vs the exact ratio, per image (four IDs)
#     tab_*.tex, supp_*.tex, numbers.tex   table rows and the macros used in the text
#   The Euphonium rule is read from rule.json (selected by analysis/select_rule.py, protocol/PROTOCOL_10NFE.md); every other
#   configuration (grid candidates, 40-pass estimate, exact ratio, ...) is an Ablation.
#   Run from anywhere: python paper/gen/make_paper.py   (needs results/*.json: bash reproduce.sh runs the analysis first)
import os, sys, json
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')                  # reproducible PDF metadata
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE); ROOT = os.path.dirname(PAPER)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from common import feat, nll, auroc, dg, dg_cross, TIERS, INPUTS, RESULTS, BENCH

for f in font_manager.findSystemFonts():
    if 'NotoSansCJK-Regular' in f or 'NotoSansCJK-Bold' in f: font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2, 'ytick.major.size': 2,
                     'legend.fontsize': 6.5, 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'axes.unicode_minus': False})
NAVY, RED, GRAY, DARK, TEAL = '#1f3b73', '#c0392b', '#8c8c8c', '#404040', '#2a9d8f'
SETS = [('cifar10_test', 'CIFAR-10(分布内)', NAVY), ('svhn_test', 'SVHN', RED), ('cifar100_test', 'CIFAR-100', GRAY), ('textures', 'Textures', DARK)]
COL = {s: c for s, _, c in SETS}
OUT = os.path.join(PAPER, 'figs'); os.makedirs(OUT, exist_ok=True)
M = {}


def load(s):
    a, b = nll('F_edmc10', s), nll('F_edmc10_rot90', s)
    assert (a['idx'] == b['idx']).all()
    return a['idx'], a['bpd'], b['bpd'] - a['bpd']          # Λ (bits/dim) = NLL(R90 x) - NLL(x) = log p(x) - log p(R90 x)


DATA = {s: load(s) for s, _, _ in SETS}

# ================================================================ Figure 1: concept (likelihood axis vs orientation axis)
EX = {}
for s, title in (('cifar10_test', 'CIFAR-10(分布内)'), ('svhn_test', 'SVHN(分布外)')):
    idx, bpd, lam = DATA[s]
    pick = [int(np.argmin(np.abs(lam - np.quantile(lam, q)))) for q in (0.25, 0.5, 0.75)]
    EX[s] = [(int(idx[j]), bpd[j], bpd[j] + lam[j], lam[j]) for j in pick]
fig = plt.figure(figsize=(3.3, 1.9))
L, B, W, H = 0.145, 0.165, 0.63, 0.6
ax = fig.add_axes([L, B, W, H]); ax_t = fig.add_axes([L, B + H + 0.012, W, 0.1], sharex=ax); ax_r = fig.add_axes([L + W + 0.01, B, 0.17, H], sharey=ax)
rng = np.random.RandomState(0)
for s, lab, c in SETS:
    idx, bpd, lam = DATA[s]
    sel = rng.choice(len(bpd), 400, replace=False)
    ax.scatter(bpd[sel], lam[sel], s=2.2, c=c, alpha=0.65, lw=0, label=lab, rasterized=True)
    ax_t.hist(bpd, bins=np.linspace(0.8, 5.0, 60), histtype='step', color=c, lw=0.9, density=True)
    ax_r.hist(lam, bins=np.linspace(-0.12, 0.40, 66), histtype='step', color=c, lw=0.9, density=True, orientation='horizontal')
ax.axhline(0, c='k', lw=0.4, ls=':')
ax.set_xlim(0.8, 5.0); ax.set_ylim(-0.12, 0.40)
ax.set_xlabel('尤度の軸：NLL$(x)$ (bits/dim)', fontsize=7.0, labelpad=2)
ax.set_ylabel('向きの軸：$\\Lambda=\\log p(x)-\\log p(\\mathbf{R}_{90}x)$', fontsize=7.0, labelpad=2)
for a in (ax_t, ax_r):
    a.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for k in a.spines: a.spines[k].set_visible(False)
ax.legend(loc='upper left', markerscale=4, frameon=False, handletextpad=0.2, borderaxespad=0.2, ncol=1, labelspacing=0.2, fontsize=6.2)
ax.text(4.95, 0.385, '分布内：正立のほうが\nもっともらしい($\\Lambda>0$)', ha='right', va='top', fontsize=6, color=NAVY)
ax.text(0.9, -0.105, 'SVHN：尤度は高いが $\\Lambda\\approx0$', ha='left', va='bottom', fontsize=6, color=RED)
ax.tick_params(labelsize=6.4)
fig.savefig(f'{OUT}/fig_concept.pdf', dpi=300); plt.close(fig)
idx0, bpd0, lam0 = DATA['cifar10_test']
KEY = {'svhn_test': 'SVHN', 'cifar100_test': 'Ch', 'textures': 'Tex'}   # macro names (no digits)
for s, lab, c in SETS[1:]:
    _, bpd1, lam1 = DATA[s]
    M['nll' + KEY[s]] = 100 * auroc(-bpd0, -bpd1); M['lam' + KEY[s]] = 100 * auroc(lam0, lam1)
M['lamMedID'] = np.median(lam0); M['lamMedSVHN'] = np.median(DATA['svhn_test'][2])
M['nllMedID'] = np.median(bpd0); M['nllMedSVHN'] = np.median(DATA['svhn_test'][1])
print('fig_concept', {k: round(float(v), 3) for k, v in M.items()})
print('examples', {s: [(i, round(a, 2), round(b, 2), round(l, 3)) for i, a, b, l in ex] for s, ex in EX.items()})

# ================================================================ the Euphonium rule (rule.json; selected by analysis/select_rule.py, protocol/PROTOCOL_10NFE.md)
RULE = json.load(open(os.path.join(ROOT, 'rule.json')))
ATOMS = [(str(T), int(k)) for T, k in RULE['atoms']]          # atoms (rotation T, scale k): Euphonium adds the log-ratios log e_k(T x)/e_k(x)
NOISE, NFEG = RULE['noise'], int(RULE['nfe'])
def nfe_atoms(atoms, noise='pair'):
    """forward passes per image: the upright image once per used scale (twice for a pair) + every atom once (twice for a pair)"""
    return (2 if noise == 'pair' else 1) * (len(set(k for _, k in atoms)) + len(atoms))
assert NFEG == nfe_atoms(ATOMS, NOISE), 'rule.json: nfe does not match the atoms'
TG = tuple(dict.fromkeys(T for T, _ in ATOMS))                 # rotations used (ordered)
KG = sorted(set(k for _, k in ATOMS))                           # scales used (union; marked in the figures)
KGT = {T: sorted(k for TT, k in ATOMS if TT == T) for T in TG}  # scales per rotation
R3 = ('rot90', 'rot180', 'rot270')
def atom_score(F, atoms, mode='logr'):
    """sum over atoms: logr = sum log e_k(Tx)/e_k(x), abs = sum [e_k(Tx) - e_k(x)]  [N]"""
    e = F['e'][:, :, 0, :]; i0 = F['tfs'].index('id'); s = np.zeros(e.shape[0])
    for T, k in atoms:
        a, b = e[:, k, F['tfs'].index(T)], e[:, k, i0]
        s += np.log(a / b) if mode == 'logr' else (a - b)
    return s
def atom_lrs(F, atoms):
    """per-atom log-ratios [N, |atoms|]"""
    e = F['e'][:, :, 0, :]; i0 = F['tfs'].index('id')
    return np.stack([np.log(e[:, k, F['tfs'].index(T)] / e[:, k, i0]) for T, k in atoms], 1)
def escore(F): return atom_score(F, ATOMS)
def atoms_of(tfs, ks): return [(T, k) for T in tfs for k in ks]
RUN = {'pair': {'edmc10': 'F_edmc10', 'l6svhn': 'F_l6svhn', 'dpceleba': 'F_dpceleba', 'edmc100': 'F_edmc100', 'in64': 'F2_in64', 'fmnist': 'F_fmnist'},
       'single': {'edmc10': 'N_edmc10', 'l6svhn': 'N_l6svhn', 'dpceleba': 'N_dpceleba', 'edmc100': 'N_edmc100', 'in64': 'N_in64', 'fmnist': 'N_fmnist'}}
NLLRUN = {'edmc10': 'F_edmc10', 'l6svhn': 'F_l6svhn', 'edmc100': 'F_edmc100', 'in64': 'F_in64'}      # IDs with the exact likelihood (PF-ODE) and Lambda_90
DEVRUN = {'edmc10': 'B_edmc10', 'l6svhn': 'P_l6svhn', 'dpceleba': 'P_dpceleba', 'edmc100': 'P_edmc100', 'in64': 'I2_edm2xs', 'fmnist': 'P_fmnist'}   # ID development images
COM = json.load(open(os.path.join(BENCH, 'common_idx.json')))                     # the images every compared method has (the subsets of the tables)
F4 = json.load(open(os.path.join(RESULTS, 'baselines_pairs.json'))); EXR = json.load(open(os.path.join(RESULTS, 'exact_ratio_pairs.json')))   # analysis/baselines_eval.py, analysis/exact_ratio.py
def sub(score, idx, keep):
    pos = {int(i): j for j, i in enumerate(idx.tolist())}; return score[[pos[i] for i in keep if i in pos]]
FEAT = {}
def FT(run, s):
    if (run, s) not in FEAT:
        F = feat(run, s)
        if F['e'].shape[2] > 1: F['e'] = F['e'][:, :, :1, :]
        FEAT[(run, s)] = F
    return FEAT[(run, s)]
def lrs(F, ks, T='rot90'):
    """per-scale log-ratios log e_k(T x)/e_k(x)  [N, |ks|]"""
    e = F['e'][:, :, 0, :]; return np.log(e[:, ks, F['tfs'].index(T)] / e[:, ks, F['tfs'].index('id')])
def gscore(t, s, atoms=ATOMS, noise=NOISE):
    """the statistic of the atom set (noise) on the common images of pair (t, s) (s = '__id__' is the ID set)"""
    name = TIERS[t]['id'] if s == '__id__' else s
    F = FT(RUN[noise][t], name); return sub(atom_score(F, atoms), F['idx'], COM[t][s])
ODD, EVEN, ALL = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9], list(range(10))
# candidates: (name, atoms, noise, forward passes, columns "rotation" and "scales" of Table A3)
#   Euphonium (protocol/PROTOCOL_10NFE.md): every pure-pair atom set with <= 10 passes (1285) on the selection data; e10 is selected, e10b and e10c are tied 2nd and 3rd.
#     e8, e6, e4: the best of the selection data with a budget of 8, 6 and 4 passes (analysis/ablations/frontier.py; reference)
#   Ablation (protocol/PROTOCOL_20NFE.md): the twelve grid candidates c1--c12 with <= 20 passes, r60 and r80 (all ten scales), dev (chosen on development copies of the main OOD sets)
def cand_grid(cid, tfs, ks, noise, nfe): return (cid, atoms_of(tfs, ks), noise, nfe, ', '.join(RT0[t] for t in tfs), None, ks)
RT0 = {'rot90': '$\\Rot{90}$', 'rot180': '$\\Rot{180}$', 'rot270': '$\\Rot{270}$'}
def cand_atoms(cid, atoms, nfe):
    per = {}
    for T, k in atoms: per.setdefault(T, []).append(k)
    return (cid, atoms, 'pair', nfe, ', '.join(RT0[T] for T in per), '; '.join('%s：$\\{%s\\}$' % (RT0[T], ','.join('\\sigma_{%d}' % (k + 1) for k in sorted(ks))) for T, ks in per.items()), None)
CAND = [cand_grid('c1', ('rot180',), ODD, 'pair', 20), cand_grid('c2', ('rot180',), EVEN, 'pair', 20), cand_grid('c3', ('rot90',), ODD, 'pair', 20), cand_grid('c4', ('rot90',), EVEN, 'pair', 20),
        cand_grid('c5', ('rot90', 'rot180'), [0, 4, 8], 'pair', 18), cand_grid('c6', ('rot90', 'rot180'), [1, 5, 9], 'pair', 18),
        cand_grid('c7', ('rot180',), ALL, 'single', 20), cand_grid('c8', ('rot90',), ALL, 'single', 20), cand_grid('c9', ('rot90', 'rot180'), ODD, 'single', 15), cand_grid('c10', ('rot90', 'rot180'), EVEN, 'single', 15),
        cand_grid('c11', R3, ODD, 'single', 20), cand_grid('c12', R3, EVEN, 'single', 20),
        cand_grid('r60', ('rot90', 'rot180'), ALL, 'pair', 60), cand_grid('r80', R3, ALL, 'pair', 80), cand_grid('dev', ('rot180',), [0, 2, 3, 5, 7], 'pair', 20),
        cand_atoms('e10', [('rot90', 0), ('rot90', 6), ('rot180', 0)], 10), cand_atoms('e10b', [('rot90', 0), ('rot90', 5), ('rot180', 0)], 10),
        cand_atoms('e10c', [('rot90', 0), ('rot180', 0), ('rot270', 5)], 10), cand_atoms('e8', [('rot90', 5), ('rot180', 0)], 8), cand_atoms('e6', [('rot90', 0), ('rot180', 0)], 6), cand_atoms('e4', [('rot90', 0)], 4)]
for c in CAND[-6:]: assert nfe_atoms(c[1]) == c[3], c
GA = {}      # (candidate, t, s) -> AUROC (%) on the main evaluation (single-noise candidates only where the N_ runs of the final split exist)
for cid, atoms, noise, nfe, _, _, _ in CAND:
    for t in TIERS:
        try: g0 = gscore(t, '__id__', atoms, noise)
        except FileNotFoundError: continue
        for s in TIERS[t]['ood']:
            try: GA[(cid, t, s)] = 100 * auroc(g0, gscore(t, s, atoms, noise))
            except FileNotFoundError: pass
MAINID = [c[0] for c in CAND if sorted(c[1]) == sorted(ATOMS) and c[2] == NOISE]
assert MAINID, 'the rule of rule.json is not among the candidates'; MAINID = MAINID[0]
G90 = {}      # (t, s) -> AUROC (%) of the 40-pass estimate (R90, 10 scales, pair): natural and upside-down OOD sets (Tables A10, A11: column Eu10)
for t in TIERS:
    g0 = gscore(t, '__id__', atoms_of(('rot90',), ALL), 'pair')
    for s in TIERS[t]['ood']:
        if 'synth' not in s: G90[(t, s)] = 100 * auroc(g0, gscore(t, s, atoms_of(('rot90',), ALL), 'pair'))
print('rule', RULE, 'main cand', MAINID)
print('G (main) per ID, natural sets:', {t: round(np.mean([GA[(MAINID, t, s)] for s in TIERS[t]['ood'] if 'vflip' not in s and 'synth' not in s]), 1) for t in TIERS})
SEL = json.load(open(os.path.join(RESULTS, 'ablations', 'grid_selection.json')))   # values of the grid candidates on the selection data (ID development images vs extra OOD sets): analysis/ablations/grid_candidates.py
IDS = [('edmc10', 'CIFAR-10'), ('l6svhn', 'SVHN'), ('dpceleba', 'CelebA'), ('edmc100', 'CIFAR-100'), ('in64', 'ImageNet-64'), ('fmnist', 'FashionMNIST')]
KEYID = {'edmc10': 'Cten', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'Ch', 'in64': 'IN', 'fmnist': 'FM'}
IDCOL = {'edmc10': NAVY, 'l6svhn': RED, 'edmc100': GRAY, 'in64': TEAL, 'dpceleba': '#8e44ad', 'fmnist': '#d68910'}
NAT = {t: [s for s in TIERS[t]['ood'] if not s.endswith('__vflip') and 'synth' not in s] for t, _ in IDS}    # natural OOD sets
NAMES = {'cifar10_test': 'CIFAR-10', 'cifar100_test': 'CIFAR-100', 'svhn_test': 'SVHN', 'celeba_test': 'CelebA', 'textures': 'Textures', 'tin_val': 'Tiny-ImageNet',
         'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet', 'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'synth_noise': '一様ノイズ', 'synth_const': '定数画像',
         'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O',
         'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
         'g32_notmnist': 'notMNIST', 'g32_synth_noise': '一様ノイズ', 'g32_synth_const': '定数画像'}
def sn(s): return NAMES[s] if s in NAMES else ('上下反転' if s.endswith('__vflip') else s)
def val_ode(t, s): return 100 * EXR[t]['sets'][s]['auroc'] if t in EXR and s in EXR[t]['sets'] else None
def val_g90(t, s): return G90.get((t, s))

# ================================================================ checking the estimates: per-image correspondence with the exact ratio (four IDs)
#   est = shared-noise estimate on 10 scales (first approximation of Eq. (15), 40 passes, bits/dim),  G = Euphonium (sum of the atoms' log-ratios, 10 passes)
def align(idx_a, idx_b):
    pa = {int(v): i for i, v in enumerate(idx_a.tolist())}; k = [(pa[int(v)], j) for j, v in enumerate(idx_b.tolist()) if int(v) in pa]
    return np.array([a for a, _ in k]), np.array([b for _, b in k])
VER = {}      # t -> dict(lam=..., est=..., G=..., rho_est, rho_G, rho_pool_G, sets...)
for t in NLLRUN:
    T = TIERS[t]; run = RUN['pair'][t]; nrun = NLLRUN[t]
    sets = [T['id']] + [s for s in NAT[t] if os.path.exists(f'{INPUTS}/nll/{nrun}_rot90/{s}.npz')]
    d = {}
    for s in sets:
        F = FT(run, s); N0, N1 = nll(nrun, s), nll(nrun + '_rot90', s); assert (N0['idx'] == N1['idx']).all()
        ia, ib = align(F['idx'], N0['idx']); lam = (N1['bpd'] - N0['bpd'])[ib]
        sg = F['sigmas'][:10]; h = float(np.log(sg[1] / sg[0])); D = F['dim']
        est = h * (F['e'][ia][:, :10, 0, F['tfs'].index('rot90')] - F['e'][ia][:, :10, 0, F['tfs'].index('id')]).sum(1) / (D * np.log(2))
        d[s] = dict(lam=lam, est=est, G=escore(F)[ia], rho_est=float(spearmanr(lam, est)[0]), rho_G=float(spearmanr(lam, escore(F)[ia])[0]))
    lam_all = np.concatenate([d[s]['lam'] for s in sets]); G_all = np.concatenate([d[s]['G'] for s in sets]); est_all = np.concatenate([d[s]['est'] for s in sets])
    VER[t] = dict(sets=sets, d=d, rho_pool_G=float(spearmanr(lam_all, G_all)[0]), rho_pool_est=float(spearmanr(lam_all, est_all)[0]))
    k = KEYID[t]; M['rhoEst' + k] = d[T['id']]['rho_est']; M['rhoG' + k] = d[T['id']]['rho_G']; M['rhoPoolG' + k] = VER[t]['rho_pool_G']; M['rhoPoolEst' + k] = VER[t]['rho_pool_est']
    M['lamSd' + k] = float(d[T['id']]['lam'].std()); M['lamMean' + k] = float(d[T['id']]['lam'].mean())
    print('verify', t, 'ID rho est %.3f G %.3f | pooled est %.3f G %.3f' % (d[T['id']]['rho_est'], d[T['id']]['rho_G'], VER[t]['rho_pool_est'], VER[t]['rho_pool_G']),
          {sn(s): (round(d[s]['rho_est'], 2), round(d[s]['rho_G'], 2)) for s in sets[1:]})
M['rhoID'] = M['rhoEstCten']; M['rhoCh'] = M['rhoEstCh']
M['rhoEstMin'] = min(M['rhoEst' + KEYID[t]] for t in NLLRUN); M['rhoEstMax'] = max(M['rhoEst' + KEYID[t]] for t in NLLRUN)
M['rhoGMin'] = min(M['rhoG' + KEYID[t]] for t in NLLRUN); M['rhoGMax'] = max(M['rhoG' + KEYID[t]] for t in NLLRUN)
M['rhoPoolGMin'] = min(M['rhoPoolG' + KEYID[t]] for t in NLLRUN); M['rhoPoolGMax'] = max(M['rhoPoolG' + KEYID[t]] for t in NLLRUN)
M['rhoEstMinName'] = dict(IDS)[min(NLLRUN, key=lambda t: M['rhoEst' + KEYID[t]])]

# ---- per-scale log-ratio AUROC (13 scales, mean over the natural OOD sets) and the AUROC of sums over other atom sets (six IDs)
NK = 13
PS = {}; PSD = {}
for t, name in IDS:
    run = RUN['pair'][t]; F0 = FT(run, TIERS[t]['id']); L0 = lrs(F0, list(range(NK)))
    per = np.zeros((len(NAT[t]), NK)); gsum = {}
    for i, s in enumerate(NAT[t]):
        F1 = FT(run, s); L1 = lrs(F1, list(range(NK)))
        for k in range(NK): per[i, k] = 100 * auroc(sub(L0[:, k], F0['idx'], COM[t]['__id__']), sub(L1[:, k], F1['idx'], COM[t][s]))
    PS[t] = per.mean(0)
    PSD[t] = {}
    for lab, at in (('five', ATOMS), ('ten', atoms_of(('rot90',), ALL)), ('thirteen', atoms_of(('rot90',), list(range(NK)))), ('coarse', atoms_of(('rot90',), [10, 11, 12]))):
        a = [100 * auroc(sub(atom_score(F0, at), F0['idx'], COM[t]['__id__']), sub(atom_score(FT(run, s), at), FT(run, s)['idx'], COM[t][s])) for s in NAT[t]]
        PSD[t][lab] = float(np.mean(a))
    for lab, at in (('dfive', ATOMS), ('dten', atoms_of(('rot90',), ALL))):          # difference form (first approximation of Eq. (15); dten is the estimate compared with the exact ratio in Figure 3(a))
        a = [100 * auroc(sub(atom_score(F0, at, 'abs'), F0['idx'], COM[t]['__id__']), sub(atom_score(FT(run, s), at, 'abs'), FT(run, s)['idx'], COM[t][s])) for s in NAT[t]]
        PSD[t][lab] = float(np.mean(a))
    M['dFive' + KEYID[t]] = PSD[t]['dfive']; M['dTen' + KEYID[t]] = PSD[t]['dten']
    # per-scale AUROC of the R180 log-ratios (Euphonium uses R180; second row of Table A9)
    L180 = lrs(F0, list(range(NK)), 'rot180'); per180 = np.zeros((len(NAT[t]), NK))
    for i, s in enumerate(NAT[t]):
        F1 = FT(run, s); L1 = lrs(F1, list(range(NK)), 'rot180')
        for k in range(NK): per180[i, k] = 100 * auroc(sub(L180[:, k], F0['idx'], COM[t]['__id__']), sub(L1[:, k], F1['idx'], COM[t][s]))
    PS180 = PS.setdefault('_180', {}); PS180[t] = per180.mean(0)
    PSD[t]['ode'] = float(np.mean([val_ode(t, s) for s in NAT[t]])) if all(val_ode(t, s) is not None for s in NAT[t]) else None
    PSD[t]['dprime'] = (L0.mean(0) / L0.std(0)).tolist()
    k = KEYID[t]
    M['psFine' + k] = float(PS[t][0]); M['psCoarse' + k] = float(PS[t][8]); M['psMax' + k] = float(PS[t].max()); M['psArgmax' + k] = int(np.argmax(PS[t]) + 1)
    M['psMid' + k] = float(np.mean(PS[t][[4, 6, 8]])); M['psDropA' + k], M['psDropB' + k], M['psDropC' + k] = float(PS[t][10]), float(PS[t][11]), float(PS[t][12])
    M['gFive' + k], M['gTen' + k], M['gThirteen' + k], M['gCoarse' + k] = PSD[t]['five'], PSD[t]['ten'], PSD[t]['thirteen'], PSD[t]['coarse']
    print('per-scale', name, np.round(PS[t], 1).tolist(), 'five %.1f ten %.1f thirteen %.1f coarse %.1f ode %s' % (PSD[t]['five'], PSD[t]['ten'], PSD[t]['thirteen'], PSD[t]['coarse'], PSD[t]['ode']))
SIG13 = FT(RUN['pair']['edmc10'], 'cifar10_test')['sigmas'][:NK]
M['sigDropA'] = float(SIG13[10]); M['sigDropB'] = float(SIG13[11]); M['sigDropC'] = float(SIG13[12]); M['sigTen'] = float(SIG13[9])
# Table A9: per-scale AUROC (six IDs x 13 scales) and sums over atom sets
L = []
for t, name in IDS:
    cells = ['%s' % name] + [('\\textbf{%.1f}' if k in KGT.get('rot90', []) else '%.1f') % v for k, v in enumerate(PS[t])] + ['%.1f' % PSD[t]['five'], '%.1f' % PSD[t]['ten'], '%.1f' % PSD[t]['thirteen'], '%.1f' % PSD[t]['dfive'], '%.1f' % PSD[t]['dten'], '--' if PSD[t]['ode'] is None else '%.1f' % PSD[t]['ode']]
    L.append(' & '.join(cells) + '\\\\')
    cells180 = ['%s($\\Rot{180}$)' % name] + [('\\textbf{%.1f}' if k in KGT.get('rot180', []) else '%.1f') % v for k, v in enumerate(PS['_180'][t])] + ['', '', '', '', '', '']
    L.append(' & '.join(cells180) + '\\\\')
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'supp_scale.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: per-scale log-ratio AUROC (mean over the natural OOD sets), sums over atom sets, exact ratio\n' + '\n'.join(L) + '\n')

# ================================================================ Figure 3: checking the estimates (one column, two panels): (a) estimate vs exact ratio, (b) per-scale AUROC
INTEG = {}   # for Figure A1: quartiles of the integrand (CIFAR-10 model, 10 scales)
fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.62), gridspec_kw=dict(width_ratios=[0.9, 1.1], wspace=0.5))
RHO = {}
for s, lab, c in SETS:
    F = feat('F_edmc10', s); i0, j = F['tfs'].index('id'), F['tfs'].index('rot90'); D = F['dim']
    sg = F['sigmas'][:10]; h = float(np.log(sg[1] / sg[0]))
    d = (F['e'][:, :10, 0, j] - F['e'][:, :10, 0, i0]) / (D * np.log(2))
    INTEG[s] = (sg, np.quantile(d, [0.25, 0.5, 0.75], axis=0))
    N0, N1 = nll('F_edmc10', s), nll('F_edmc10_rot90', s)
    ode = N1['bpd'] - N0['bpd']
    pos = {v: i for i, v in enumerate(F['idx'].tolist())}
    k = np.array([i for i, v in enumerate(N0['idx'].tolist()) if v in pos]); jj = np.array([pos[N0['idx'][i]] for i in k])
    est = h * d[jj].sum(1)
    RHO[s] = spearmanr(ode[k], est)[0]
    ax[0].scatter(ode[k], est, s=1.6, c=c, alpha=0.55, lw=0, rasterized=True, label=lab)
assert abs(RHO['cifar10_test'] - M['rhoEstCten']) < 1e-6
lim = (-0.12, 0.40)
ax[0].plot(lim, lim, c='k', lw=0.5, ls='--'); ax[0].set_xlim(lim); ax[0].set_ylim(-0.12, 0.30)
ax[0].set_xlabel('厳密な $\\Lambda$ (PF-ODE, 512 回)', fontsize=6.3, labelpad=1.5)
ax[0].set_ylabel('共有ノイズによる推定 (40 回)', fontsize=6.3, labelpad=1.5)
ax[0].set_title('(a) 推定と厳密な比 (bits/dim)', fontsize=6.2, loc='left', pad=2)
ax[0].text(0.39, -0.105, '分布内の画像での\n順位相関 $\\rho$=%.2f' % RHO['cifar10_test'], ha='right', va='bottom', fontsize=5.8, color=NAVY)
ax[0].legend(loc='upper left', frameon=False, markerscale=4, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.1, fontsize=5.2)
# (b) per-scale log-ratio AUROC and the AUROC of the exact ratio (four IDs)
ax[1].axvspan(SIG13[9] * 1.31, SIG13[12] * 1.6, color='k', alpha=0.06, lw=0)
for t in NLLRUN:
    c = IDCOL[t]; name = dict(IDS)[t]
    ax[1].plot(SIG13, PS[t], '-', lw=0.9, c=c, label=name)
    K90 = KGT.get('rot90', [])
    ax[1].plot(SIG13[K90], PS[t][K90], 'o', ms=2.6, c=c, mew=0)
    ax[1].plot([SIG13[k] for k in range(NK) if k not in K90], [PS[t][k] for k in range(NK) if k not in K90], 'o', ms=2.0, mfc='white', mec=c, mew=0.6)
    if PSD[t]['ode'] is not None:
        ax[1].axhline(PSD[t]['ode'], c=c, lw=0.8, ls='--')
        ax[1].text(SIG13[12] * 1.55, PSD[t]['ode'] + (1.2 if t == 'edmc10' else -1.2 if t == 'l6svhn' else 0), '%.1f' % PSD[t]['ode'], fontsize=4.6, color=c, ha='right', va='bottom' if t == 'edmc10' else 'top' if t == 'l6svhn' else 'center')
ax[1].axhline(50, c='k', lw=0.4, ls=':')
ax[1].set_xscale('log'); ax[1].set_xlim(SIG13[0] / 1.5, SIG13[12] * 1.6); ax[1].set_ylim(27, 110)
ax[1].set_xlabel('ノイズの大きさ $\\sigma_k$', fontsize=6.5, labelpad=1.5)
ax[1].set_ylabel('AUROC (%)', fontsize=6.5, labelpad=1.5)
ax[1].set_title('(b) 対数比の AUROC(スケール別)', fontsize=6.2, loc='left', pad=2)
ax[1].legend(loc='lower left', frameon=False, handletextpad=0.4, ncol=1, borderaxespad=0.1, labelspacing=0.1, fontsize=5.2)
ax[1].text(SIG13[12] * 1.5, 29, '用いない\n3 スケール', ha='right', va='bottom', fontsize=5.0, color='#333333')
ax[1].text(SIG13[0] / 1.3, 108.5, '●：用いる原子 %s($\\mathbf{R}_{90}$)  破線：厳密な $\\Lambda$' % ', '.join('$\\sigma_{%d}$' % (k + 1) for k in KGT.get('rot90', [])), ha='left', va='top', fontsize=4.4, color='#333333')   # the R180 atom is shown in Figure A1(b)
for a in ax: a.tick_params(labelsize=6.0)
fig.subplots_adjust(left=0.135, right=0.985, top=0.9, bottom=0.2)
fig.savefig(f'{OUT}/fig_verify.pdf', dpi=300); plt.close(fig)
print('rho', {k: round(float(v), 3) for k, v in RHO.items()})

# ================================================================ Figure A2: Euphonium vs the exact ratio, per image (four IDs)
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.65), gridspec_kw=dict(wspace=0.38))
for i, t in enumerate(NLLRUN):
    T = TIERS[t]; d = VER[t]['d']; name = dict(IDS)[t]; c = IDCOL[t]
    lam_o = np.concatenate([d[s]['lam'] for s in VER[t]['sets'][1:]]); G_o = np.concatenate([d[s]['G'] for s in VER[t]['sets'][1:]])
    rng = np.random.RandomState(0); so = rng.choice(len(lam_o), min(1500, len(lam_o)), replace=False)
    ax[i].scatter(lam_o[so], G_o[so], s=1.5, c='#c8c8c8', alpha=0.6, lw=0, rasterized=True, label='分布外(自然画像の集合)')
    ax[i].scatter(d[T['id']]['lam'], d[T['id']]['G'], s=1.5, c=(DARK if t == 'edmc100' else c), alpha=0.6, lw=0, rasterized=True, label='分布内')
    ax[i].axhline(0, c='k', lw=0.4, ls=':'); ax[i].axvline(0, c='k', lw=0.4, ls=':')
    la, lb = np.concatenate([lam_o, d[T['id']]['lam']]), np.concatenate([G_o, d[T['id']]['G']])
    ax[i].set_xlim(np.quantile(la, 0.003), np.quantile(la, 0.997)); ax[i].set_ylim(np.quantile(lb, 0.003), np.quantile(lb, 0.997))
    ax[i].set_title('%s のモデル' % name, fontsize=6.4, loc='left', pad=2)
    ax[i].set_xlabel('厳密な $\\Lambda$ (bits/dim)', fontsize=6.3, labelpad=1.5)
    if i == 0: ax[i].set_ylabel('Euphonium (%d 回)' % NFEG, fontsize=6.5, labelpad=1.5)
    ax[i].text(0.97, 0.03, '$\\rho$：分布内 %.2f\n分布内+分布外 %.2f' % (d[T['id']]['rho_G'], VER[t]['rho_pool_G']), ha='right', va='bottom', fontsize=5.8, transform=ax[i].transAxes)
    ax[i].tick_params(labelsize=6.0)
    if i == 0: ax[i].legend(loc='upper left', frameon=False, markerscale=4, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.15, fontsize=5.4)
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.2)
fig.savefig(f'{OUT}/fig_gode.pdf', dpi=300); plt.close(fig)

# ================================================================ Figure A1: inside Euphonium (per-scale log-ratios and the distribution of the score)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.6), gridspec_kw=dict(width_ratios=[1.05, 1.05, 1], wspace=0.42))
ax = axs[1:]
for s, lab, c in SETS:
    sg, (q1, q2, q3) = INTEG[s]
    axs[0].plot(sg, q2, '-o', ms=1.8, lw=1, c=c, label=lab)
    axs[0].fill_between(sg, q1, q3, color=c, alpha=0.15, lw=0)
    if s == 'cifar10_test': axs[0].fill_between(sg, 0, q2, color=c, alpha=0.12, lw=0, hatch='////', edgecolor=c)
axs[0].axhline(0, c='k', lw=0.4, ls=':'); axs[0].set_xscale('log')
axs[0].set_xlabel('ノイズの大きさ $\\sigma$', fontsize=6.5, labelpad=1.5)
axs[0].set_ylabel('$[e_\\sigma(\\mathbf{R}_{90}x)-e_\\sigma(x)]\\,/\\,(D\\ln 2)$', fontsize=6.3, labelpad=1.5)
axs[0].set_title('(a) 式 (5) の被積分関数(中央値,四分位範囲)', fontsize=6.2, loc='left', pad=2)
axs[0].legend(loc='upper right', frameon=False, handletextpad=0.3, ncol=1, borderaxespad=0.1, labelspacing=0.15, fontsize=5.6)
axs[0].text(0.011, 0.0045, '斜線の面積 = 分布内の\n画像の $\\Lambda$ (中央値)', fontsize=5.5, color=NAVY, ha='left', va='bottom')
axs[0].tick_params(labelsize=6.0)
GS = {}
RUN0 = RUN[NOISE]['edmc10']
TLS = {'rot90': '-', 'rot180': '--', 'rot270': ':'}; TMK = {'rot90': 'o', 'rot180': 's', 'rot270': '^'}
for s, lab, c in SETS:
    F = FT(RUN0, s); i0 = F['tfs'].index('id'); sg = F['sigmas'][:10]
    for T in TG:
        j = F['tfs'].index(T)
        lr = np.log(F['e'][:, :10, 0, j] / F['e'][:, :10, 0, i0])            # per-scale log-ratio log(e_k(T x)/e_k(x))
        q1, q2, q3 = np.quantile(lr, [0.25, 0.5, 0.75], axis=0)
        first = (T == TG[0])
        ax[0].plot(sg, q2, TLS[T], lw=0.9, c=c, label=lab if first else None)
        if first: ax[0].fill_between(sg, q1, q3, color=c, alpha=0.13, lw=0)
        ks = KGT[T]; ax[0].plot(sg[ks], q2[ks], TMK[T], ms=3.0, c=c, mew=0)        # the atoms used
    GS[s] = escore(F)
TLAB = {'rot90': '$\\mathbf{T}=\\mathbf{R}_{90}$', 'rot180': '$\\mathbf{T}=\\mathbf{R}_{180}$', 'rot270': '$\\mathbf{T}=\\mathbf{R}_{270}$'}
for T in TG: ax[0].plot([], [], TLS[T], marker=TMK[T], ms=3.0, c='k', lw=0.9, label=TLAB[T] + '(印は用いる原子)')
ax[0].axhline(0, c='k', lw=0.4, ls=':'); ax[0].set_xscale('log')
ax[0].set_xlabel('ノイズの大きさ $\\sigma_k$', fontsize=6.5, labelpad=1.5)
ax[0].set_ylabel('$\\log\\,\\hat e_k(\\mathbf{T}x)\\,/\\,\\hat e_k(x)$', fontsize=6.3, labelpad=1.5)
ax[0].set_title('(b) Euphonium のスケールごとの対数比(中央値)', fontsize=6.2, loc='left', pad=2)
ax[0].legend(loc='upper left', frameon=False, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.12, fontsize=4.8, ncol=2, columnspacing=0.5)
ax[0].set_ylim(-0.03, 0.22)
gmax = float(np.quantile(GS['cifar10_test'], 0.995)); bins = np.linspace(-0.5 * gmax, gmax, 56)
for s, lab, c in SETS:
    ax[1].hist(GS[s], bins=bins, histtype='step', color=c, lw=0.9, density=True)
ax[1].axvline(0, c='k', lw=0.4, ls=':')
ax[1].set_xlabel('Euphonium (%d 個の対数比の和)' % len(ATOMS), fontsize=6.5, labelpad=1.5); ax[1].set_ylabel('密度', fontsize=6.5, labelpad=1.5)
ax[1].set_title('(c) Euphonium の分布', fontsize=6.2, loc='left', pad=2)
ax[1].set_xlim(-0.5 * gmax, gmax)
ax[1].text(0.98, 0.97, '中央値\n分布内 %.2f\nSVHN %.3f' % (np.median(GS['cifar10_test']), np.median(GS['svhn_test'])), ha='right', va='top', fontsize=5.6, transform=ax[1].transAxes)
for a in ax: a.tick_params(labelsize=6.0)
fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.2)
fig.savefig(f'{OUT}/fig_G.pdf', dpi=300); plt.close(fig)
M['GmedID'] = float(np.median(GS['cifar10_test'])); M['GmedSVHN'] = float(np.median(GS['svhn_test'])); M['GposID'] = 100 * float((GS['cifar10_test'] > 0).mean())
M['nTerms'] = len(ATOMS)
print('G medians', {s: round(float(np.median(v)), 3) for s, v in GS.items()})

# ---- example values of Figure 2 (fig_shared.tex): the median dog of Figure 1 (CIFAR-10 test 3535) and an SVHN 8 (test 4304), at the coarsest atom of the rule
TSH, KSH = max(ATOMS, key=lambda a: a[1])        # coarsest atom (the example of Figure 2)
for key, s, i in (('Dog', 'cifar10_test', 3535), ('Svhn', 'svhn_test', 4304)):
    F = FT(RUN0, s); pos = F['idx'].tolist().index(i); i0, j = F['tfs'].index('id'), F['tfs'].index(TSH)
    M['shE' + key] = F['e'][pos, KSH, 0, i0] / F['dim']; M['shET' + key] = F['e'][pos, KSH, 0, j] / F['dim']
    M['shLR' + key] = float(np.log(F['e'][pos, KSH, 0, j] / F['e'][pos, KSH, 0, i0])); M['shG' + key] = float(escore(F)[pos])
M['shSigma'] = float(F['sigmas'][KSH]); M['shRotText'] = {'rot180': '\\Rot{180}', 'rot90': '\\Rot{90}', 'rot270': '\\Rot{270}'}[TSH]
print('fig_shared examples', {k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in M.items() if k.startswith('sh')})
# ---- check of the symmetry argument: rank correlation of the R270 and R90 (and R180 and R90) log-ratio sums (10 scales) on the CIFAR-10 development images (B_edmc10)
Fb = feat('B_edmc10', 'cifar10_test'); eb = Fb['e'][:, :10, 0, :]; ib = lambda t: Fb['tfs'].index(t)
gsum = lambda a: np.log(eb[:, :, ib(a)] / eb[:, :, ib('id')]).sum(1)
M['rhoTwoSeventy'] = float(spearmanr(gsum('rot270'), gsum('rot90'))[0]); M['rhoOneEighty'] = float(spearmanr(gsum('rot180'), gsum('rot90'))[0])

# ================================================================ tables: comparison with the eight methods (same models, same images)
ORDER = ['svhn_test', 'textures', 'celeba_test', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val', 'cifar100_test']
BASE = [('NLL', 'NLL(尤度)'), ('IC-png', 'IC (PNG)'), ('IC-j2k', 'IC (J2K)'), ('TT', 'TT'), ('MSMA*', 'MSMA 型'), ('DiffPath', 'DiffPath'), ('GEPC', 'GEPC'), ('EigenScore', 'EigenScore')]
BNAME = dict(BASE)
def val(t, m, s):
    """AUROC (%) of method m on pair (t, s), or None.  G: Euphonium, G90: R90 on 10 scales (pair, 40 passes; natural and upside-down sets), ODE: exact ratio (PF-ODE)"""
    if m == 'G': return GA.get((MAINID, t, s))
    if m == 'G90': return val_g90(t, s)
    if m == 'ODE': return val_ode(t, s)
    v = F4[t]['common'].get(m, {}).get(s); return None if v is None else 100 * v['auroc']
def fmt(v, sota, ours=False, plain=False, sota2=None):
    """cell of a comparison method: dagger = SOTA (best of the eight methods).  Cell of Ours: light-blue background and bold when it ties or beats the best
    of the eight (BEST), underline when only the best of the eight is higher (second overall); within 0.05 points is a tie; sota2 = second best of the eight.
    Only Euphonium is called with ours=True; Ablations (40-pass estimate, exact ratio, other configurations) are not marked (plain=True)"""
    if v is None: return '--'
    if plain: return '%.1f' % v
    if ours:
        if v >= sota - 0.05: return '\\cellcolor{firstblue}\\textbf{%.1f}' % v
        if sota2 is not None and v >= sota2 - 0.05: return '\\underline{%.1f}' % v
        return '%.1f' % v
    return ('%.1f$^{\\dagger}$' % v) if (sota is not None and abs(v - sota) < 0.05) else ('%.1f' % v)
RT = {'rot90': '$\\Rot{90}$', 'rot180': '$\\Rot{180}$', 'rot270': '$\\Rot{270}$'}
M['ruleRotText'] = ' と '.join(RT[t] for t in TG) if len(TG) <= 2 else ', '.join(RT[t] for t in TG)
M['ruleNscale'] = len(KG); M['ruleNoiseText'] = '符号反転対' if NOISE == 'pair' else '$+\\veps$ のみ'
M['ruleScaleText'] = ', '.join('%.3g' % s for s in F['sigmas'][KG])
M['ruleAtomsText'] = '\\{' + ',\\,'.join('(%s,%d)' % (RT[T].strip('$'), k + 1) for T, k in ATOMS) + '\\}'       # the atom set A of Eq. (17)
M['ruleAtomsWords'] = '；'.join('%s は %s' % (RT[T], ', '.join('$\\sigma_{%d}$' % (k + 1) for k in KGT[T])) for T in TG)   # atoms in words for the text, e.g. R90: sigma_1, sigma_7; R180: sigma_1
M['ruleKsetText'] = '\\{' + ','.join('%d' % (k + 1) for k in KG) + '\\}'
M['ruleTsetText'] = '\\{' + ','.join(RT[t].strip('$') for t in TG) + '\\}'
M['ruleCand'] = MAINID
M['ruleSumText'] = '\\sum_{(\\mT,k)\\in\\mathcal A}\\log\\frac{\\hat e_k(\\mT\\vec{x})}{\\hat e_k(\\vec{x})}'
M['ruleOrientN'] = 1 + len(TG); M['ruleNatoms'] = len(ATOMS)
M['ruleNfeUp'] = 2 * len(KG); M['ruleNfeRot'] = 2 * len(ATOMS)
GLAB = 'Euphonium(%s, %d 回)' % ('，'.join('%s：%s' % (RT[T], ','.join('$\\sigma_{%d}$' % (k + 1) for k in KGT[T])) for T in TG), NFEG)

# ---- Table 2 (left): the nine natural OOD sets of CIFAR-10 (method x set)
cols = ORDER + ['mean']
S = {m: {s: val('edmc10', m, s) for s in ORDER} for m in [b for b, _ in BASE] + ['G', 'ODE']}
for m in S: S[m]['mean'] = np.mean([S[m][s] for s in ORDER])
sota = {c: max(S[b][c] for b, _ in BASE) for c in cols}
sota2 = {c: sorted((S[b][c] for b, _ in BASE), reverse=True)[1] for c in cols}
L = [lab + ' & ' + ' & '.join(fmt(S[b][c], sota[c]) for c in cols) + '\\\\' for b, lab in BASE]
L.append('\\midrule')
L.append('Ours：' + GLAB + ' & ' + ' & '.join(fmt(S['G'][c], sota[c], ours=True, sota2=sota2[c]) for c in cols) + '\\\\')
L.append('Ablation：厳密な比 $\\Lambda_{90}$(PF-ODE, 512 回) & ' + ' & '.join(fmt(S['ODE'][c], sota[c], plain=True) for c in cols) + '\\\\\\bottomrule')
open(os.path.join(PAPER, 'tab_cifar10.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: AUROC (%) on the CIFAR-10 model, final-split images\n' + '\n'.join(L) + '\n')
M['tabMeanNLL'] = S['NLL']['mean']; M['tabMeanG'] = S['G']['mean']; M['tabMeanODE'] = S['ODE']['mean']
M['tabMeanGninety'] = np.mean([val('edmc10', 'G90', s) for s in ORDER])   # mean over the nine sets of the 40-pass shared-noise estimate (R90, 10 scales)
M['tabMinG'] = min(S['G'][s] for s in ORDER); M['GSVHN'] = S['G']['svhn_test']; M['odeSVHN'] = S['ODE']['svhn_test']; M['GCh'] = S['G']['cifar100_test']
M['tabMinGName'] = sn(min(ORDER, key=lambda s: S['G'][s]))
bb = max(BASE, key=lambda x: S[x[0]]['mean']); M['bestBaseCten'] = S[bb[0]]['mean']; M['bestBaseNameCten'] = BNAME[bb[0]]
M['winsCten'] = sum(1 for s in ORDER if S['G'][s] >= max(S[b][s] for b, _ in BASE) - 0.05)
M['secondCten'] = sum(1 for s in ORDER if S['G'][s] < sota[s] - 0.05 and S['G'][s] >= sota2[s] - 0.05)
M['tabNbelow'] = sum(1 for s in ORDER if S['NLL'][s] < 60); M['dpSVHN'] = S['DiffPath']['svhn_test']; M['gepcMeanCten'] = S['GEPC']['mean']
M['icjSVHN'] = S['IC-j2k']['svhn_test']; M['GrankSVHN'] = 1 + sum(1 for b, _ in BASE if S[b]['svhn_test'] > S['G']['svhn_test'] + 0.05)   # rank of Euphonium among the eight methods on CIFAR-10 vs SVHN
M['minBaseCten'] = min(max(S[b][s] for b, _ in BASE) for s in ORDER)
lose = [s for s in ORDER if S['G'][s] < max(S[b][s] for b, _ in BASE) - 0.05]
M['loseCtenList'] = ', '.join(sn(s) for s in lose) if lose else 'なし'
M['loseCtenDetail'] = '；'.join('%sでは%sの%.1f\\%%に対し Euphonium は%.1f\\%%' % (sn(s), BNAME[max(BASE, key=lambda x: S[x[0]][s])[0]], max(S[b][s] for b, _ in BASE), S['G'][s]) for s in lose) if lose else ''
print('CIFAR-10 means', {m: round(S[m]['mean'], 1) for m in S}, 'wins', M['winsCten'], 'lose', lose)

# ---- Table 2 (right): mean over the natural OOD sets of every ID (method x ID); tab_main.tex holds both halves
T = {}
for t, name in IDS:
    T[t] = {m: np.mean([val(t, m, s) for s in NAT[t]]) if all(val(t, m, s) is not None for s in NAT[t]) else None for m in [b for b, _ in BASE] + ['G', 'ODE']}
    T[t]['n'] = len(NAT[t])
sota = {t: max(T[t][b] for b, _ in BASE) for t, _ in IDS}
sota2 = {t: sorted((T[t][b] for b, _ in BASE), reverse=True)[1] for t, _ in IDS}
L = [BNAME[b] + ' & ' + ' & '.join(fmt(T[t][b], sota[t]) for t, _ in IDS) + '\\\\' for b, _ in BASE]
L.append('\\midrule')
L.append('Ours：Euphonium(%d 回) & ' % NFEG + ' & '.join(fmt(T[t]['G'], sota[t], ours=True, sota2=sota2[t]) for t, _ in IDS) + '\\\\')
L.append('Ablation：厳密な比 $\\Lambda_{90}$(512 回) & ' + ' & '.join(fmt(T[t]['ODE'], sota[t], plain=True) for t, _ in IDS) + '\\\\\\bottomrule')
open(os.path.join(PAPER, 'tab_ids.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: mean AUROC (%) over the natural OOD sets of every ID\n' + '\n'.join(L) + '\n')
# combined table: left the nine CIFAR-10 sets and their mean, right the other five IDs
sotaC = {c: max(S[b][c] for b, _ in BASE) for c in cols}; sotaC2 = {c: sorted((S[b][c] for b, _ in BASE), reverse=True)[1] for c in cols}
OTH = [t for t, _ in IDS if t != 'edmc10']
L = []
for b, lab in BASE:
    L.append(lab + ' & ' + ' & '.join(fmt(S[b][c], sotaC[c]) for c in cols) + ' & ' + ' & '.join(fmt(T[t][b], sota[t]) for t in OTH) + '\\\\')
L.append('\\midrule')
L.append('Ours：Euphonium(%d 回) & ' % NFEG + ' & '.join(fmt(S['G'][c], sotaC[c], ours=True, sota2=sotaC2[c]) for c in cols) + ' & ' + ' & '.join(fmt(T[t]['G'], sota[t], ours=True, sota2=sota2[t]) for t in OTH) + '\\\\')
L.append('Ablation：厳密な比 $\\Lambda_{90}$(512 回) & ' + ' & '.join(fmt(S['ODE'][c], sotaC[c], plain=True) for c in cols) + ' & ' + ' & '.join(fmt(T[t]['ODE'], sota[t], plain=True) for t in OTH) + '\\\\\\bottomrule')
open(os.path.join(PAPER, 'tab_main.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: left, AUROC (%) on the nine CIFAR-10 sets and their mean; right, mean AUROC over the natural OOD sets of the other five IDs\n' + '\n'.join(L) + '\n')
for t, name in IDS:
    k = KEYID[t]; M['oN' + k] = T[t]['n']; M['oNLL' + k] = T[t]['NLL']; M['oG' + k] = T[t]['G']
    if T[t]['ODE'] is not None: M['oODE' + k] = T[t]['ODE']
    bb = max(BASE, key=lambda x: T[t][x[0]]); M['oBest' + k] = T[t][bb[0]]; M['oBestName' + k] = BNAME[bb[0]]
    worst = min(NAT[t], key=lambda s: val(t, 'G', s)); M['oWorstName' + k] = sn(worst); M['oWorstG' + k] = val(t, 'G', worst)
    if val(t, 'ODE', worst) is not None: M['oWorstODE' + k] = val(t, 'ODE', worst)
    assert abs(T[t]['G'] - PSD[t]['five']) < 0.05, (t, T[t]['G'], PSD[t]['five'])     # same images and atoms as the per-scale table: the two computations must agree
M['nIDbest'] = sum(1 for t, _ in IDS if T[t]['G'] >= sota[t] - 0.05)
M['nIDsecond'] = sum(1 for t, _ in IDS if T[t]['G'] < sota[t] - 0.05 and T[t]['G'] >= sota2[t] - 0.05)
M['idLoseList'] = ', '.join(name for t, name in IDS if T[t]['G'] < sota[t] - 0.05) or 'なし'
M['idLoseDetail'] = '，'.join('%sでは%sの%.1f\\%%' % (name, M['oBestName' + KEYID[t]], sota[t]) for t, name in IDS if T[t]['G'] < sota[t] - 0.05)
print('per-ID', {name: {m: (round(v, 1) if isinstance(v, float) else v) for m, v in T[t].items()} for t, name in IDS})

# ---- Tables A10, A11: all 63 pairs (six IDs x every OOD set)
rows = []
for t, name in IDS:
    for s in TIERS[t]['ood']:
        vals = [val(t, b, s) for b, _ in BASE] + [val(t, 'G', s), val(t, 'G90', s), val(t, 'ODE', s)]
        rows.append((name, sn(s), vals))
def line(name, s, vals):
    bv = sorted((v for v in vals[:8] if v is not None), reverse=True); sota = bv[0]; sota2 = bv[1] if len(bv) > 1 else None
    cells = [fmt(v, sota) for v in vals[:8]] + [fmt(vals[8], sota, ours=True, sota2=sota2)] + [fmt(v, sota, plain=True) for v in vals[9:]]
    return '%s & %s & ' % (name, s) + ' & '.join(cells) + '\\\\'
for part, names in (('a', ['CIFAR-10', 'SVHN', 'CelebA']), ('b', ['CIFAR-100', 'ImageNet-64', 'FashionMNIST'])):
    L = []; cur = None
    for name, s, vals in rows:
        if name not in names: continue
        if cur is not None and name != cur: L.append('\\midrule')
        L.append(line(name if name != cur else '', s, vals)); cur = name
    L[-1] += '\\bottomrule'
    open(os.path.join(PAPER, 'supp_all_%s.tex' % part), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: AUROC (%) on all 63 pairs; Euphonium in light-blue bold where it ties or beats the best of the eight methods\n' + '\n'.join(L) + '\n')
M['nPairs'] = len(rows); M['winsAll'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] >= max(x for x in v[:8] if x is not None) - 0.05)
M['secondAll'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] < max(x for x in v[:8] if x is not None) - 0.05 and v[8] >= sorted((x for x in v[:8] if x is not None), reverse=True)[1] - 0.05)
M['bestAll'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] >= max(x for x in v if x is not None) - 0.05)
M['winsAllStrict'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] > max(x for x in v[:8] if x is not None) + 0.05)
M['winsAllOne'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] >= max(x for x in v[:8] if x is not None) - 1.0)
M['loseAllOne'] = sum(1 for _, _, v in rows if v[8] is not None and v[8] < max(x for x in v[:8] if x is not None) - 1.0)
M['meanSixG'] = np.mean([T[t]['G'] for t, _ in IDS]); M['meanSixBest'] = max(np.mean([T[t][b] for t, _ in IDS]) for b, _ in BASE)
M['meanSixBestName'] = BNAME[max(BASE, key=lambda x: np.mean([T[t][x[0]] for t, _ in IDS]))[0]]
vf = {t: GA[(MAINID, t, [s for s in TIERS[t]['ood'] if s.endswith('__vflip')][0])] for t, _ in IDS}
vfb = {t: max(val(t, b, [s for s in TIERS[t]['ood'] if s.endswith('__vflip')][0]) for b, _ in BASE) for t, _ in IDS}
M['vflipGlo'], M['vflipGhi'], M['vflipBaseLo'], M['vflipBaseHi'] = min(vf.values()), max(vf.values()), min(vfb.values()), max(vfb.values())
M['vflipWins'] = sum(1 for t in vf if vf[t] >= vfb[t] - 0.05)
# reversed pairs (natural and upside-down sets with an AUROC below 49%)
REV = [(dict(IDS)[t], sn(s), GA[(MAINID, t, s)], val(t, 'ODE', s)) for t, _ in IDS for s in TIERS[t]['ood'] if 'synth' not in s and GA[(MAINID, t, s)] < 49]
M['nRev'] = len(REV); M['revText'] = ', '.join('%sに対する%s(%.1f\\%%)' % (a, b, g) for a, b, g, o in REV)
print('all pairs', M['nPairs'], 'wins', M['winsAll'], M['winsAllStrict'], 'six-ID mean G %.1f best base %s %.1f' % (M['meanSixG'], M['meanSixBestName'], M['meanSixBest']), 'vflip G', {t: round(v, 1) for t, v in vf.items()}, 'vflip best base', {t: round(v, 1) for t, v in vfb.items()}, 'reversed', REV)

# ---- Table A8: orientation strength of every ID (mean of Euphonium on the ID images) and the lowest pair
L = []
for t, name in IDS:
    g0 = gscore(t, '__id__'); nat = [s for s in TIERS[t]['ood'] if 'synth' not in s]
    rev = sum(1 for s in nat if val(t, 'G', s) < 49)
    worst = min(nat, key=lambda s: val(t, 'G', s)); wo = val(t, 'ODE', worst)
    L.append('%s & %.2f $\\pm$ %.2f & %d/%d & %s & %.1f & %s\\\\' % (name, np.mean(g0), np.std(g0), rev, len(nat), sn(worst), val(t, 'G', worst), '--' if wo is None else '%.1f' % wo))
    M['gbar' + KEYID[t]] = float(np.mean(g0)); M['gsd' + KEYID[t]] = float(np.std(g0))
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'supp_orient.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: mean and sd of Euphonium on the ID images, number of reversed pairs, lowest pair\n' + '\n'.join(L) + '\n')

# ---- Table A3: the candidates (selection data and main evaluation)
NOISET = {'pair': '対', 'single': '単'}
def kset(ks):
    if ks == ALL: return '10 点すべて'
    if ks == ODD: return '1 つおき(奇)'
    if ks == EVEN: return '1 つおき(偶)'
    if ks == [0, 2, 3, 5, 7]: return '$\\{\\sigma_1,\\sigma_3,\\sigma_4,\\sigma_6,\\sigma_8\\}$'
    return '$\\{' + ','.join('\\sigma_{%d}' % (k + 1) for k in ks) + '\\}$'
# selection-data values of the pure-pair candidates, computed directly from the selection data (ID development images vs extra OOD sets)
RGBPOOL = ['stl10_test', 'gtsrb_test', 'flowers102_test', 'eurosat', 'mnist_rgb32', 'pets_test', 'caltech101', 'food101_test', 'aircraft_test', 'imagenette_val', 'sst2_test']
POOL = {'edmc10': RGBPOOL, 'l6svhn': RGBPOOL, 'dpceleba': RGBPOOL, 'edmc100': RGBPOOL,
        'in64': ['in64_' + s for s in RGBPOOL if s not in ('imagenette_val', 'mnist_rgb32')] + ['in64_mnist_rgb'],
        'fmnist': ['g32_emnist_letters', 'g32_k49', 'g32_semeion', 'g32_cifar10_gray', 'g32_svhn_gray', 'g32_gtsrb_gray']}
def sel_pool(atoms):
    """six-ID selection-data AUROC of a pure-pair atom set: ID development images (DEVRUN) vs the extra OOD sets (F_ runs)"""
    per = {}
    for t in TIERS:
        g0 = atom_score(FT(DEVRUN[t], TIERS[t]['id']), atoms)
        per[t] = {s: 100 * auroc(g0, atom_score(FT(RUN['pair'][t], s), atoms)) for s in POOL[t]}
    return per
SEL2 = {cid: sel_pool(atoms) for cid, atoms, noise, nfe, _, _, _ in CAND if cid.startswith('e')}
# the same computation reproduces the grid-candidate file (grid_selection.json) for c3
chk = sel_pool(atoms_of(('rot90',), ODD)); assert all(abs(np.mean(list(chk[t].values())) - np.mean(list(SEL['c3']['per'][t].values()))) < 0.05 for t in TIERS), 'c3: selection values differ from grid_selection.json'
L = []
for cid, atoms, noise, nfe, rot_text, scale_text, ks in sorted(CAND, key=lambda c: 0 if c[0].startswith('e') else 1):
    if cid.startswith('e'): per_sel = SEL2[cid]
    else: per_sel = SEL.get(cid, {}).get('per', {}) if cid.startswith('c') else SEL.get({'r60': 'R2x10pair', 'r80': 'R3x10pair'}.get(cid, cid), {}).get('per', {})
    sel = [np.mean(list(per_sel[t].values())) for t in TIERS if per_sel.get(t)] if per_sel else []
    selm = '%.1f' % np.mean(sel) if len(sel) == 6 else '--'
    if all((cid, t, s) in GA for t in TIERS for s in TIERS[t]['ood']):
        per = [np.mean([GA[(cid, t, s)] for s in NAT[t]]) for t, _ in IDS]
        wins = sum(1 for t, _ in IDS for s in TIERS[t]['ood'] if GA[(cid, t, s)] >= max(val(t, b, s) for b, _ in BASE if val(t, b, s) is not None) - 0.05)
        mainm, winss = '%.1f' % np.mean(per), '%d' % wins
    else: mainm, winss = '--', '--'
    lab = ('\\textbf{Ours}(%s)' % cid) if cid == MAINID else (('参考(%s)' % cid) if cid.startswith('e') else ('Ablation(%s)' % cid))
    row = [lab, rot_text, scale_text if scale_text is not None else kset(ks), NOISET[noise], '%d' % nfe, selm, mainm, winss]
    if cid == MAINID: row = ['\\cellcolor{firstblue}' + c for c in row]
    if cid == 'c1': L.append('\\midrule')
    L.append(' & '.join(row) + '\\\\')
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'supp_select.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: Euphonium candidates (PROTOCOL_10NFE.md, top) and the grid candidates (Ablation, PROTOCOL_20NFE.md, bottom): mean AUROC (%) on the selection data and on the main evaluation, pairs won\n' + '\n'.join(L) + '\n')
sel_main = [np.mean(list(SEL2[MAINID][t].values())) for t in TIERS]
M['selMean'] = float(np.mean(sel_main)); M['nfeG'] = NFEG; M['odeOverG'] = 512 / NFEG; M['nCand'] = len([c for c in CAND if c[0].startswith('c')])
M['selMeanPrev'] = float(np.mean([np.mean(list(SEL['c3']['per'][t].values())) for t in TIERS]))
for t in TIERS: M['selPrev' + KEYID[t]] = float(np.mean(list(SEL['c3']['per'][t].values()))); M['selNew' + KEYID[t]] = float(np.mean(list(SEL2[MAINID][t].values())))
# main evaluation of the best grid candidate c3 (Ablation; referred to in the text)
M['prevMainMean'] = float(np.mean([np.mean([GA[('c3', t, s)] for s in NAT[t]]) for t, _ in IDS]))
M['prevWins'] = sum(1 for t, _ in IDS for s in TIERS[t]['ood'] if GA[('c3', t, s)] >= max(val(t, b, s) for b, _ in BASE if val(t, b, s) is not None) - 0.05)
for t, _ in IDS: M['prevG' + KEYID[t]] = float(np.mean([GA[('c3', t, s)] for s in NAT[t]]))
M['prevCtenMean'] = float(np.mean([GA[('c3', 'edmc10', s)] for s in ORDER]))
M['prevChCten'] = GA[('c3', 'edmc100', 'cifar10_test')]; M['prevINCel'] = GA[('c3', 'in64', 'in64_celeba_test')]; M['prevCtenCh'] = GA[('c3', 'edmc10', 'cifar100_test')]
M['prevSVHNmean'] = float(np.mean([GA[('c3', 'l6svhn', s)] for s in NAT['l6svhn']]))
# selection rule: largest six-ID mean on the selection data; within 0.1 points tied -> fewest passes, then the largest worst-ID mean (PROTOCOL_10NFE.md)
def sel2_mean(c): return float(np.mean([np.mean(list(SEL2[c][t].values())) for t in TIERS]))
def sel2_min(c): return float(min(np.mean(list(SEL2[c][t].values())) for t in TIERS))
E2 = [c[0] for c in CAND if c[0].startswith('e')]; top = max(sel2_mean(c) for c in E2)
tied = [c for c in E2 if sel2_mean(c) >= top - 0.1]; nfe_of_c = {c[0]: c[3] for c in CAND}
best_sel = min(tied, key=lambda c: (nfe_of_c[c], -sel2_min(c)))
print('selection (Euphonium candidates)', {c: (round(sel2_mean(c), 2), round(sel2_min(c), 1), nfe_of_c[c]) for c in E2}, 'tied', tied, 'best', best_sel, 'main', MAINID); assert best_sel == MAINID, 'the best candidate on the selection data is not the rule of rule.json'
M['selMinNew'] = sel2_min(MAINID); M['selMinSecond'] = sel2_min('e10b'); M['selSecond'] = sel2_mean('e10b')
M['secondMainMean'] = float(np.mean([np.mean([GA[('e10b', t, s)] for s in NAT[t]]) for t, _ in IDS]))
M['secondWins'] = sum(1 for t, _ in IDS for s in TIERS[t]['ood'] if GA[('e10b', t, s)] >= max(val(t, b, s) for b, _ in BASE if val(t, b, s) is not None) - 0.05)
# the three tied candidates with two decimals (the text shows that they are within 0.1 points) and the main evaluation of the third, e10c
M['selMeanTwo'] = '%.2f' % sel2_mean(MAINID); M['selSecondTwo'] = '%.2f' % sel2_mean('e10b'); M['selThirdTwo'] = '%.2f' % sel2_mean('e10c'); M['selMinThird'] = sel2_min('e10c')
M['thirdMainMean'] = float(np.mean([np.mean([GA[('e10c', t, s)] for s in NAT[t]]) for t, _ in IDS]))
M['thirdWins'] = sum(1 for t, _ in IDS for s in TIERS[t]['ood'] if GA[('e10c', t, s)] >= max(val(t, b, s) for b, _ in BASE if val(t, b, s) is not None) - 0.05)
M['selEight'] = sel2_mean('e8')
for cid, word in (('e8', 'Eight'), ('e6', 'Six'), ('e4', 'Four')):
    M['budgetMain' + word] = float(np.mean([np.mean([GA[(cid, t, s)] for s in NAT[t]]) for t, _ in IDS]))
    M['budgetSel' + word] = sel2_mean(cid)
# the same search in the difference form (first approximation of Eq. (15)) and single-noise candidates (analysis/ablations/frontier.py; the pair results agree with the computation above)
FR = json.load(open(os.path.join(RESULTS, 'ablations', 'frontier.json'), encoding='utf-8'))
assert abs(FR['A']['logr_nfe10']['sel'] - sel2_mean(MAINID)) < 0.02 and FR['A']['logr_nfe10']['cand'] == 'R90{1,7} + R180{1}', FR['A']['logr_nfe10']
for b in ('8', '6', '4'):
    assert abs(FR['A']['logr_nfe' + b]['sel'] - M['budgetSel' + {'8': 'Eight', '6': 'Six', '4': 'Four'}[b]]) < 0.02, (b, FR['A']['logr_nfe' + b])
def cand_words(txt):
    out = []
    for part in txt.split(' + '):
        T, ks = part.split('{'); ks = ks.rstrip('}').split(',')
        out.append('%s：%s' % ({'R90': '$\\Rot{90}$', 'R180': '$\\Rot{180}$', 'R270': '$\\Rot{270}$'}[T], ', '.join('$\\sigma_{%s}$' % k for k in ks)))
    return '；'.join(out)
d = FR['A']['diff_nfe10']; M['selDiffBest'] = d['sel']; M['selDiffBestMin'] = d['sel_min']; M['diffBestCand'] = cand_words(d['cand']); M['diffBestMain'] = d['main_mean']; M['diffBestWins'] = int(d['wins']); M['diffBestSVHN'] = d['main']['l6svhn']; M['diffBestCh'] = d['main']['edmc100']
d = FR['D']['diff']; M['dEuMain'] = d['main_mean']; M['dEuWins'] = int(d['wins']); M['dEuIDbest'] = int(d['nIDbest']); M['dEuSVHN'] = d['main']['l6svhn']
M['coarseTopLogr'] = len(FR['B']['logr']['coarse_in_top10']); M['coarseTopDiff'] = len(FR['B']['diff']['coarse_in_top10'])
if 'logr_single_nfe5' in FR['C']:
    d = FR['C']['logr_single_nfe5']; M['singleFiveCand'] = cand_words(d['cand']); M['singleFiveNfe'] = int(d['nfe']); M['singleFiveSel'] = d['sel']; M['singleFiveMain'] = d['main_mean']; M['singleFiveWins'] = int(d['wins'])
    d = FR['C']['logr_pair_e10_on_common']; M['pairOnCommonSel'] = d['sel']; M['pairOnCommonMain'] = d['main_mean']
# feature kNN of the same model (experiment of the companion final-presentation resume; results/external/knn_feature.json)
KN = json.load(open(os.path.join(RESULTS, 'external', 'knn_feature.json'), encoding='utf-8'))
M['knnCten'] = KN['edmc10']['mean_nat']['C']; M['knnMeanSix'] = float(np.mean([KN[t]['mean_nat']['C'] for t, _ in IDS])); M['knnLoseN'] = sum(1 for t, _ in IDS if KN[t]['mean_nat']['C'] > M['oG' + KEYID[t]] + 0.05)
# number of candidates: 1-5 atoms from 3 rotations x 10 scales with <= 10 passes (1285, PROTOCOL_10NFE.md)
import itertools as _it
M['nCandTwo'] = sum(1 for r in range(1, 6) for c in _it.combinations([(T, k) for T in ('rot90', 'rot180', 'rot270') for k in range(10)], r) if nfe_atoms(list(c)) <= 10)
print('stage-2 candidates', M['nCandTwo'], 'sel two-decimals', M['selMeanTwo'], M['selSecondTwo'], M['selThirdTwo'])

# ================================================================ sign or magnitude: statistics from the same 10 passes, and the R90 prediction difference of the GEPC run
#   G        : signed (Euphonium)                                MabsG : -|G| (magnitude only; larger = OOD, the reading of GEPC)
#   Mabs     : -sum |log e_k(T x)/e_k(x)| (magnitude per atom)      Mtwo  : -|G - median on the development images| (two-sided)
#   GEPCr    : the GEPC run (same final-split images; rotated noisy images = shared noise): R90-only prediction difference ||a-b||^2/||b||^2 at its six fine levels (mean pooling): -sum_l log(...)
#   GP       : prediction difference ||T^-1 epshat(T x)-epshat(x)||^2 of the atoms from the same 10 passes (only if inputs/eu/ holds the runs of experiments/jobs/GP.txt): -sum log(g / a)
def gepc_r90(t, s):
    p = f'{INPUTS}/base2/gepc_{t}/{s}.npz'
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); g = [str(x) for x in z['gepc_g']].index('rot90'); j = [float(x) for x in z['pools']].index(0.0)
    m = z['num'][:, :, g, j] / z['den'][:, :, j]
    return dict(idx=z['idx'], s=-np.log(m).sum(1))
def gp_run(t, s):
    """prediction differences of Euphonium's atoms from the same 10 passes (experiments/score_euphonium.py --pred_diff, job list
    experiments/jobs/GP.txt; exported to inputs/eu/<tag>/<set>.npz): -sum log(pd / a_up) and -sum pd / a_up over the atoms"""
    p = f'{INPUTS}/eu/{t}/{s}.npz'
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); ks = [int(k) for k in z['scales']]
    a = np.stack([z['a_up'][:, ks.index(int(str(c).split(':')[1]))] for c in z['atoms']], 1).astype(np.float64); g = z['pd'].astype(np.float64)
    return dict(idx=z['idx'], G=z['eu'].astype(np.float64), GPlog=-np.log(g / a).sum(1), GPlin=-(g / a).sum(1))
SIGN = {}
HAVE_GP = all(os.path.exists(f'{INPUTS}/eu/{t}/{TIERS[t]["id"]}.npz') for t, _ in IDS)
for t, name in IDS:
    Tt = TIERS[t]; run = RUN['pair'][t]
    gdev = escore(FT(DEVRUN[t], Tt['id'])); med = float(np.median(gdev))
    def stats(s):
        F = FT(run, Tt['id'] if s == '__id__' else s); Lr = atom_lrs(F, ATOMS); G = Lr.sum(1)
        return dict(idx=F['idx'], G=G, MabsG=-np.abs(G), Mabs=-np.abs(Lr).sum(1), Mtwo=-np.abs(G - med))
    S0 = stats('__id__'); E0 = gepc_r90(t, Tt['id']); P0 = gp_run(t, Tt['id'])
    SIGN[t] = {}
    for s in Tt['ood']:
        S1 = stats(s); k0, k1 = COM[t]['__id__'], COM[t][s]
        r = {k: 100 * auroc(sub(S0[k], S0['idx'], k0), sub(S1[k], S1['idx'], k1)) for k in ('G', 'MabsG', 'Mabs', 'Mtwo')}
        assert abs(r['G'] - GA[(MAINID, t, s)]) < 1e-6
        E1 = gepc_r90(t, s)
        r['GEPCr'] = 100 * auroc(sub(E0['s'], E0['idx'], k0), sub(E1['s'], E1['idx'], k1)) if (E0 is not None and E1 is not None) else None
        r['GEPC'] = val(t, 'GEPC', s)
        P1 = gp_run(t, s)
        if P0 is not None and P1 is not None:
            r['GPlog'] = 100 * auroc(sub(P0['GPlog'], P0['idx'], k0), sub(P1['GPlog'], P1['idx'], k1)); r['GPlin'] = 100 * auroc(sub(P0['GPlin'], P0['idx'], k0), sub(P1['GPlin'], P1['idx'], k1))
            r['GPcheck'] = 100 * auroc(sub(P0['G'], P0['idx'], k0), sub(P1['G'], P1['idx'], k1))
        SIGN[t][s] = r
SROWS = [('G', 'Ours：Euphonium(符号つき)'), ('MabsG', '$-|\\Eu|$(大きさだけ. 大きいほど分布外)'), ('Mabs', '$-\\sum_{(\\mT,k)}|\\log\\hat e_k(\\mT\\vec{x})/\\hat e_k(\\vec{x})|$(原子ごとの大きさ)'),
         ('GEPCr', '$\\Rot{90}$ の予測差 $\\|\\vec a-\\vec b\\|^2/\\|\\vec b\\|^2$(GEPCの実行)'), ('GEPC', 'GEPC(6 変換, 表~\\ref{tab:cifar10})')]
if HAVE_GP: SROWS.insert(3, ('GPlog', '予測差 $\\|\\vec a_k-\\vec b_k\\|^2/\\|\\vec b_k\\|^2$(同じ %d 回)' % NFEG))
def smean(t, k): return float(np.mean([SIGN[t][s][k] for s in NAT[t]]))
def sign_table(rows, path, note):
    L = []
    for k, lab in rows:
        cells = [lab] + ['%.1f' % smean(t, k) for t, _ in IDS] + ['%.1f' % np.mean([smean(t, k) for t, _ in IDS])]
        if k == 'G': cells = ['\\cellcolor{firstblue}' + c for c in cells]
        L.append(' & '.join(cells) + '\\\\')
    L[-1] += '\\bottomrule'
    open(os.path.join(PAPER, path), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: ' + note + '\n' + '\n'.join(L) + '\n')
# Table 3 is transposed (rows = IDs, columns = statistics)
MAINK = [r[0] for r in SROWS if r[0] not in ('Mabs', 'GEPC')]
L = []
for t, name in IDS + [(None, '平均')]:
    vals = [smean(t, k) if t is not None else float(np.mean([smean(u, k) for u, _ in IDS])) for k in MAINK]
    cells = [name] + ['%.1f' % v for v in vals]; cells[1] = '\\cellcolor{firstblue}' + cells[1]
    L.append(' & '.join(cells) + '\\\\')
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'tab_sign.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: sign or magnitude (Table 3; rows = IDs, columns = statistics; mean AUROC over the natural OOD sets)\n' + '\n'.join(L) + '\n')
M['signColsText'] = ' & '.join({'G': 'Ours：Euphonium', 'MabsG': '$-|\\Eu|$', 'GPlog': '予測差(同じ %d 回)' % NFEG, 'GEPCr': '$\\Rot{90}$ の予測差(GEPC の実行)'}[k] for k in MAINK)
M['signNcols'] = len(MAINK)
# Table A6, also transposed (rows = IDs, columns = every statistic)
ALLK = [r[0] for r in SROWS] + ['Mtwo']
L = []
for t, name in IDS + [(None, '平均')]:
    vals = [smean(t, k) if t is not None else float(np.mean([smean(u, k) for u, _ in IDS])) for k in ALLK]
    cells = [name] + ['%.1f' % v for v in vals]; cells[1] = '\\cellcolor{firstblue}' + cells[1]
    L.append(' & '.join(cells) + '\\\\')
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'supp_sign.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: sign or magnitude (Table A6; rows = IDs, columns = statistics)\n' + '\n'.join(L) + '\n')
SHORT = {'G': 'Euphonium', 'MabsG': '$-|\\Eu|$', 'Mabs': '$-\\sum|\\ell_{\\mT,k}|$', 'GPlog': '予測差(同じ %d 回)' % NFEG, 'GEPCr': '予測差(GEPC)', 'GEPC': 'GEPC', 'Mtwo': '$-|\\Eu-\\mathrm{med}|$'}
M['signAllColsText'] = ' & '.join(SHORT[k] for k in ALLK); M['signAllNcols'] = len(ALLK)
for k, _ in SROWS + [('Mtwo', '')]:
    for t, _ in IDS: M['sign' + k + KEYID[t]] = smean(t, k)
    M['sign' + k + 'Mean'] = float(np.mean([smean(t, k) for t, _ in IDS]))
M['signAbsFlipCten'] = 100 - M['signMabsGCten']          # value with the magnitude reading flipped (larger = ID), CIFAR-10
M['signAbsFlipMean'] = 100 - M['signMabsGMean']
M['signRevChCten'] = SIGN['edmc100']['cifar10_test']['MabsG']; M['signRevINCel'] = SIGN['in64']['in64_celeba_test']['MabsG']
M['signMtwoChCten'] = SIGN['edmc100']['cifar10_test']['Mtwo']        # two-sided statistic on the reversed pair
M['haveGP'] = 1 if HAVE_GP else 0
if HAVE_GP:
    chk = max(abs(SIGN[t][s]['GPcheck'] - SIGN[t][s]['G']) for t, _ in IDS for s in NAT[t]); M['gpCheckMaxDiff'] = chk; print('GP run: max |AUROC(G from GP run) - AUROC(G)| = %.3f' % chk)
print('sign vs magnitude (natural means):', {k: {KEYID[t]: round(smean(t, k), 1) for t, _ in IDS} for k, _ in SROWS + [('Mtwo', '')]})

# ================================================================ data or model: model x data cross-evaluation (CIFAR-10 and CIFAR-100) and training length
CROSS = {}
for mt, mname in (('edmc10', 'CIFAR-10'), ('edmc100', 'CIFAR-100')):
    for s in ('cifar10_test', 'cifar100_test'):
        keep = COM[mt]['__id__' if s == TIERS[mt]['id'] else s]                       # the same common images as the tables
        F = FT(RUN['pair'][mt], s); g = sub(escore(F), F['idx'], keep)
        N0, N1 = nll(NLLRUN[mt], s), nll(NLLRUN[mt] + '_rot90', s); lam = sub(N1['bpd'] - N0['bpd'], N0['idx'], keep)
        CROSS[(mt, s)] = dict(gbar=float(g.mean()), gd=float(g.mean() / g.std()), lam=float(lam.mean()), lamd=float(lam.mean() / lam.std()), n=int(len(g)), nlam=int(len(lam)))
        k = KEYID[mt] + {'cifar10_test': 'Cten', 'cifar100_test': 'Ch'}[s]
        M['crossG' + k] = CROSS[(mt, s)]['gbar']; M['crossD' + k] = CROSS[(mt, s)]['gd']; M['crossL' + k] = CROSS[(mt, s)]['lam']
gcel = gscore('in64', 'in64_celeba_test'); M['gbarINCel'] = float(gcel.mean()); M['gsdINCel'] = float(gcel.std())
# mean of Euphonium of the four 32x32 models on each of the ten sets (all final-split images): narrow-domain models give ~0 elsewhere, the two broad models agree
RGB10 = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
GB = {mt: {s: float(escore(FT(RUN['pair'][mt], s)).mean()) for s in RGB10} for mt in ('edmc10', 'edmc100', 'l6svhn', 'dpceleba')}
M['gbarNarrowMax'] = max(abs(v) for mt in ('l6svhn', 'dpceleba') for s, v in GB[mt].items() if s != TIERS[mt]['id'])
M['gbarBroadDiffMax'] = max(abs(GB['edmc10'][s] - GB['edmc100'][s]) for s in RGB10)
print('Gbar by model x set', {mt: {s: round(v, 2) for s, v in d.items()} for mt, d in GB.items()}, 'narrowMax %.3f broadDiffMax %.3f' % (M['gbarNarrowMax'], M['gbarBroadDiffMax']))
L = []
for mt, mname in (('edmc10', 'CIFAR-10'), ('edmc100', 'CIFAR-100')):
    L.append('%s のモデル & ' % mname + ' & '.join('%.2f (%.2f) & %.3f' % (CROSS[(mt, s)]['gbar'], CROSS[(mt, s)]['gd'], CROSS[(mt, s)]['lam']) for s in ('cifar10_test', 'cifar100_test')) + '\\\\')
L[-1] += '\\bottomrule'
open(os.path.join(PAPER, 'tab_cross.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: model x image cross-evaluation (mean of Euphonium, d\', mean exact Lambda_90 in bits/dim; final-split images)\n' + '\n'.join(L) + '\n')
print('cross', {k: (round(v['gbar'], 3), round(v['gd'], 2), round(v['lam'], 4)) for k, v in CROSS.items()})
# training length (our U-Net, same architecture and training; R180, five scales {0.01, 0.029, 0.05, 0.147, 0.43}, ID images only)
SNAP = {}
for fam, key in (('D3d_c10m', 'Cten'), ('D3d_tin', 'TIN')):
    vals = {}
    fams = [fam, fam + '32'] if key == 'TIN' else [fam]       # D3d_tin32_*: continued training of the same Tiny-ImageNet model (16M, 32M images)
    for p in sorted(os.listdir(f'{INPUTS}/feat')):
        if not any(p.startswith(f_ + '_') for f_ in fams): continue
        nimg = p.split('_')[-1]; f = [x for x in os.listdir(f'{INPUTS}/feat/{p}')][0]
        F = feat(p, f[:-4]); g = lrs(F, list(range(F['e'].shape[1])), T='rot180').sum(1)
        vals[float(nimg[:-1])] = float(g.mean())
    SNAP[key] = dict(sorted(vals.items()))
    M['snapFirst' + key] = SNAP[key][min(SNAP[key])]; M['snapLast' + key] = SNAP[key][max(SNAP[key])]
    M['snapNFirst' + key] = min(SNAP[key]); M['snapNLast' + key] = max(SNAP[key])
    if 8.0 in SNAP[key]: M['snapEight' + key] = SNAP[key][8.0]
print('training snapshots (R180 Gbar)', SNAP)

# ---- Table A4: effect of the shared noise (CIFAR-10, the 12 pairs, all final-split images), with the Euphonium rule
T0 = TIERS['edmc10']
def sets_of(run): return [s for s in T0['ood'] if os.path.exists(f'{INPUTS}/feat/{run}/{s}.npz')]
def ev(run, fn):
    si = fn(FT(run, T0['id'])); return {s: auroc(si, fn(FT(run, s))) for s in sets_of(run)}
mean_ = lambda a: 100 * float(np.mean(list(a.values())))
if NOISE == 'pair': runs, run_alt, run_perimg = ['F_edmc10', 'F_edmc10_s1', 'F_edmc10_s2'], 'F_edmc10_noanti', 'F_edmc10_perimg'
else: runs, run_alt, run_perimg = ['N_edmc10', 'N_edmc10_s1', 'N_edmc10_s2'], 'F_edmc10', 'N_edmc10_perimg'
have = all(os.path.exists(f'{INPUTS}/feat/{r}') for r in runs + [run_alt, run_perimg])
if have:
    S12 = sets_of(runs[1]); assert len(S12) == 12
    def escore_cross(F1, F2):
        """independent noise: errors of the rotated image from run F1, of the upright image from run F2 (another seed)"""
        e1, e2 = F1['e'][:, :, 0, :], F2['e'][:, :, 0, :]; s = np.zeros(e1.shape[0])
        for T, k in ATOMS: s += np.log(e1[:, k, F1['tfs'].index(T)] / e2[:, k, F2['tfs'].index('id')])
        return s
    M['estMain'] = mean_({s: v for s, v in ev(runs[0], escore).items() if s in S12})
    per = [mean_({s: v for s, v in ev(q, escore).items() if s in S12}) for q in runs]
    M['estSeedSd'] = float(np.std(per, ddof=1))
    si = escore_cross(FT(runs[1], T0['id']), FT(runs[0], T0['id']))
    M['estIndep'] = mean_({s: auroc(si, escore_cross(FT(runs[1], s), FT(runs[0], s))) for s in S12})
    M['estAlt'] = mean_(ev(run_alt, escore)); M['estPerimg'] = mean_(ev(run_perimg, escore))
    M['estAltText'] = '共有ノイズ,符号反転対なし' if NOISE == 'pair' else '共有ノイズ,符号反転対(%d 回)' % (2 * NFEG)
    print('controls', {k: round(M[k], 2) for k in ('estMain', 'estSeedSd', 'estIndep', 'estAlt', 'estPerimg')}, 'seeds', [round(p, 2) for p in per])
else:
    print('controls: runs missing', [r for r in runs + [run_alt, run_perimg] if not os.path.exists(f'{INPUTS}/feat/{r}')])

INT = {'varN', 'varDim', 'winsAllOne', 'loseAllOne', 'signAllNcols', 'signNcols', 'tabNbelow', 'winsCten', 'secondCten', 'nIDbest', 'nIDsecond', 'nPairs', 'winsAll', 'secondAll', 'winsAllStrict', 'bestAll', 'nfeG', 'nTerms', 'ruleNscale', 'vflipWins', 'nCand', 'ruleOrientN', 'nRev', 'haveGP', 'ruleNatoms', 'ruleNfeUp', 'ruleNfeRot', 'prevWins', 'secondWins', 'thirdWins', 'nCandTwo', 'GrankSVHN', 'diffBestWins', 'dEuWins', 'dEuIDbest', 'coarseTopLogr', 'coarseTopDiff', 'singleFiveNfe', 'singleFiveWins', 'knnLoseN'}
# ---- Table A5: the shared noise removes the |eps|^2 fluctuation from the difference (R90, final-split ID images, three seeds)
if have:
    Fs = [FT(r, T0['id']) for r in runs]; assert all((F['idx'] == Fs[0]['idx']).all() for F in Fs)
    ii, jj = Fs[0]['tfs'].index('id'), Fs[0]['tfs'].index('rot90')
    L = []
    for k in range(10):
        e0 = [F['e'][:, k, 0, ii] for F in Fs]; eT = [F['e'][:, k, 0, jj] for F in Fs]
        m0 = [float(a.mean()) for a in e0]; md = [float((b - a).mean()) for a, b in zip(e0, eT)]
        dev0 = max(abs(v - np.mean(m0)) for v in m0); devd = max(abs(v - np.mean(md)) for v in md)
        var_ind = float(np.var(eT[1] - e0[0])); var_sh = float(np.var(eT[0] - e0[0]))
        L.append('%.3g & %.0f & %.1f & %.1f & %.1f & %.0f & %.0f\\\\' % (Fs[0]['sigmas'][k], m0[0], md[0], dev0, devd, var_ind, var_sh))
        if k == 0: M['varFineMean'] = m0[0]; M['varFineDiff'] = md[0]
        M['varDevMax'] = max(M.get('varDevMax', 0), dev0); M['varDevDiffMax'] = max(M.get('varDevDiffMax', 0), devd)
    L[-1] += '\\bottomrule'
    open(os.path.join(PAPER, 'supp_variance.tex'), 'w', encoding='utf-8').write('% written by paper/gen/make_paper.py: effect of the shared noise (R90, final-split ID images, three seeds)\n' + '\n'.join(L) + '\n')
    M['varN'] = int(len(Fs[0]['idx'])); M['varDim'] = int(Fs[0]['dim'])
    M['varFineDiffMin'] = float(min(md[0] for k in range(1)))
    print('variance table written', M['varDevMax'], M['varDevDiffMax'])

with open(os.path.join(PAPER, 'numbers.tex'), 'w', encoding='utf-8') as f:
    f.write('% written by paper/gen/make_paper.py: numbers used in the text\n')
    for k, v in M.items():
        assert not any(ch.isdigit() for ch in k), k
        if isinstance(v, str): f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
        elif k in ('shGDog', 'shGSvhn', 'shLRDog', 'shLRSvhn'): f.write('\\newcommand{\\%s}{%+.2f}\n' % (k, v))
        elif k.startswith('lamMed') or k.startswith('lamSd') or k.startswith('lamMean'): f.write('\\newcommand{\\%s}{%.3f}\n' % (k, v))
        elif k.startswith('rho') or k.startswith('nllMed') or k == 'shSigma' or k.startswith('crossG') or k.startswith('crossD') or k.startswith('gbar') or k.startswith('gsd') or k.startswith('sigDrop') or k == 'sigTen': f.write('\\newcommand{\\%s}{%.2f}\n' % (k, v))
        elif k in INT or k.startswith('oN') or k.startswith('psArgmax'): f.write('\\newcommand{\\%s}{%d}\n' % (k, v))
        elif k.startswith('Gmed') or k.startswith('shE') or k.startswith('crossL') or k.startswith('snapFirst') or k.startswith('snapLast') or k.startswith('snapEight'): f.write('\\newcommand{\\%s}{%.3f}\n' % (k, v))
        elif k.startswith('snapN'): f.write('\\newcommand{\\%s}{%g}\n' % (k, v))
        elif k == 'odeOverG': f.write('\\newcommand{\\%s}{%.0f}\n' % (k, round(v)))
        else: f.write('\\newcommand{\\%s}{%.1f}\n' % (k, v))
print('numbers:', {k: (v if isinstance(v, str) else round(float(v), 3)) for k, v in M.items()})
