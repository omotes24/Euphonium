# baselines_eval.py -- the eight comparison methods on the 63 pairs: per-image scores from their statistics, AUROC / FPR95 on the
#   images every method has (benchmark/common_idx.json).  Writes results/baselines_pairs.json = {tag: {method: {set: {auroc, fpr95,
#   n_id, n_ood}}}}.  All scores are oriented "higher = more in-distribution"; every detector is fitted on the ID development images
#   only (<id>__cal: the first images of the fixed development permutation), with the official settings:
#     NLL          PF-ODE log-likelihood (experiments/nll.py)
#     IC-png/j2k   log-likelihood in bits + PNG / JPEG 2000 code length (input-complexity correction; experiments/complexity.py)
#     TT           typicality test: -|log p(x) - mean of log p over the development images|
#     MSMA*        Mahalanobis distance (Ledoit-Wolf) of the log score norms at 20 scales (experiments/extract.py)
#     DiffPath     6-D path statistics + Gaussian mixture (GridSearchCV over n_components {50, 100} x 4 covariance types, 10 folds)
#     GEPC         per-level equivariance features, level selection by the coefficient of variation, 1/CV weights, KDE log-densities
#     EigenScore   top-3 posterior-covariance eigenvalues, z-scored per timestep on ID, summed
#   (the raw statistics of the last three are written by experiments/baselines.py)
import os, sys, json
import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TIERS, INPUTS, RESULTS, BASELINES, nll, base, complexity, common_idx, sub, auroc, fpr95


class Maha:
    """Gaussian fit (Ledoit-Wolf) on standardised calibration features; score = -Mahalanobis distance^2"""
    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.P = LedoitWolf().fit((X - self.mu) / self.sd).precision_
        return self

    def score(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return -np.einsum('ij,jk,ik->i', Z, self.P, Z)


def load2(raw, tag, name):
    p = f'{INPUTS}/base2/{raw}_{tag}/{name}.npz'
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- DiffPath (eval_6d.py of the official code)
def dp6(z): return np.stack([z['e1'].sum(1), z['e2'].sum(1), z['e3'].sum(1), z['d1'].sum(1), z['d2'].sum(1), z['d3'].sum(1)], 1)


class DiffPath6D:
    def fit(self, Z, n_jobs=min(8, os.cpu_count() or 1)):
        grid = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('GMM', GaussianMixture(random_state=0))]),
                            {'GMM__n_components': [50, 100], 'GMM__covariance_type': ['full', 'tied', 'diag', 'spherical'],
                             'GMM__max_iter': [200]}, cv=10, n_jobs=n_jobs)
        grid.fit(dp6(Z)); self.best = grid.best_params_; self.m = grid.best_estimator_
        return self

    def score(self, Z): return self.m.score_samples(dp6(Z))


# ----------------------------------------------------------------------------- GEPC (gepc/methods/gepc.py; configs/gepc_*.yaml)
GEPC_CONF = {   # keep_k 2, weight_t inv_cv, density_mode kde
    'cifar10': dict(snr=(0.99997, 0.9999, 0.9979, 0.9969), rho=0.3, feats=('gepc_s',), agg='sum'),
    'svhn':    dict(snr=(0.99997, 0.9999, 0.9979, 0.9969), rho=0.37, feats=('gepc_s',), agg='mean'),
    'celeba':  dict(snr=(0.99997, 0.99, 0.999, 0.9969), rho=0.3, feats=('gepc_s', 'gepc_s_cos', 'gepc_s_pair'), agg='sum')}
GEPC_TIER = {'edmc10': 'cifar10', 'l6svhn': 'svhn', 'dpceleba': 'celeba', 'edmc100': 'cifar10', 'in64': 'cifar10', 'fmnist': 'cifar10'}


def gepc_feats(z, l, j):
    """official per-level features (all OOD-high); the identity element of the 7-element group contributes zero"""
    return dict(gepc_s=z['num'][:, l, :, j].sum(1) / 7 / z['den'][:, l, j],
                gepc_s_cos=(1 - z['cos'][:, l]).sum(1) / 7,
                gepc_s_pair=z['pair'][:, l, j] / z['den'][:, l, j])


class KDE1D:
    """official _KDE1D: Gaussian KDE, bandwidth 0.9 min(std, IQR / 1.34) N^-1/5"""
    def fit(self, x):
        x = np.asarray(x, np.float64); x = x[np.isfinite(x)]; N = max(1, len(x))
        q25, q75 = np.quantile(x, [0.25, 0.75]); iqr = q75 - q25; std = max(float(x.std(ddof=1)), 1e-8)
        sig = min(std, iqr / 1.34 if iqr > 0 else std); self.bw = 0.9 * sig * N ** (-0.2)
        if not np.isfinite(self.bw) or self.bw <= 1e-6: self.bw = 0.1
        self.ref = x; return self

    def logp(self, z):
        var = self.bw ** 2
        return logsumexp(-0.5 * np.log(2 * np.pi * var) - (np.asarray(z, np.float64)[:, None] - self.ref[None, :]) ** 2 / (2 * var), axis=1) - np.log(len(self.ref))


class GEPC:
    def __init__(self, conf='cifar10', keep_k=2, weight='inv_cv'):
        self.c = GEPC_CONF[conf]; self.keep_k, self.weight = keep_k, weight

    def _levels(self, Z):
        snr = [float(s) for s in Z['gepc_snr']]
        return [snr.index(s) for s in self.c['snr']], [float(p) for p in Z['pools']].index(self.c['rho'])

    def fit(self, Z):
        L, j = self._levels(Z)
        F = {l: gepc_feats(Z, l, j) for l in L}
        cv = {}
        for l in L:
            v = F[l]['gepc_s']; mu, sd = v.mean(), v.std(ddof=1)
            cv[l] = (max(sd, 1e-8) / (abs(mu) + 1e-6), max(sd, 1e-6) / (abs(mu) + 1e-6))   # (mask, weights) as in the official code
        w = {l: 1.0 / (cv[l][1] + 1e-6) for l in L} if self.weight == 'inv_cv' else {l: 1.0 for l in L}
        s = sum(w.values()); self.w = {l: w[l] / s for l in L}
        self.kept = sorted(sorted(L, key=lambda l: cv[l][0])[:self.keep_k])
        self.kde = {(l, f): KDE1D().fit(F[l][f]) for l in L for f in self.c['feats']}
        self.j = j; return self

    def score(self, Z):
        S, W = [], []
        for l in self.kept:
            F = gepc_feats(Z, l, self.j)
            Ls = np.stack([self.kde[(l, f)].logp(F[f]) for f in self.c['feats']], 0)
            S.append(Ls.mean(0) if self.c['agg'] == 'mean' else Ls.sum(0)); W.append(self.w[l])
        W = np.array(W) / (np.sum(W) + 1e-12)
        return (W[:, None] * np.stack(S, 0)).sum(0)          # ID-like log-density (the official OOD score is its negative)


# ----------------------------------------------------------------------------- EigenScore (Alg. 1)
class EigenScore:
    def __init__(self, K=3): self.K = K
    def m(self, Z): return Z['lam'][..., :self.K].sum(-1).mean(2)          # [N, T]: top-K sum, mean over the repetitions
    def fit(self, Z): m = self.m(Z); self.mu, self.sd = m.mean(0), m.std(0) + 1e-12; return self
    def score(self, Z): return -((self.m(Z) - self.mu) / self.sd).sum(1)


RAW = {'DiffPath': ('diffpath', lambda tag: DiffPath6D()), 'GEPC': ('gepc', lambda tag: GEPC(GEPC_TIER[tag])), 'EigenScore': ('eigen', lambda tag: EigenScore())}


def tier_scores(tag, T):
    """{set: {method: (score, idx)}} for the ID set and every OOD set of one ID model"""
    run = f'F_{tag}'
    calN, calB = nll(run, f"{T['id']}__cal"), base(run, f"{T['id']}__cal")
    mu_ll = calN['ll'].mean(); mm = Maha().fit(np.log(calB['s']))
    fitted = {}
    for m, (raw, mk) in RAW.items():
        Z = load2(raw, tag, T['id'] + '__cal'); fitted[m] = mk(tag).fit(Z)
    out = {}
    for s in [T['id']] + T['ood']:
        d = {}
        N = nll(run, s)
        if N is not None:
            d['NLL'] = (N['ll'], N['idx']); d['TT'] = (-np.abs(N['ll'] - mu_ll), N['idx'])
            C = complexity(s, N['idx'])
            if C is not None:
                d['IC-png'] = (N['ll'] / np.log(2) + C['png'], N['idx']); d['IC-j2k'] = (N['ll'] / np.log(2) + C['j2k'], N['idx'])
        B = base(run, s)
        if B is not None: d['MSMA*'] = (mm.score(np.log(B['s'])), B['idx'])
        for m, (raw, _) in RAW.items():
            Z = load2(raw, tag, s)
            if Z is not None: d[m] = (fitted[m].score(Z), Z['idx'])
        out[s] = d
    return out, {'DiffPath_grid': {k: str(v) for k, v in fitted['DiffPath'].best.items()}, 'GEPC_kept_levels': [int(l) for l in fitted['GEPC'].kept]}


if __name__ == '__main__':
    COM, ALL = common_idx(), {}
    for tag, T in TIERS.items():
        S, info = tier_scores(tag, T)
        res = {}
        for m in BASELINES:
            res[m] = {}
            si, ii = S[T['id']][m]; a = sub(si, ii, COM[tag]['__id__'])
            for s in T['ood']:
                if m not in S[s]: continue                    # PF-ODE methods were not run on the synthetic sets
                so, io = S[s][m]; b = sub(so, io, COM[tag][s])
                res[m][s] = dict(auroc=auroc(a, b), fpr95=fpr95(a, b), n_id=len(a), n_ood=len(b))
        ALL[tag] = dict(common=res, fit=info)
        print(f'{tag:9s} ' + '  '.join(f'{m} {100 * np.mean([v["auroc"] for s, v in res[m].items() if "vflip" not in s and "synth" not in s]):.1f}' for m in BASELINES), flush=True)
    json.dump(ALL, open(os.path.join(RESULTS, 'baselines_pairs.json'), 'w'), indent=1)
    print('saved results/baselines_pairs.json')
