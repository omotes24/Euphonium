# core.py -- shared utilities: data, model loading, denoiser API, per-scale statistics
import os, sys, math, pickle, time
import numpy as np
import torch

ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPY = f'{ROOT}/data/npy'


# ----------------------------------------------------------------------------- data
def load_npy(name, n=None, seed=0):
    """uint8 [N,32,32,3]; optional fixed random subset of size n (sorted indices)."""
    x = np.load(f'{NPY}/{name}.npy', mmap_mode='r')
    if n is not None and n < len(x):
        idx = np.sort(np.random.RandomState(seed).choice(len(x), n, replace=False))
        return np.ascontiguousarray(x[idx]), idx
    return np.ascontiguousarray(x), np.arange(len(x))


def select(name, n=None, seed=0, complement=False, cap=0):
    """Development subset = first n images of a fixed permutation (seed).  complement=True returns the images outside it
    (the final evaluation split), optionally capped to `cap` images drawn with a second fixed seed."""
    if not complement: return load_npy(name, n, seed)
    full = np.load(f'{NPY}/{name}.npy', mmap_mode='r')
    idx = np.arange(len(full))
    if n is not None and n < len(full):
        _, dev_idx = load_npy(name, n, seed)
        idx = np.setdiff1d(idx, dev_idx)
    if cap and cap < len(idx): idx = np.sort(np.random.RandomState(1000 + seed).choice(idx, cap, replace=False))
    return np.ascontiguousarray(full[idx]), idx


def to_t(x_uint8, device):
    x = torch.from_numpy(x_uint8).to(device)
    if x.ndim == 3: x = x[..., None]
    return x.permute(0, 3, 1, 2).float() / 127.5 - 1.0


# ----------------------------------------------------------------------------- models
class Denoiser:
    """Unified API: den(x_noisy, sigma) -> E[x0 | x_noisy], with x_noisy = x0 + sigma * eps (EDM convention)."""
    def __init__(self, fn, name, sigma_min=0.002, sigma_max=80.0):
        self.fn, self.name, self.sigma_min, self.sigma_max = fn, name, sigma_min, sigma_max

    def __call__(self, x, sigma):
        if not torch.is_tensor(sigma): sigma = torch.full([x.shape[0]], float(sigma), device=x.device)
        elif sigma.ndim == 0: sigma = sigma.expand(x.shape[0])
        return self.fn(x, sigma)


FP16 = os.environ.get('SR_FP16', '1') == '1'


def edm_precond_fn(unet, sigma_data=0.5, fp16=None):
    """EDM preconditioning around a raw U-Net F(c_in * x, c_noise); identical to NVlabs EDMPrecond, optional fp16 autocast."""
    def fn(x, sigma):
        s = sigma.reshape(-1, 1, 1, 1).float()
        c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
        c_out = s * sigma_data / (s ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + s ** 2).sqrt()
        c_noise = s.log() / 4
        with torch.autocast('cuda', dtype=torch.float16, enabled=(FP16 if fp16 is None else fp16)):
            F = unet(c_in * x, c_noise.flatten(), class_labels=None, augment_labels=None)
        return c_skip * x + c_out * F.float()
    return fn


def load_edm_pkl(path, device, key='ema'):
    if f'{ROOT}/ext/edm' not in sys.path: sys.path.insert(0, f'{ROOT}/ext/edm')
    with open(path, 'rb') as f:
        net = pickle.load(f)[key].to(device).eval().requires_grad_(False)
    return Denoiser(edm_precond_fn(net.model, float(net.sigma_data)), os.path.basename(path)), net


def load_edm_sd(path, device):
    """plain state_dict of an NVlabs EDMPrecond with the DDPM++ CIFAR configuration (third-party CIFAR-100 checkpoint)."""
    if f'{ROOT}/ext/edm' not in sys.path: sys.path.insert(0, f'{ROOT}/ext/edm')
    from training.networks import EDMPrecond
    net = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, model_type='SongUNet', embedding_type='positional',
                     encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
                     model_channels=128, channel_mult=[2, 2, 2], augment_dim=9)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    net.to(device).eval().requires_grad_(False)
    return Denoiser(edm_precond_fn(net.model, 0.5), os.path.basename(path)), net


def load_iddpm(path, device, image_size=32, num_channels=128, num_res_blocks=3, dropout=0.3, steps=4000, schedule='cosine'):
    """OpenAI improved-diffusion checkpoint (eps-prediction, learned sigma), e.g. DiffPath's CelebA 32x32 model.
    EDM convention: x_t / sqrt(abar_t) = x0 + sigma_t * eps with sigma_t = sqrt((1 - abar_t) / abar_t); the nearest timestep is used."""
    p = f'{ROOT}/ext/improved-diffusion'
    if p not in sys.path: sys.path.insert(0, p)
    from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
    a = model_and_diffusion_defaults()
    a.update(image_size=image_size, num_channels=num_channels, num_res_blocks=num_res_blocks, learn_sigma=True,
             diffusion_steps=steps, noise_schedule=schedule, dropout=dropout)
    model, diffusion = create_model_and_diffusion(**a)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.to(device).eval().requires_grad_(False)
    ac = torch.tensor(np.asarray(diffusion.alphas_cumprod), dtype=torch.float64, device=device)
    logsig = 0.5 * (torch.log1p(-ac) - torch.log(ac))
    def fn(x, sigma):
        t = (logsig[None, :] - sigma.double().log()[:, None]).abs().argmin(1)
        a_t = ac[t].sqrt().float().view(-1, 1, 1, 1)
        s_t = logsig[t].exp().float().view(-1, 1, 1, 1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=FP16):
            out = model(a_t * x, t.float() * (1000.0 / steps))
        return x - s_t * out[:, :x.shape[1]].float()
    return Denoiser(fn, os.path.basename(path)), model


def load_layer6(path, device, sign=-1.0):
    """layer6ai-labs/dgm_ood_detection checkpoint: diffusers UNet2DModel, continuous VP-SDE (beta 0.1 -> 20), raw t in [0,1]."""
    from diffusers import UNet2DModel
    net = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
                      block_out_channels=[128, 128, 256, 256, 512, 512],
                      down_block_types=('DownBlock2D',) * 4 + ('AttnDownBlock2D', 'DownBlock2D'),
                      up_block_types=('UpBlock2D', 'AttnUpBlock2D') + ('UpBlock2D',) * 4)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    pre = 'score_network.score_network.'
    sd = {k[len(pre):]: v for k, v in ck['module_state_dict'].items() if k.startswith(pre)}
    net.load_state_dict(sd); net.to(device).eval().requires_grad_(False)
    b0, b1 = 0.1, 20.0
    def fn(x, sigma):
        s2 = sigma.double() ** 2
        L = torch.log1p(s2)                                   # -log(alpha^2) = 0.5 (b1-b0) t^2 + b0 t
        t = (-b0 + torch.sqrt(b0 ** 2 + 2 * (b1 - b0) * L)) / (b1 - b0)
        alpha = (1 / (1 + s2)).sqrt().float().view(-1, 1, 1, 1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=FP16):
            out = net(alpha * x, t.float()).sample
        return x - sigma.view(-1, 1, 1, 1).float() * (sign * out.float())
    return Denoiser(fn, os.path.basename(path)), net


def load_edm2_pkl(path, device, key='ema'):
    """NVlabs EDM2 pickle (needs ext/edm2 on the path; do not mix with EDM-v1 pickles in one process)."""
    assert f'{ROOT}/ext/edm' not in sys.path, 'EDM and EDM2 pickles cannot be mixed in one process'
    if f'{ROOT}/ext/edm2' not in sys.path: sys.path.insert(0, f'{ROOT}/ext/edm2')
    with open(path, 'rb') as f:
        net = pickle.load(f)[key].to(device).eval().requires_grad_(False)
    def fn(x, sigma):
        return net(x, sigma, None, force_fp32=not FP16).to(torch.float32)
    return Denoiser(fn, os.path.basename(path)), net


def load_own(path, device, key='ema'):
    """Checkpoints written by train.py: dict(cfg=..., ema=state_dict, net=state_dict)."""
    from unet import build_net
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = build_net(ck['cfg']).to(device)
    net.load_state_dict(ck[key]); net.eval().requires_grad_(False)
    if not FP16 and hasattr(net, 'fp16'): net.fp16 = False          # SR_FP16=0 -> full precision
    def fn(x, sigma):
        return net(x, sigma)
    return Denoiser(fn, os.path.basename(path)), net


def load_model(spec, device):
    """spec: 'edm:<file under ckpt/>' or 'own:<path under runs/>'."""
    kind, p = spec.split(':', 1)
    if kind == 'edm': return load_edm_pkl(f'{ROOT}/ckpt/{p}', device)
    if kind == 'edm2': return load_edm2_pkl(f'{ROOT}/ckpt/{p}', device)
    if kind == 'edmsd': return load_edm_sd(f'{ROOT}/ckpt/{p}', device)
    if kind == 'edmraw':                         # any NVlabs EDM-repo pickle called as-is (e.g. VPPrecond baseline), fp32
        if f'{ROOT}/ext/edm' not in sys.path: sys.path.insert(0, f'{ROOT}/ext/edm')
        with open(f'{ROOT}/ckpt/{p}', 'rb') as f:
            net = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
        return Denoiser(lambda x, s: net(x, s, None).to(torch.float32), p), net
    if kind == 'iddpmc10':                       # OpenAI improved-diffusion CIFAR-10 (L_hybrid)
        return load_iddpm(f'{ROOT}/ckpt/{p}', device, image_size=32, num_channels=128, num_res_blocks=3, dropout=0.3, steps=4000)
    if kind == 'iddpm': return load_iddpm(f'{ROOT}/ckpt/{p}', device)
    if kind == 'l6': return load_layer6(f'{ROOT}/ckpt/{p}', device)
    if kind == 'l6pos': return load_layer6(f'{ROOT}/ckpt/{p}', device, sign=1.0)
    if kind == 'edmmc':                          # same weights, dropout kept active (MC dropout)
        den, net = load_edm_pkl(f'{ROOT}/ckpt/{p}', device); net.model.train(); return den, net
    if kind == 'ownmc':
        den, net = load_own(f'{ROOT}/runs/{p}', device); net.unet.train(); return den, net
    if kind == 'own': return load_own(f'{ROOT}/runs/{p}', device)
    if kind == 'ownraw': return load_own(f'{ROOT}/runs/{p}', device, key='net')
    raise ValueError(spec)


# ----------------------------------------------------------------------------- transforms (orthogonal pixel maps)
def _shuf(x, g, perm, inverse=False):
    B, C, H, W = x.shape
    p = H // g
    y = x.reshape(B, C, g, p, g, p).permute(0, 1, 2, 4, 3, 5).reshape(B, C, g * g, p, p)
    idx = torch.as_tensor(perm, device=x.device)
    if inverse:
        inv = torch.empty_like(idx); inv[idx] = torch.arange(len(idx), device=x.device); idx = inv
    y = y[:, :, idx]
    return y.reshape(B, C, g, g, p, p).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

_P4 = np.random.RandomState(123).permutation(16).tolist()
_P2 = [3, 2, 1, 0]

TF = {
    'id':     (lambda x: x, lambda x: x),
    'hflip':  (lambda x: x.flip(3), lambda x: x.flip(3)),
    'vflip':  (lambda x: x.flip(2), lambda x: x.flip(2)),
    'rot90':  (lambda x: x.rot90(1, (2, 3)), lambda x: x.rot90(-1, (2, 3))),
    'rot180': (lambda x: x.rot90(2, (2, 3)), lambda x: x.rot90(-2, (2, 3))),
    'rot270': (lambda x: x.rot90(3, (2, 3)), lambda x: x.rot90(-3, (2, 3))),
    'transp': (lambda x: x.transpose(2, 3), lambda x: x.transpose(2, 3)),
    'atransp': (lambda x: x.transpose(2, 3).flip(2).flip(3), lambda x: x.flip(2).flip(3).transpose(2, 3)),
    'roll16': (lambda x: x.roll((16, 16), (2, 3)), lambda x: x.roll((-16, -16), (2, 3))),
    'rollx1': (lambda x: x.roll(1, 3), lambda x: x.roll(-1, 3)),          # one-pixel circular shifts: semantics-preserving controls
    'rolly1': (lambda x: x.roll(1, 2), lambda x: x.roll(-1, 2)),
    'rollx2': (lambda x: x.roll(2, 3), lambda x: x.roll(-2, 3)),
    'rollh':  (lambda x: x.roll(16, 3), lambda x: x.roll(-16, 3)),
    'rollv':  (lambda x: x.roll(16, 2), lambda x: x.roll(-16, 2)),
    'shuf4':  (lambda x: _shuf(x, 4, _P4), lambda x: _shuf(x, 4, _P4, True)),
    'neg':    (lambda x: -x, lambda x: -x),
    'cperm':  (lambda x: x[:, [1, 2, 0]], lambda x: x[:, [2, 0, 1]]),
    # orthogonal maps of the colour space (gray axis fixed): hue rotations and luminance negation
    'hue120': (lambda x: x[:, [1, 2, 0]], lambda x: x[:, [2, 0, 1]]),
    'hue240': (lambda x: x[:, [2, 0, 1]], lambda x: x[:, [1, 2, 0]]),
    'hue180': (lambda x: 2 * x.mean(1, keepdim=True) - x, lambda x: 2 * x.mean(1, keepdim=True) - x),
    'lneg':   (lambda x: x - 2 * x.mean(1, keepdim=True), lambda x: x - 2 * x.mean(1, keepdim=True)),
}


# ----------------------------------------------------------------------------- DCT bands
def dct_matrix(n, device):
    k = torch.arange(n, device=device, dtype=torch.float64)[:, None]
    i = torch.arange(n, device=device, dtype=torch.float64)[None, :]
    m = torch.cos(math.pi * (i + 0.5) * k / n) * math.sqrt(2.0 / n)
    m[0] = m[0] / math.sqrt(2.0)
    return m.float()


def band_masks(n, device, edges=None):
    if edges is None: edges = tuple(2 ** i for i in range(int(round(math.log2(n))) + 1))   # 1, 2, ..., n: every DCT frequency is covered
    u = torch.arange(n, device=device)
    f = torch.maximum(u[:, None], u[None, :])
    lo, masks = 0, []
    for hi in edges:
        masks.append(((f >= lo) & (f < hi)).float()); lo = hi
    return torch.stack(masks)                                       # [nb, n, n]


class Bands:
    def __init__(self, n, device):
        self.C = dct_matrix(n, device); self.M = band_masks(n, device)
        self.count = self.M.sum((1, 2))

    def energy(self, r):
        """r [B,C,H,W] -> band energies [B, nb] (sum of squared DCT coefficients over channels)."""
        c = self.C @ r @ self.C.T
        return torch.einsum('bchw,nhw->bn', c * c, self.M)


# ----------------------------------------------------------------------------- per-scale statistics
def shared_noise(shape, seed, k, j, device):
    """One noise image shared by every test image; draws come in antithetic pairs (2m, 2m+1)."""
    g = torch.Generator(device='cpu').manual_seed(int(seed) * 1000003 + k * 1009 + (j // 2))
    eps = torch.randn([1] + list(shape), generator=g).to(device)
    return -eps if (j % 2 == 1) else eps


def shared_probe(shape, seed, k, m, device):
    g = torch.Generator(device='cpu').manual_seed(int(seed) * 7000003 + k * 1013 + m + 17)
    return (torch.randint(0, 2, [1] + list(shape), generator=g).float() * 2 - 1).to(device)


@torch.no_grad()
def scale_stats(den, x, sigmas, n_draws=2, tfs=('id',), band_tfs=('id',), seed=0, bands=None,
                clean=True, hutch=0, power=0):
    """Per-image, per-scale statistics of a denoiser.  x: [B,C,H,W] in [-1,1].
    e_<T>  [B,K]     mean_j ||x - T^-1 D(T(x + s*eps_j))||^2 / s^2 / dim      (noised input, shared noise)
    eb_<T> [B,K,nb]  the same residual energy split into DCT bands (sum over the band, / s^2)
    B, V   [B,K]     bias / variance split of e_id over the draws
    s      [B,K]     ||x - D(x)||^2 / s^2 / dim at the clean input            (MSMA-type score norm)
    tr     [B,K]     tr J_D(x) / dim at the clean input (Hutchinson, finite differences)
    lam    [B,K]     dominant eigenvalue of J_D(x) by power iteration (optional)
    """
    dev = x.device
    Bn, shape = x.shape[0], x.shape[1:]
    dim = float(np.prod(shape))
    K = len(sigmas)
    out = {f'e_{t}': torch.zeros(Bn, K, device=dev) for t in tfs}
    if bands is not None:
        for t in band_tfs: out[f'eb_{t}'] = torch.zeros(Bn, K, len(bands.count), device=dev)
    out['B'] = torch.zeros(Bn, K, device=dev); out['V'] = torch.zeros(Bn, K, device=dev)
    if clean: out['s'] = torch.zeros(Bn, K, device=dev)
    if hutch > 0: out['tr'] = torch.zeros(Bn, K, device=dev)
    if power > 0: out['lam'] = torch.zeros(Bn, K, device=dev)
    for k, sig in enumerate(sigmas):
        sig = float(sig)
        dsum = torch.zeros_like(x); d2sum = torch.zeros(Bn, device=dev)
        for j in range(n_draws):
            xn = x + sig * shared_noise(shape, seed, k, j, dev)
            for t in tfs:
                f, finv = TF[t]
                d = finv(den(f(xn), sig))
                r = (x - d) / sig
                out[f'e_{t}'][:, k] += r.square().flatten(1).sum(1) / dim / n_draws
                if bands is not None and t in band_tfs:
                    out[f'eb_{t}'][:, k] += bands.energy(r) / n_draws
                if t == 'id':
                    dsum += d; d2sum += d.square().flatten(1).sum(1)
        dbar = dsum / n_draws
        out['B'][:, k] = (x - dbar).square().flatten(1).sum(1) / sig ** 2 / dim
        out['V'][:, k] = (d2sum / n_draws - dbar.square().flatten(1).sum(1)) / sig ** 2 / dim
        if hutch > 0:
            h = min(0.05 * sig, 0.01)
            acc = torch.zeros(Bn, device=dev); dc = torch.zeros_like(x)
            for m in range(hutch):
                v = shared_probe(shape, seed, k, m, dev)
                dp, dm = den(x + h * v, sig), den(x - h * v, sig)
                acc += (v * (dp - dm) / (2 * h)).flatten(1).sum(1) / hutch
                dc += (dp + dm) / (2 * hutch)
            out['tr'][:, k] = acc / dim
            if clean: out['s'][:, k] = (x - dc).square().flatten(1).sum(1) / sig ** 2 / dim
        elif clean:
            out['s'][:, k] = (x - den(x, sig)).square().flatten(1).sum(1) / sig ** 2 / dim
        if power > 0:
            h = min(0.05 * sig, 0.01)
            v = shared_probe(shape, seed, k, 999, dev).expand_as(x).clone()
            v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            lam = torch.zeros(Bn, device=dev)
            for it in range(power):
                jv = (den(x + h * v, sig) - den(x - h * v, sig)) / (2 * h)
                lam = (v * jv).flatten(1).sum(1)
                v = jv / jv.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            out['lam'][:, k] = lam
    return {k: v.float().cpu() for k, v in out.items()}


# ----------------------------------------------------------------------------- metrics
def auroc(id_scores, ood_scores):
    """AUROC with ID as the positive class; scores are 'higher = more in-distribution'."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    return float(roc_auc_score(y, s))


def fpr95(id_scores, ood_scores):
    thr = np.percentile(id_scores, 5.0)          # accept 95% of ID
    return float(np.mean(ood_scores >= thr))


class Maha:
    """Gaussian fit on calibration features; score = -Mahalanobis distance^2 (higher = more ID)."""
    def __init__(self, shrink='lw'):
        self.shrink = shrink

    def fit(self, F):
        from sklearn.covariance import LedoitWolf, EmpiricalCovariance
        F = np.asarray(F, np.float64)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-12
        Z = (F - self.mu) / self.sd
        est = LedoitWolf().fit(Z) if self.shrink == 'lw' else EmpiricalCovariance().fit(Z)
        self.P = est.precision_
        return self

    def score(self, F):
        Z = (np.asarray(F, np.float64) - self.mu) / self.sd
        return -np.einsum('ij,jk,ik->i', Z, self.P, Z)


# ----------------------------------------------------------------------------- two-rail statistics (round B onwards)
@torch.no_grad()
def rails(den, x, sigmas, n_pairs=2, tfs=('id',), seed=0, bands=None, extra=True, alts=None, gpairs=(),
          anti=True, perimage=False, offset=0):
    """Two-rail statistics from the predicted noise  eps_hat = (x_noisy - D(x_noisy)) / sigma  on shared antithetic noise.
    For every transform T, scale k, antithetic pair p and DCT band b (band sums, not normalised):
        A_<T>[B,K,P,nb] = || P_b eps_hat ||^2            (predicted-noise energy)
        C_<T>[B,K,P,nb] = < P_b eps, P_b eps_hat >       (noise-prediction correlation; E_eps C = D - tr J_D by Stein's lemma)
    so that  ||eps - eps_hat||^2 = ||eps||^2 + A - 2C   (I-MMSE integrand)  and  A - C  is the orthogonality (calibration) gap.
    extra (T = id only):  V[B,K,P] = ||D+ - D-||^2 / (4 s^2),   Bm[B,K,P] = ||x - (D+ + D-)/2||^2 / s^2
    Also returns N[K,P,nb] = ||P_b eps||^2 of the shared noise.
    Ablation switches:  anti=False -> a single draw (+eps only) instead of the antithetic pair;
                        perimage=True -> an independent noise image for every test image (Ni[B,K,P,nb] holds its band energies).
    """
    dev = x.device
    Bn, shape = x.shape[0], x.shape[1:]
    K, nb = len(sigmas), (len(bands.count) if bands is not None else 1)
    out = {}
    for t in tfs:
        out[f'A_{t}'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
        out[f'C_{t}'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
    if extra:
        out['V'] = torch.zeros(Bn, K, n_pairs, device=dev); out['Bm'] = torch.zeros(Bn, K, n_pairs, device=dev)
    alts = alts or {}
    for nm in alts:                                # other denoisers evaluated on the same noised inputs (identity transform)
        out[f'A_@{nm}'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
        out[f'C_@{nm}'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
    for u, v in gpairs:                            # disagreement energy ||P_b (D_u - D_v)||^2 / s^2
        out[f'G_{u}_{v}'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
    Nn = torch.zeros(K, n_pairs, nb)

    def bsum(u, v):                               # band-wise inner product of two [B,C,H,W] fields -> [B, nb]
        if bands is None: return (u * v).flatten(1).sum(1, keepdim=True)
        cu, cv = bands.C @ u @ bands.C.T, bands.C @ v @ bands.C.T
        return torch.einsum('bchw,nhw->bn', cu * cv, bands.M)

    signs = (1.0, -1.0) if anti else (1.0,)
    w = 1.0 / len(signs)
    if perimage: out['Ni'] = torch.zeros(Bn, K, n_pairs, nb, device=dev)
    for k, sig in enumerate(sigmas):
        sig = float(sig)
        for p in range(n_pairs):
            if perimage:
                g = torch.Generator(device='cpu').manual_seed(int(seed) * 1000003 + k * 1009 + p + 15485863 * (int(offset) + 1))
                eps = torch.randn([Bn] + list(shape), generator=g).to(dev)
                out['Ni'][:, k, p] = bsum(eps, eps)
            else:
                eps = shared_noise(shape, seed, k, 2 * p, dev)
            Nn[k, p] = bsum(eps, eps)[0].cpu()
            dpm = {}
            for sgn in signs:
                e = sgn * eps
                xn = x + sig * e
                for t in tfs:
                    f, finv = TF[t]
                    d = finv(den(f(xn), sig))
                    eh = (xn - d) / sig
                    out[f'A_{t}'][:, k, p] += w * bsum(eh, eh)
                    out[f'C_{t}'][:, k, p] += w * bsum(e.expand_as(eh), eh)
                    if t == 'id': dpm[sgn] = d
                dd = {'id': dpm.get(sgn)}
                for nm, dn in alts.items():
                    d = dn(xn, sig); dd[nm] = d
                    eh = (xn - d) / sig
                    out[f'A_@{nm}'][:, k, p] += w * bsum(eh, eh)
                    out[f'C_@{nm}'][:, k, p] += w * bsum(e.expand_as(eh), eh)
                for u, v in gpairs:
                    g = (dd[u] - dd[v]) / sig
                    out[f'G_{u}_{v}'][:, k, p] += w * bsum(g, g)
            if extra and anti and 'id' in tfs:
                out['V'][:, k, p] = (dpm[1.0] - dpm[-1.0]).square().flatten(1).sum(1) / (4 * sig ** 2)
                out['Bm'][:, k, p] = (x - 0.5 * (dpm[1.0] + dpm[-1.0])).square().flatten(1).sum(1) / sig ** 2
    res = {k: v.float().cpu() for k, v in out.items()}
    res['N'] = Nn
    return res
