# bench_cpu.py <method> [B] -- wall-clock seconds and network evaluations per image of every detector of Table A2, on the CIFAR-10
#   network (NVlabs EDM, DDPM++ uncond VP configuration, 56M parameters) with random weights: the compute does not depend on the
#   weights.  CPU, fp32, batches of B images (default 4), all threads of the machine.  Appends one JSON line to results/bench_cpu.jsonl
#   (the committed file is the single session of Table A2: 2 CPU cores, the ten methods in the order of the usage line).
#   methods: euphonium (euphonium/score.py: 10 passes), c3 (the 20-pass grid candidate, Ablation), r60 (Ablation, 60 passes),
#            diffpath, gepc, msma, eigen, nll (and TT), ic (NLL + PNG / JPEG 2000 code lengths), ode (exact ratio: NLL of x and R90 x)
#   usage: for m in euphonium c3 r60 diffpath gepc msma eigen nll ic ode; do python experiments/bench_cpu.py $m 4; done   (needs ext/edm)
import os, sys, json, time, math, io
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, f'{ROOT}/ext/edm'); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import core
core.FP16 = False
from training.networks import EDMPrecond
import baselines as BL
from euphonium import euphonium, load_rule
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count())


class Count:
    """wraps a denoiser and counts network evaluations (images passed through the network)"""
    def __init__(self, den): self.den, self.n, self.sigma_min, self.sigma_max = den, 0, den.sigma_min, den.sigma_max
    def __call__(self, x, sigma): self.n += x.shape[0]; return self.den(x, sigma)


meth = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
net = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, model_type='SongUNet', embedding_type='positional',
                 encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
                 model_channels=128, channel_mult=[2, 2, 2], augment_dim=9, dropout=0.13).eval().requires_grad_(False)
den0 = core.Denoiser(core.edm_precond_fn(net.model, 0.5, fp16=False), 'edmc10'); den = Count(den0)
x = torch.rand(B, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1
SIG10 = np.array([0.01, 0.0171133, 0.0292864, 0.0501187, 0.0857696, 0.14678, 0.251189, 0.429866, 0.735643, 1.25893])
SIG20 = np.exp(np.linspace(np.log(0.004), np.log(10.0), 20))
vjp = [0]


def nll(x, steps=64, smin=0.002, smax=80.0):                      # experiments/nll.py, grid='log', one Rademacher probe
    sig = torch.exp(torch.linspace(math.log(smin), math.log(smax), steps + 1, dtype=torch.float64))
    v = (torch.randint(0, 2, x.shape[1:], generator=torch.Generator().manual_seed(1234)).float() * 2 - 1)[None].expand_as(x)
    def f_and_div(x, s):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = x - den(x, torch.full([x.shape[0]], float(s)))
            gr, = torch.autograd.grad((f * v).sum(), x); vjp[0] += x.shape[0]
        return f.detach(), (gr * v).flatten(1).sum(1).detach()
    logp = torch.zeros(x.shape[0], dtype=torch.float64)
    for k in range(steps):
        s0, s1 = float(sig[k]), float(sig[k + 1]); h = math.log(s1 / s0)
        f0, d0 = f_and_div(x, s0); f1, d1 = f_and_div(x + h * f0, s1)
        x = x + h * 0.5 * (f0 + f1); logp += h * 0.5 * (d0 + d1).double()
    dim = float(np.prod(x.shape[1:]))
    return -0.5 * (x.double() / smax).square().flatten(1).sum(1) - dim * math.log(smax) - 0.5 * dim * math.log(2 * math.pi) + logp


def codelen(x):                                                    # IC: lossless PNG and JPEG 2000 code lengths (bits)
    from PIL import Image
    u = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    out = []
    for im in u:
        for fmt, kw in (('PNG', dict(optimize=True)), ('JPEG2000', dict(irreversible=False))):
            b = io.BytesIO(); Image.fromarray(im).save(b, format=fmt, **kw); out.append(8 * b.tell())
    return out


@torch.no_grad()
def run():
    if meth == 'euphonium': return euphonium(den, x, load_rule())                                 # Ours: 10 passes
    if meth == 'c3':        return core.rails(den, x, SIG10[[0, 2, 4, 6, 8]], 1, ('id', 'rot90'), 0, None, True)   # Ablation: R90, odd scales, pair
    if meth == 'r60':       return core.rails(den, x, SIG10, 1, ('id', 'rot90', 'rot180'), 0, None, True)          # Ablation: 60 passes
    if meth == 'msma':      return [den(x, float(s)) for s in SIG20]
    if meth == 'diffpath':  return BL.diffpath(den, x)
    if meth == 'gepc':      BL.GEPC_SIG = BL.GEPC_SIG[:2]; return BL.gepc(den, x, torch.Generator().manual_seed(0))   # the 2 retained levels
    if meth == 'eigen':     return BL.eigen(den, x, torch.Generator().manual_seed(0), 2, 3, 3)
    if meth == 'nll':       return nll(x)
    if meth == 'ic':        return nll(x), codelen(x)
    if meth == 'ode':       return nll(x), nll(x.rot90(1, (2, 3)))                                 # exact ratio: NLL of x and of R90 x
    raise ValueError(meth)


den0(x[:1], 0.3)                                                   # warm-up
den.n = 0
t = time.perf_counter(); run(); el = time.perf_counter() - t
r = dict(method=meth, batch=B, sec_per_img=el / B, nfe_fwd_per_img=den.n / B, vjp_per_img=vjp[0] / B, threads=torch.get_num_threads())
if meth == 'ic':
    t = time.perf_counter(); codelen(x); r['codelen_sec_per_img'] = (time.perf_counter() - t) / B
print(json.dumps(r), flush=True)
with open(os.path.join(ROOT, 'results', 'bench_cpu.jsonl'), 'a') as f: f.write(json.dumps(r) + '\n')
