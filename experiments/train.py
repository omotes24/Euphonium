# train.py -- EDM training of an unconditional (or transform-conditional) diffusion model on a uint8 npy data set
import os, sys, time, copy, json, argparse, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import NPY, ROOT, TF
from unet import build_net

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)             # e.g. cifar10_train
p.add_argument('--out', required=True)              # directory under runs/
p.add_argument('--arch', default='m')
p.add_argument('--calib_n', type=int, default=5000)  # images held out from training (calibration split), fixed permutation
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=4e-4)
p.add_argument('--warm', type=float, default=0.5)    # lr warm-up, Mimg
p.add_argument('--mimg', type=float, default=30.0)
p.add_argument('--snap', default='0.125,0.25,0.5,1,2,4,8,16,32,64')
p.add_argument('--pmean', type=float, default=-1.2)
p.add_argument('--pstd', type=float, default=1.2)
p.add_argument('--dropout', type=float, default=0.13)
p.add_argument('--ema_hl', type=float, default=0.5)  # EMA half-life, Mimg
p.add_argument('--xflip', type=int, default=0)
p.add_argument('--tcond', default='')                # e.g. id,rot90,rot180,rot270 -> transform-conditional model
p.add_argument('--tcond_pid', type=float, default=0.5)
p.add_argument('--fp16', type=int, default=1)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()

dev = torch.device(f'cuda:{a.gpu}')
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
out = f'{ROOT}/runs/{a.out}'; os.makedirs(out, exist_ok=True)

x_all = np.load(f'{NPY}/{a.data}.npy')
if x_all.ndim == 3: x_all = x_all[..., None]
perm = np.random.RandomState(1234).permutation(len(x_all))
calib_idx, train_idx = np.sort(perm[:a.calib_n]), np.sort(perm[a.calib_n:])
data = torch.from_numpy(x_all[train_idx]).to(dev)
calib = torch.from_numpy(x_all[calib_idx][:1000]).to(dev) if a.calib_n > 0 else None
N, res, ch = data.shape[0], data.shape[1], data.shape[3]
tconds = [t for t in a.tcond.split(',') if t]
cfg = dict(arch=a.arch, res=res, ch=ch, dropout=a.dropout, fp16=bool(a.fp16), label_dim=len(tconds), sigma_data=0.5,
           data=a.data, calib_n=a.calib_n, tcond=tconds, args=vars(a))
net = build_net(cfg).to(dev).train()
ema = copy.deepcopy(net).eval().requires_grad_(False)
opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
scaler = torch.amp.GradScaler('cuda', enabled=bool(a.fp16))
nimg, step = 0, 0
latest = f'{out}/latest.pt'
if os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu', weights_only=False)
    net.load_state_dict(ck['net']); ema.load_state_dict(ck['ema']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
    nimg, step = ck['nimg'], ck['step']
    print('resumed at', nimg, flush=True)
snaps = sorted(float(s) * 1e6 for s in a.snap.split(',') if float(s) <= a.mimg)
snaps = [s for s in snaps if s > nimg]
total = int(a.mimg * 1e6)
print(f'data {a.data} train {N} calib {a.calib_n} params {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M cfg {json.dumps({k: v for k, v in cfg.items() if k != "args"})}', flush=True)


def to_f(u):
    return u.permute(0, 3, 1, 2).float() / 127.5 - 1.0


def labels_and_tf(x):
    if not tconds: return x, None
    B = x.shape[0]
    k = torch.where(torch.rand(B, device=dev) < a.tcond_pid, torch.zeros(B, dtype=torch.long, device=dev),
                    torch.randint(1, len(tconds), (B,), device=dev))
    y = torch.empty_like(x)
    for i, t in enumerate(tconds):
        m = k == i
        if m.any(): y[m] = TF[t][0](x[m])
    return y, torch.nn.functional.one_hot(k, len(tconds)).float()


def save(path, full=False):
    d = dict(cfg=cfg, ema=ema.state_dict(), net=net.state_dict(), nimg=nimg, step=step, calib_idx=calib_idx, train_idx=train_idx)
    if full: d.update(opt=opt.state_dict(), scaler=scaler.state_dict())
    torch.save(d, path + '.tmp'); os.replace(path + '.tmp', path)


@torch.no_grad()
def calib_loss():
    if calib is None: return {}
    x = to_f(calib); r = {}
    g = torch.Generator(device='cpu').manual_seed(0)
    eps = torch.randn(x.shape, generator=g).to(dev)
    for s in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
        lab = None
        if tconds: lab = torch.nn.functional.one_hot(torch.zeros(x.shape[0], dtype=torch.long, device=dev), len(tconds)).float()
        d = ema(x + s * eps, torch.full([x.shape[0]], s, device=dev), lab)
        r[s] = float(((d - x) / s).square().mean())
    return r


t0, tl, la, ln, nimg0 = time.time(), time.time(), 0.0, 0, nimg
while nimg < total:
    idx = torch.randint(0, N, (a.bs,), device=dev)
    x = to_f(data[idx])
    if a.xflip: x = torch.where(torch.rand(a.bs, 1, 1, 1, device=dev) < 0.5, x.flip(3), x)
    x, lab = labels_and_tf(x)
    sigma = (torch.randn(a.bs, device=dev) * a.pstd + a.pmean).exp()
    w = (sigma ** 2 + 0.25) / (sigma * 0.5) ** 2
    D = net(x + torch.randn_like(x) * sigma.view(-1, 1, 1, 1), sigma, lab)
    loss = (w.view(-1, 1, 1, 1) * (D - x).square()).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    for g in opt.param_groups: g['lr'] = a.lr * min(nimg / max(a.warm * 1e6, 1e-8), 1.0)
    scaler.unscale_(opt)
    for q in net.parameters():
        if q.grad is not None: torch.nan_to_num(q.grad, nan=0, posinf=1e5, neginf=-1e5, out=q.grad)
    scaler.step(opt); scaler.update()
    nimg += a.bs; step += 1
    hl = min(a.ema_hl * 1e6, nimg * 0.05)
    beta = 0.5 ** (a.bs / max(hl, 1e-8))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.copy_(pn.detach().lerp(pe, beta))
    la += float(loss.detach()); ln += 1
    if step % 500 == 0:
        el = time.time() - tl
        print(f'nimg {nimg / 1e6:.3f}M step {step} loss {la / ln:.4f} img/s {500 * a.bs / el:.0f} elapsed {(time.time() - t0) / 3600:.2f}h', flush=True)
        if not math.isfinite(la / ln): print('NAN loss -- abort', flush=True); sys.exit(1)
        tl, la, ln = time.time(), 0.0, 0
    if step % 4000 == 0: save(latest, full=True)
    if snaps and nimg >= snaps[0]:
        s = snaps.pop(0)
        save(f'{out}/snap_{s / 1e6:g}M.pt')
        print(f'snapshot {s / 1e6:g}M calib eps-loss {calib_loss()}', flush=True)
save(latest, full=True); save(f'{out}/final.pt')
print('TRAIN_DONE', calib_loss(), flush=True)
