# test_euphonium.py -- checks of euphonium/ with toy denoisers (CPU, no network, a few seconds)
#   python tests/test_euphonium.py      (or: python -m pytest tests/)
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from euphonium import euphonium, load_rule, describe, nfe, scales, shared_noise, sigma_value
from euphonium.transforms import TF

torch.set_default_dtype(torch.float32)
X = torch.rand(3, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 2 - 1      # three images in [-1, 1]


class Count:
    """counts the images passed through the denoiser"""
    def __init__(self, den): self.den, self.n = den, 0
    def __call__(self, y, sigma): self.n += y.shape[0]; return self.den(y, sigma)


def gauss_den(y, sigma):
    """posterior mean E[x0 | y] for x0 ~ N(0, I): rotation-equivariant, so the model has no orientation"""
    return y / (1.0 + sigma ** 2)


W = torch.linspace(0.2, 1.0, 8)[None, None, :, None].expand(1, 3, 8, 8)              # a ramp over the rows: not rotation-equivariant
def ramp_den(y, sigma): return W * y / (1.0 + sigma ** 2)


def test_rule():
    r = load_rule()
    assert r['atoms'] == [('rot90', 0), ('rot90', 6), ('rot180', 0)] and r['noise'] == 'pair'
    assert nfe(r['atoms'], r['noise']) == 10 and scales(r['atoms']) == [0, 6]
    assert describe(r['atoms']) == 'R90: sigma_1, sigma_7; R180: sigma_1'
    assert sigma_value(0) == 0.01 and sigma_value(6) == 0.251189


def test_shared_noise():
    a, b, c = shared_noise((3, 8, 8), 0, 6, 0, 'cpu'), shared_noise((3, 8, 8), 0, 6, 1, 'cpu'), shared_noise((3, 8, 8), 0, 6, 0, 'cpu')
    assert torch.equal(a, -b) and torch.equal(a, c)                                   # antithetic pair, deterministic
    assert not torch.equal(a, shared_noise((3, 8, 8), 0, 0, 0, 'cpu'))                 # another scale, another noise image


def test_forward_passes():
    den = Count(gauss_den)
    out = euphonium(den, X)
    assert out['nfe'] == 10 and den.n == 10 * X.shape[0]


def test_no_orientation_no_score():
    """a rotation-equivariant model gives e_k(Tx) = e_k(x) exactly: Eu = 0 (shared noise, the rotation is a permutation)"""
    out = euphonium(gauss_den, X)
    assert torch.all(out['eu'] == 0) and torch.all(out['log_ratios'] == 0)


def test_against_direct_formula():
    """Eu of a non-equivariant toy model against a direct evaluation of the definition"""
    r = load_rule(); out = euphonium(ramp_den, X)
    ref = torch.zeros(X.shape[0], dtype=torch.float64)
    for T, k in r['atoms']:
        s, eps = sigma_value(k), shared_noise(X.shape[1:], 0, k, 0, 'cpu')
        f, finv = TF[T]
        e_x = sum(((u * eps) - (X + s * u * eps - ramp_den(X + s * u * eps, s)) / s).double().square().flatten(1).sum(1) for u in (1, -1)) / 2
        e_t = sum(((u * eps) - finv((f(X + s * u * eps) - ramp_den(f(X + s * u * eps), s)) / s)).double().square().flatten(1).sum(1) for u in (1, -1)) / 2
        ref += torch.log(e_t) - torch.log(e_x)
    assert torch.allclose(out['eu'], ref, rtol=1e-6, atol=1e-9), (out['eu'], ref)
    assert out['eu'].abs().min() > 1e-4                                                 # the toy model does see orientation


def test_fused_batches():
    a, b = euphonium(ramp_den, X), euphonium(ramp_den, X, fuse=True)
    assert torch.allclose(a['eu'], b['eu'], rtol=1e-6, atol=1e-9)


def test_pred_diff():
    out = euphonium(ramp_den, X, pred_diff=True)
    assert set(out['pd']) == {('rot90', 0), ('rot90', 6), ('rot180', 0)} and set(out['a_up']) == {0, 6}
    assert all(torch.all(v > 0) for v in out['pd'].values())
    assert torch.equal(out['eu'], euphonium(ramp_den, X)['eu'])                         # same passes, same score


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests: t(); print('ok', t.__name__)
    print(len(tests), 'tests passed')
