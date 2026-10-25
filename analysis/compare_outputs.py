# compare_outputs.py <committed> <recomputed> -- compares the outputs of reproduce.sh with the committed copies
#   JSON: numbers equal up to 1e-9 (relative); TeX: every line except comments, byte for byte; figure PDFs: rendered pixels
#   (pdftoppm, 100 dpi) when poppler-utils is installed, otherwise bytes.  Prints one line per file and exits 1 if any differs.
import os, sys, json, glob, shutil, subprocess, tempfile, math

A, B = sys.argv[1], sys.argv[2]
FILES = (sorted(glob.glob(f'{A}/results/*.json')) + sorted(glob.glob(f'{A}/results/ablations/*.json'))
         + [f'{A}/protocol/history/explore_10nfe.json']
         + [p for p in sorted(glob.glob(f'{A}/paper/*.tex')) if os.path.basename(p) != 'main.tex']   # main.tex is hand-written
         + sorted(glob.glob(f'{A}/paper/figs/fig_*.pdf')))


def same_json(x, y, path=''):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y): return f'{path}: keys differ'
        for k in x:
            r = same_json(x[k], y[k], f'{path}/{k}')
            if r: return r
        return ''
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y): return f'{path}: lengths differ'
        for i, (u, v) in enumerate(zip(x, y)):
            r = same_json(u, v, f'{path}[{i}]')
            if r: return r
        return ''
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if x == y or (math.isfinite(x) and math.isfinite(y) and abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))): return ''
        return f'{path}: {x} vs {y}'
    return '' if x == y else f'{path}: {str(x)[:60]} vs {str(y)[:60]}'


def tex_lines(p):
    return [l for l in open(p, encoding='utf-8').read().split('\n') if not l.lstrip().startswith('%')]


def render(p, d):
    subprocess.run(['pdftoppm', '-r', '100', '-png', p, os.path.join(d, 'p')], check=True, stderr=subprocess.DEVNULL)


def pixels(p, q):
    try:
        from PIL import Image, ImageChops
    except ImportError:
        Image = None
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        render(p, d1); render(q, d2)
        f1, f2 = sorted(glob.glob(f'{d1}/p*.png')), sorted(glob.glob(f'{d2}/p*.png'))
        if len(f1) != len(f2): return 'page count differs'
        for u, v in zip(f1, f2):
            if Image is None:
                if open(u, 'rb').read() != open(v, 'rb').read(): return 'pixels differ'
                continue
            a, b = Image.open(u).convert('RGB'), Image.open(v).convert('RGB')
            if a.size != b.size: return 'size differs'
            box = ImageChops.difference(a, b).getbbox()
            if box: return f'pixels differ in {box}'
    return ''


bad = 0
for f in FILES:
    rel = os.path.relpath(f, A); g = os.path.join(B, rel)
    if not os.path.exists(g): print('MISSING  ', rel); bad += 1; continue
    if f.endswith('.json'): r = same_json(json.load(open(f)), json.load(open(g)))
    elif f.endswith('.tex'): r = '' if tex_lines(f) == tex_lines(g) else 'lines differ'
    elif shutil.which('pdftoppm'): r = pixels(f, g)
    else: r = '' if open(f, 'rb').read() == open(g, 'rb').read() else 'bytes differ (install poppler-utils to compare pixels)'
    print('same     ' if not r else 'DIFFERENT', rel, r)
    bad += bool(r)
print(f'{len(FILES) - bad} of {len(FILES)} files reproduced' if bad else f'all {len(FILES)} files reproduced')
sys.exit(1 if bad else 0)
