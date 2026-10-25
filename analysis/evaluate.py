# evaluate.py -- main evaluation of Euphonium (rule.json, 10 forward passes per image) on the 63 pairs of the benchmark
#   Scores from the final-split runs F_<tag> (per-image errors of the atoms; the 10-pass implementation euphonium/score.py computes
#   exactly these quantities), AUROC / FPR95 on the images every method has (benchmark/common_idx.json), and the comparison with the
#   eight methods of results/baselines_pairs.json (analysis/baselines_eval.py).  "Best": Euphonium >= the best of the eight minus
#   0.05 points.  Writes results/euphonium_pairs.json.
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import TIERS, NAT, RUN, RESULTS, BASELINES, ID_NAME, feat, first_pair, atom_score, sub, auroc, fpr95, common_idx, set_name
from euphonium import load_rule, describe


def euphonium_scores(tag, name, atoms):
    F = first_pair(feat(RUN['pair'][tag], name)); return atom_score(F, atoms), F['idx']


if __name__ == '__main__':
    rule = load_rule(); atoms = rule['atoms']
    COM = common_idx(); BL = json.load(open(os.path.join(RESULTS, 'baselines_pairs.json')))
    OUT = dict(rule=dict(atoms=[list(a) for a in atoms], nfe=rule['nfe'], noise=rule['noise'], description=describe(atoms)), pairs={}, ids={})
    wins = second = npairs = 0
    for tag, T in TIERS.items():
        si, ii = euphonium_scores(tag, T['id'], atoms); a = sub(si, ii, COM[tag]['__id__'])
        rows = {}
        for s in T['ood']:
            so, io = euphonium_scores(tag, s, atoms); b = sub(so, io, COM[tag][s])
            base = {m: 100 * BL[tag]['common'][m][s]['auroc'] for m in BASELINES if s in BL[tag]['common'][m]}
            best_m = max(base, key=base.get); ranked = sorted(base.values(), reverse=True)
            au = 100 * auroc(a, b)
            r = dict(auroc=au, fpr95=100 * fpr95(a, b), n_id=len(a), n_ood=len(b), best_baseline=best_m, best_baseline_auroc=base[best_m],
                     best=bool(au >= ranked[0] - 0.05), second=bool(au < ranked[0] - 0.05 and au >= ranked[1] - 0.05))
            rows[s] = r; npairs += 1; wins += r['best']; second += r['second']
        nat = NAT[tag]
        mean_eu = float(np.mean([rows[s]['auroc'] for s in nat]))
        mean_base = {m: float(np.mean([100 * BL[tag]['common'][m][s]['auroc'] for s in nat])) for m in BASELINES}
        bm = max(mean_base, key=mean_base.get)
        OUT['pairs'][tag] = rows
        OUT['ids'][tag] = dict(name=ID_NAME[tag], natural_mean=mean_eu, best_baseline=bm, best_baseline_mean=mean_base[bm],
                               best=bool(mean_eu >= mean_base[bm] - 0.05), id_mean_score=float(np.mean(a)), id_sd_score=float(np.std(a)))
        print(f'{ID_NAME[tag]:13s} natural OOD mean AUROC {mean_eu:5.1f}   best of the eight: {bm} {mean_base[bm]:5.1f}   '
              f'pairs best {sum(r["best"] for r in rows.values())}/{len(rows)}', flush=True)
    OUT['summary'] = dict(mean_over_ids=float(np.mean([v['natural_mean'] for v in OUT['ids'].values()])), pairs=npairs, best=wins, second=second,
                          ids_best=sum(v['best'] for v in OUT['ids'].values()))
    print('mean over the six IDs %.1f; best on %d/%d pairs (second on %d); best on %d of 6 IDs' % (OUT['summary']['mean_over_ids'], wins, npairs, second, OUT['summary']['ids_best']))
    json.dump(OUT, open(os.path.join(RESULTS, 'euphonium_pairs.json'), 'w'), indent=1)
    print('saved results/euphonium_pairs.json')
