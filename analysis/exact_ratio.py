# exact_ratio.py -- Ablation: the exact self-rotation likelihood ratio Lambda_90(x) = log p(x) - log p(R90 x) by the PF-ODE
#   (512 network evaluations per image: Heun with 64 steps, forward and backward pass, for x and for R90 x; experiments/nll.py).
#   Computed for the four IDs with PF-ODE runs of the rotated images (CIFAR-10, SVHN, CIFAR-100, ImageNet-64) on their natural OOD
#   sets, on the images every method has.  Writes results/exact_ratio_pairs.json = {tag: {'id': {...}, 'sets': {set: {auroc, ...}}}}.
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TIERS, NAT, NLLRUN, RESULTS, nll, sub, auroc, common_idx

DIM = {'edmc10': 3072, 'l6svhn': 3072, 'edmc100': 3072, 'in64': 12288}
if __name__ == '__main__':
    COM, OUT = common_idx(), {}
    for tag, run in NLLRUN.items():
        T = TIERS[tag]
        N0, R0 = nll(run, T['id']), nll(run + '_rot90', T['id']); assert (N0['idx'] == R0['idx']).all()
        kid = COM[tag]['__id__']
        llr = sub(N0['ll'] - R0['ll'], N0['idx'], kid)
        d = dict(n_id=len(llr), mean_bpd=float(np.mean(llr) / DIM[tag] / np.log(2)), sd_bpd=float(np.std(llr) / DIM[tag] / np.log(2)),
                 frac_pos=float(np.mean(llr > 0)), sets={})
        for s in NAT[tag]:
            N1, R1 = nll(run, s), nll(run + '_rot90', s)
            if N1 is None or R1 is None: continue
            assert (N1['idx'] == R1['idx']).all()
            have = set(N1['idx'].tolist()); keep = [i for i in COM[tag][s] if i in have]
            llr1 = sub(N1['ll'] - R1['ll'], N1['idx'], keep)
            d['sets'][s] = dict(auroc=auroc(llr, llr1), n_ood=len(llr1), mean_ood_bpd=float(np.mean(llr1) / DIM[tag] / np.log(2)))
        OUT[tag] = d
        print(f'{tag:8s} ID mean {d["mean_bpd"]:.3f} bits/dim  AUROC ' + ' '.join(f'{100 * v["auroc"]:.1f}' for v in d['sets'].values()), flush=True)
    json.dump(OUT, open(os.path.join(RESULTS, 'exact_ratio_pairs.json'), 'w'), indent=1)
    print('saved results/exact_ratio_pairs.json')
