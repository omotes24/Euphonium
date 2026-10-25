# mk_n20_jobs.py -- job list of the single-noise runs N_<tag> (+eps only, no antithetic pair) on the ten finest scales and four
#   orientations (id, R90, R180, R270), so that every single-noise candidate of the Ablations is a subset (protocol/PROTOCOL_20NFE.md;
#   analysis/ablations/grid_candidates.py, frontier.py (C), protocol/history/explore_10nfe.py).  Writes experiments/jobs/N20.txt:
#   part 1, selection data (never part of the main evaluation): ID development images (first 1000 of the fixed permutation) and
#          the extra OOD sets (final split, capped)                      -> runs/feat/N_<tag>/<id>__dev.npz, runs/feat/N_<tag>/<set>.npz
#   part 2, the main-evaluation sets with the final-split selection of FINAL.txt -> runs/feat/N_<tag>/<set>.npz (same names as F_<tag>)
#   usage (repository root): python experiments/mk_n20_jobs.py [--force] && python experiments/runq2.py experiments/jobs/N20.txt <gpu>
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
S16 = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
SIG10 = ','.join('%.6g' % s for s in S16[:10])
TFS = 'id,rot90,rot180,rot270'
RGB = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
POOL_RGB = ['stl10_test', 'gtsrb_test', 'flowers102_test', 'eurosat', 'mnist_rgb32', 'pets_test', 'caltech101', 'food101_test', 'aircraft_test', 'imagenette_val', 'sst2_test']
M = {
    'edmc10':   dict(spec='edm:edm-cifar10-32x32-uncond-vp.pkl', id='cifar10_test', env='', bs=500, sets=RGB, syn=['synth_noise', 'synth_const'], pool=POOL_RGB, cap=5000, pcap=2000),
    'l6svhn':   dict(spec='l6:layer6_svhn_vpsde.pt', id='svhn_test', env='', bs=500, sets=RGB, syn=['synth_noise', 'synth_const'], pool=POOL_RGB, cap=5000, pcap=2000),
    'dpceleba': dict(spec='iddpm:diffpath_celeba32.pt', id='celeba_test', env='', bs=500, sets=RGB, syn=['synth_noise', 'synth_const'], pool=POOL_RGB, cap=5000, pcap=2000),
    'edmc100':  dict(spec='edmsd:edm_cifar100_uncond_vp.pt', id='cifar100_test', env='SR_FP16=0 ', bs=250, sets=RGB, syn=['synth_noise', 'synth_const'], pool=POOL_RGB, cap=5000, pcap=2000),
    'fmnist':   dict(spec='own:models/fmnist_s/final.pt', id='g32_fmnist_test', env='', bs=500, sets=['g32_fmnist_test', 'g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist'],
                     syn=['g32_synth_noise', 'g32_synth_const'], pool=['g32_emnist_letters', 'g32_k49', 'g32_semeion', 'g32_cifar10_gray', 'g32_svhn_gray', 'g32_gtsrb_gray'], cap=5000, pcap=2000),
    'in64':     dict(spec='edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', id='in64_imagenet_test', env='', bs=125,
                     sets=['in64_imagenet_test', 'in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test'], syn=[],
                     pool=['in64_' + s for s in POOL_RGB if s not in ('imagenette_val', 'mnist_rgb32')] + ['in64_mnist_rgb'], cap=2000, pcap=1000),
}


def job(t, data, sel, out, pre=''):
    m = M[t]
    return (f"{m['env']}python experiments/extract2.py --gpu {{gpu}} --model {m['spec']} --data {data} {sel}{pre} --out runs/feat/N_{t}/{out}.npz "
            f"--sigmas {SIG10} --pairs 1 --tfs {TFS} --noanti --bs {m['bs']}")


part1, part2 = [], []
for t, m in M.items():
    part1.append(job(t, m['id'], '--n 1000', m['id'] + '__dev'))
    part1 += [job(t, s, f"--n 0 --complement --cap {m['pcap']}", s) for s in m['pool']]
for t, m in M.items():
    part2.append(job(t, m['id'], f"--n 3000 --complement --cap {m['cap']}", m['id']))
    part2 += [job(t, s, f"--n 3000 --complement --cap {m['cap']}", s) for s in m['sets'] if s != m['id']]
    part2.append(job(t, m['id'], f"--n 3000 --complement --cap {m['cap']}", m['id'] + '__vflip', ' --pre vflip'))
    part2 += [job(t, s, f"--n 0 --complement --cap {m['cap']}", s) for s in m['syn']]
jobs = ['# N20: single-noise runs, written by experiments/mk_n20_jobs.py', '# part 1: selection data (ID development images, extra OOD sets)'] + part1 + ['# part 2: main evaluation, final split'] + part2
path = 'experiments/jobs/N20.txt'
assert '--force' in sys.argv or not os.path.exists(path), path + ' exists (claims of running workers are by line index; --force overwrites)'
os.makedirs('experiments/jobs', exist_ok=True)
open(path, 'w').write('\n'.join(jobs) + '\n')
print('wrote', path, len(part1), 'selection jobs,', len(part2), 'final jobs')
