# mk_gp_jobs.py -- job list for the sign-or-magnitude control that the paper reports as pending ("計算中"): Euphonium's 10 passes
#   on the 63 pairs (the common images of benchmark/common_idx.json) with --pred_diff, so that the prediction difference
#   ||T^-1 epshat(T x) - epshat(x)||^2 of every atom (the quantity GEPC-type detectors read as "larger = OOD") comes from exactly the
#   passes that give Euphonium.  Writes experiments/jobs/GP.txt and runs/jobs/gp_idx/*.json; outputs go to runs/eu/<tag>/<set>.npz,
#   which experiments/export.py copies to inputs/eu/ (paper/gen/make_paper.py then adds the column to Tables 3 and A6).
#   usage (repository root): python experiments/mk_gp_jobs.py && python experiments/runq2.py experiments/jobs/GP.txt <gpu>
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
M = {'edmc10': dict(spec='edm:edm-cifar10-32x32-uncond-vp.pkl', id='cifar10_test', env='', bs=500),
     'l6svhn': dict(spec='l6:layer6_svhn_vpsde.pt', id='svhn_test', env='', bs=500),
     'dpceleba': dict(spec='iddpm:diffpath_celeba32.pt', id='celeba_test', env='', bs=500),
     'edmc100': dict(spec='edmsd:edm_cifar100_uncond_vp.pt', id='cifar100_test', env='SR_FP16=0 ', bs=250),
     'in64': dict(spec='edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', id='in64_imagenet_test', env='', bs=125),
     'fmnist': dict(spec='own:models/fmnist_s/final.pt', id='g32_fmnist_test', env='', bs=500)}
COM = json.load(open('benchmark/common_idx.json'))
os.makedirs('runs/jobs/gp_idx', exist_ok=True); os.makedirs('experiments/jobs', exist_ok=True)
jobs = []
for t, m in M.items():
    for s, idx in COM[t].items():
        name = m['id'] if s == '__id__' else s
        data, pre = (name[:-7], ' --pre vflip') if name.endswith('__vflip') else (name, '')
        jp = f'runs/jobs/gp_idx/{t}__{name}.json'; json.dump([int(i) for i in idx], open(jp, 'w'))
        jobs.append(f"{m['env']}python experiments/score_euphonium.py --gpu {{gpu}} --model {m['spec']} --data {data} --idx {jp}{pre} "
                    f"--pred_diff --out runs/eu/{t}/{name}.npz --bs {m['bs']}")
open('experiments/jobs/GP.txt', 'w').write('\n'.join(jobs) + '\n')
print('wrote experiments/jobs/GP.txt', len(jobs), 'jobs')
