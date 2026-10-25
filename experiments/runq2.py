# runq2.py -- shared GPU job queue with atomic claims.  usage: python runq2.py jobs.txt <gpu>
# Several workers (one per GPU, started at any time) can consume the same job file; a job is claimed by creating a marker file.
# The job file may be APPENDED to while workers run (never reorder or delete lines: claims are by line index).
import sys, os, subprocess, time

jobs_file, gpu = sys.argv[1], sys.argv[2]
base = os.path.splitext(jobs_file)[0]
claims, logs = base + '_claims', base + '_logs'
os.makedirs(claims, exist_ok=True); os.makedirs(logs, exist_ok=True)
t0 = time.time()
while True:
    jobs = [l.rstrip('\n') for l in open(jobs_file)]
    did = False
    for i, cmd in enumerate(jobs):
        if not cmd.strip() or cmd.lstrip().startswith('#'): continue
        try:
            fd = os.open(f'{claims}/{i:05d}', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'gpu {gpu} {time.strftime("%F %T")}\n'.encode()); os.close(fd)
        except FileExistsError:
            continue
        cmd = cmd.replace('{gpu}', gpu)
        with open(f'{logs}/{i:05d}.log', 'w') as f:
            f.write(cmd + '\n'); f.flush()
            rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        print(f'[{time.time() - t0:7.0f}s] job {i} gpu {gpu} rc={rc} :: {cmd[:150]}', flush=True)
        did = True
        break                                  # rescan from the top so that priority order is respected
    if not did: break
print('WORKER_DONE gpu', gpu, flush=True)
