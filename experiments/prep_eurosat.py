# prep_eurosat.py -- EuroSAT RGB (27,000 Sentinel-2 patches, 64x64) from the Zenodo record of the data set (the DFKI host used by
#   torchvision does not resolve from the GPU server): eurosat (32x32, prep_data.resize_cc) and in64_eurosat (64x64, prep_in64.crop64 = identity here)
import os, sys, glob, subprocess, zipfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_newood import resize_cc, crop64, save, done, TV
URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
d = f'{TV}/eurosat'; os.makedirs(d, exist_ok=True); z = f'{d}/EuroSAT_RGB.zip'
if not os.path.exists(z):
    subprocess.run(['curl', '-sL', '--fail', '--retry', '5', '-o', z + '.part', URL], check=True); os.rename(z + '.part', z)
if not glob.glob(f'{d}/**/*.jpg', recursive=True):
    with zipfile.ZipFile(z) as zf: zf.extractall(d)
paths = sorted(glob.glob(f'{d}/**/*.jpg', recursive=True))
print('eurosat images', len(paths), paths[:1], flush=True); assert len(paths) == 27000, len(paths)
if not (done('eurosat') and done('in64_eurosat')):
    x32, x64 = [], []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            x32.append(resize_cc(im)); x64.append(crop64(im))
        if i % 5000 == 0: print('eurosat', i, flush=True)
    if not done('eurosat'): save('eurosat', np.stack(x32), (32, 32, 3))
    if not done('in64_eurosat'): save('in64_eurosat', np.stack(x64), (64, 64, 3))
    with open(f'{TV}/eurosat_paths.txt', 'w') as f: f.write('\n'.join(os.path.relpath(p, d) for p in paths))
print('EUROSAT_DONE', flush=True)
