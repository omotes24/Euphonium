# prep_newood3.py -- a second group of additional out-of-distribution sets.  In this paper they are selection data only
#   (protocol/PROTOCOL_10NFE.md), never part of the main evaluation.
#   Same conventions and helper functions as prep_newood.py (32x32 RGB: shorter side -> 32 + centre crop; 64x64: EDM2/ADM crop;
#   grey: 28x28 luminance zero padded to 32x32).
#     pets_test (Oxford-IIIT Pet, test 3669)       caltech101 (8677, all images, no official split)   food101_test (25250)
#     aircraft_test (FGVC-Aircraft, test 3333; the 20-pixel copyright banner at the bottom is cut before resizing)
#     imagenette_val (3925; ImageNet subset -> 32x32 tiers only)        sst2_test (Rendered SST-2, text images, 1821)
#     in64_<set> for pets, caltech101, food101, aircraft, sst2
#     g32_usps (USPS test 2007, 16x16 -> 28), g32_semeion (1593, 16x16 -> 28), g32_k49 (Kuzushiji-49 test 38547, 28x28)
#   usage: python experiments/prep_newood3.py [pets caltech food aircraft imagenette sst2 usps semeion k49]
import os, sys, traceback, urllib.request
import numpy as np
from PIL import Image
from torchvision import datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_newood import TV, OUT, resize_cc, crop64, gray_pad, done, save, pil_set


def pets():
    ds = datasets.OxfordIIITPet(TV, split='test', target_types='category', download=True)
    pil_set(ds, 'pets_test', 'in64_pets_test')


def caltech():
    ds = datasets.Caltech101(TV, target_type='category', download=True)          # BACKGROUND_Google excluded by torchvision
    pil_set(ds, 'caltech101', 'in64_caltech101')


def food():
    ds = datasets.Food101(TV, split='test', download=True)
    pil_set(ds, 'food101_test', 'in64_food101_test')


def aircraft():
    ds = datasets.FGVCAircraft(TV, split='test', annotation_level='variant', download=True)
    class L:                                                                        # cut the copyright banner (bottom 20 pixels)
        def __len__(self): return len(ds)
        def __getitem__(self, i):
            im = ds[i][0]; w, h = im.size
            return im.crop((0, 0, w, h - 20)), 0
    pil_set(L(), 'aircraft_test', 'in64_aircraft_test')


def imagenette():
    ds = datasets.Imagenette(TV, split='val', size='320px', download=True)
    if done('imagenette_val'): print('exists imagenette_val'); return
    x = [resize_cc(ds[i][0]) for i in range(len(ds))]
    save('imagenette_val', np.stack(x), (32, 32, 3))


def sst2():
    ds = datasets.RenderedSST2(TV, split='test', download=True)
    pil_set(ds, 'sst2_test', 'in64_sst2_test')


def usps():
    if done('g32_usps'): print('exists g32_usps'); return
    ds = datasets.USPS(TV, train=False, download=True)
    save('g32_usps', np.stack([gray_pad(ds[i][0]) for i in range(len(ds))]), (32, 32))


def semeion():
    if done('g32_semeion'): print('exists g32_semeion'); return
    ds = datasets.SEMEION(TV, download=True)
    save('g32_semeion', np.stack([gray_pad(ds[i][0]) for i in range(len(ds))]), (32, 32))


def k49():
    if done('g32_k49'): print('exists g32_k49'); return
    f = f'{TV}/k49-test-imgs.npz'
    if not os.path.exists(f): urllib.request.urlretrieve('http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz', f)
    x = np.load(f)['arr_0'].astype(np.uint8)
    save('g32_k49', np.pad(x, ((0, 0), (2, 2), (2, 2))), (32, 32))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['pets', 'caltech', 'food', 'aircraft', 'imagenette', 'sst2', 'usps', 'semeion', 'k49']
    for t in todo:
        try:
            print('==', t, flush=True); globals()[t]()
        except Exception:
            traceback.print_exc(); print('FAILED', t, flush=True)
    print('PREP_NEWOOD3_DONE', flush=True)
