# data

The GPU jobs read uint8 arrays `data/npy/<set>.npy` (`[N,32,32,3]`; `[N,28,28]` for the grey-scale sets before padding;
`[N,64,64,3]` for ImageNet-64). They are not redistributed; the scripts below rebuild them from the public sources, and
[`MD5_data_npy.tsv`](MD5_data_npy.tsv) lists the checksum, shape and dtype of every array we used (it also lists arrays of
our earlier experiments that this paper does not use), so a rebuilt array can be checked:

```bash
cd data/npy && md5sum -c --ignore-missing <(tail -n +2 ../MD5_data_npy.tsv | awk -F'\t' '{print $1"  "$2}')
```

The split of a set: the development images are the first `n` of a fixed permutation (`experiments/core.py: select`, seed 0),
the final images are the rest (`--complement`), optionally capped with a second fixed seed (`--cap`). The main evaluation uses
final images only; `benchmark/common_idx.json` lists the images every method is evaluated on. The rule was selected on the ID
development images and on OOD sets that are not part of the main evaluation (`protocol/PROTOCOL_10NFE.md`).

Run from the repository root, in this order (downloads go to `data/raw/`):

| step | writes | sources |
|---|---|---|
| `python experiments/prep_data.py` | CIFAR-10/100, SVHN, CelebA, Textures (DTD), Tiny-ImageNet val, ImageNet32 val, LSUN and ImageNet (resize; the versions of CSI), Places365 val, `mnist_rgb32` | public URLs in the script; CIFAR-100, DTD and Places365 (256) are read from `$SR_DATASETS` |
| `python experiments/prep_gray.py` | 28×28 grey-scale sets (FashionMNIST, MNIST, KMNIST, notMNIST, Omniglot) | public URLs in the script |
| `python experiments/prep_derived.py` | the 32×32 padded grey-scale sets `g32_*`, the synthetic sets `synth_noise`, `synth_const` (and their grey versions) | derived from the arrays above |
| `python experiments/prep_in64.py` | ImageNet-64: ImageNet-1K (the test list of OpenOOD v1.5), SSB-hard, NINCO, iNaturalist, OpenImage-O, Textures, Places, CelebA (`in64_*`) | OpenOOD v1.5 image lists (`data/openood/benchmark_imglist.zip` of the OpenOOD release) and images under `$SR_DATASETS` |
| `python experiments/prep_newood.py`, `prep_eurosat.py`, `prep_newood3.py` | the OOD sets of the selection data (STL-10, GTSRB, Flowers-102, EuroSAT, Oxford-IIIT Pet, Caltech-101, Food-101, FGVC-Aircraft, Imagenette, Rendered SST-2; EMNIST letters, Kuzushiji-49, SEMEION and grey versions of CIFAR-10, SVHN, GTSRB) | torchvision, Zenodo (EuroSAT), public URLs |
| `python experiments/prep_newid.py` | Tiny-ImageNet train at 32×32 (training data of the snapshot models of A.7) | as above |
| `python experiments/complexity.py <set>` | PNG / JPEG 2000 code lengths of every image (`data/complexity/`), for the IC comparison method | the arrays |

`$SR_DATASETS` (default `data/external/`) holds the sets that the scripts do not download, in this layout:

```
cifar100/cifar-100-python/            CIFAR-100, python version
extra_ood/dtd/                        Describable Textures Dataset
places365_val256/val_256/*.jpg        Places365 validation images, 256x256
openood_official/images_largescale/   OpenOOD v1.5: imagenet_1k/{train,val}, inaturalist/extracted, openimage_o/extracted
openood_official/images_classic/      OpenOOD v1.5: texture, ... (classic benchmark)
ood_benchmarks/ssb_hard/, ood_benchmarks/ninco/NINCO/NINCO_OOD_classes/
```

The datasets keep their own licences.
