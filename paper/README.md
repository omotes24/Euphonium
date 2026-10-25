# paper

LaTeX source of *Euphonium：拡散モデルの自己回転尤度比による分布外検知* (Japanese; the 26f WIP midterm resume, typeset with
the MIRU 2025 class in one column, 17 pages: Sections 1–5, references, supplementary material A.1–A.9). `main.pdf` is the
compiled version.

## Build

1. Copy `miru2025j.cls` and `miru2025j.bst` of the MIRU 2025 LaTeX template (MIRU 2025 website) into this folder. They are
   not part of this repository (the class is derived from `ipsj.cls`, © Information Processing Society of Japan). We used
   the files with these MD5 checksums:

   ```
   2ec2f9a4b08bd9236bfd547b3392cd1c  miru2025j.cls
   c41aa576f39d99f3df222fdcc3dcbc8a  miru2025j.bst
   ```

2. `bash paper/build.sh` → `paper/main.pdf`. It first writes `miru2025j-unsrt.bst`, the distributed style without its `SORT`
   command (the only change), so that the references are numbered in order of citation. It needs pLaTeX (`platex`, `pbibtex`,
   `dvipdfmx`, `latexmk`; we used TeX Live 2023) with graphicx, xcolor, amsmath, amssymb, booktabs, adjustbox, array and TikZ.

On Overleaf: compiler "LaTeX", main document `main.tex`; upload the two class files and a `miru2025j-unsrt.bst` made as in
`build.sh` (delete the line `SORT` of `miru2025j.bst`), and add `$latex = 'platex'; $bibtex = 'pbibtex';` (the `latexmkrc` here).

## Generated files

Everything with numbers in it is generated from the per-image statistics (`inputs/`, release asset `inputs-v1`) and the
result files (`../results/`) by `gen/make_paper.py`; `bash paper/build.sh --gen` (or `bash reproduce.sh` in the repository
root) regenerates them before typesetting. The committed copies are the ones the PDF was built from.

| file | in the paper |
|---|---|
| `numbers.tex` | every number of the text as a macro (`\nfeG` = 10, `\oGCten` = 91.2, …) |
| `figs/fig_concept.pdf` | Figure 1: likelihood axis (NLL) and orientation axis (exact Λ of the PF-ODE), CIFAR-10 model |
| `figs/fig_shared.tex` | Figure 2 (TikZ): the shared-noise estimate; thumbnails `figs/sh_*.pdf` from `gen/mk_thumbs.py` (needs the CIFAR-10 test images, `data/README.md`) |
| `figs/fig_verify.pdf` | Figure 3: (a) the shared-noise estimate of Λ90 (Ablation: ten scales, 40 passes) against the exact ratio of the PF-ODE, (b) AUROC of every per-scale log-ratio |
| `tab_main.tex` | Table 2: Euphonium and the eight comparison methods on the same images |
| `tab_sign.tex`, `supp_sign.tex` | Table 3, Table A6: sign or magnitude: statistics of the same 10 passes and the R90 prediction difference of the GEPC run. The control with the prediction difference of Euphonium's own 10 passes is "計算中" (in progress) in Section 4; `experiments/mk_gp_jobs.py` writes its jobs, and `gen/make_paper.py` adds the column once `inputs/eu/` exists |
| `supp_select.tex` | Table A3: the Euphonium candidates of the selection (top) and the Ablations (bottom) |
| `supp_variance.tex` | Table A5: what the shared noise removes (three noise seeds) |
| `figs/fig_G.pdf` | Figure A1: integrand, per-scale log-ratios of R90 and R180 with the three atoms of Euphonium, distribution of the score |
| `tab_cross.tex`, `supp_orient.tex` | Tables A7, A8: model × image cross-evaluation; orientation strength of the six IDs |
| `figs/fig_gode.pdf` | Figure A2: Euphonium against the exact ratio, per image |
| `supp_scale.tex` | Table A9: per-scale evidence and the exact ratio |
| `supp_all_a.tex`, `supp_all_b.tex` | Tables A10, A11: all 63 pairs |
| `tab_cifar10.tex`, `tab_ids.tex` | the two halves of Table 2 as separate tables (not used by `main.tex`) |

Written by hand in `main.tex`: Table 1 (related work), Table A1 (models), Table A2 (computation; CPU times from
`../results/bench_cpu.jsonl`, written by `experiments/bench_cpu.py`), Table A4 (its numbers are macros of `numbers.tex`).

Notation in the tables: † = best of the eight comparison methods; light blue and bold = Euphonium ties (within 0.05 points)
or beats the best comparison method; underline = Euphonium is second, below only the best comparison method. Ablations carry
no marks.
