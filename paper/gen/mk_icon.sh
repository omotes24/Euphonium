#!/bin/bash
# mk_icon.sh -- figs/euphonium_icon.svg, the mark of the README, from figs/euphonium_icon.tex (the TikZ mark left of the title)
#   usage: bash paper/gen/mk_icon.sh        (needs pdflatex with TikZ and pdftocairo from poppler-utils)
set -euo pipefail
cd "$(dirname "$0")/../figs"
T=$(mktemp -d)
printf '%s\n' '\documentclass[border=0.4mm]{standalone}' '\usepackage{tikz}' '\input{euphonium_icon}' \
  '\begin{document}\euphoniumicon\end{document}' > "$T/icon.tex"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$T" "$T/icon.tex" > /dev/null
pdftocairo -svg "$T/icon.pdf" euphonium_icon.svg
rm -rf "$T"
echo "wrote paper/figs/euphonium_icon.svg"
