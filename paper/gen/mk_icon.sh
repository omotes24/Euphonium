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
# intrinsic size x4 (same drawing): GitHub shows an image at most at its intrinsic size, so the README height="40" needs >= 40 px
python3 -c "import re; p='euphonium_icon.svg'; t=open(p).read(); t=re.sub(r'<svg([^>]*?) width=\"([0-9.]+)\" height=\"([0-9.]+)\"', lambda m: '<svg%s width=\"%.3f\" height=\"%.3f\"' % (m.group(1), 4 * float(m.group(2)), 4 * float(m.group(3))), t, count=1); open(p, 'w').write(t)"
rm -rf "$T"
echo "wrote paper/figs/euphonium_icon.svg"
