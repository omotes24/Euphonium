#!/bin/bash
# build.sh -- typesets paper/main.pdf (pLaTeX: platex -> pbibtex -> dvipdfmx)
#   bash paper/build.sh          typeset only (the generated tables, figures and macros are committed)
#   bash paper/build.sh --gen    regenerate them first with paper/gen/make_paper.py (needs inputs/ and results/, see reproduce.sh)
#   Needs miru2025j.cls and miru2025j.bst of the MIRU 2025 LaTeX template in paper/ (not included, see paper/README.md).
set -euo pipefail
cd "$(dirname "$0")"
for f in miru2025j.cls miru2025j.bst; do
  [ -f "$f" ] || { echo "paper/$f is missing: copy it from the MIRU 2025 LaTeX template (paper/README.md)"; exit 1; }
done
# references in order of citation: the distributed style without its SORT command (the only change)
[ "$(grep -c '^SORT$' miru2025j.bst)" = 1 ] || { echo "paper/miru2025j.bst: expected exactly one SORT line (paper/README.md gives its checksum)"; exit 1; }
{ echo '% miru2025j-unsrt.bst -- miru2025j.bst without SORT, so that references are numbered in order of citation (written by paper/build.sh)'
  sed 's/^SORT$/% SORT removed (order of citation)/' miru2025j.bst; } > miru2025j-unsrt.bst
if [ "${1:-}" = "--gen" ]; then python gen/make_paper.py; fi
rm -f main.aux main.bbl main.blg main.log main.dvi main.fdb_latexmk main.fls main.pdf
platex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -n '^!' -A3 main.log | head -20; exit 1; }
latexmk main.tex >> build.log 2>&1 || { grep -n '^!' -A3 main.log | head -20; exit 1; }
if grep -q 'undefined references\|Citation .* undefined' main.log; then echo 'warning: undefined references (paper/main.log)'; fi
echo "paper/main.pdf: $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/ {print $2}') pages"
