#!/usr/bin/env bash
#
# Regenerate the IAA manuscript PDFs from source.
#
#   ./build_pdfs.sh
#
# Produces:
#   response-letter-to-reviewer.pdf
#   revised_manuscript.pdf
#   SUPP_INFO/IAA_SuppInfo.pdf
#
# LaTeX packages that are required but not installed system-wide are fetched
# into a throwaway tree under $TMPDIR, so no root access is needed and nothing
# is installed globally. Logs are kept outside the source tree.

set -euo pipefail

cd "$(dirname "$0")"

deps_only=no
case "${1:-}" in
    --deps-only) deps_only=yes ;;
    "")          ;;
    *)           echo "usage: $0 [--deps-only]" >&2; exit 2 ;;
esac

FALLBACK_TEXMF="${TMPDIR:-/tmp}/iaa-manuscripts-texmf"
LOGDIR="$FALLBACK_TEXMF/logs"
mkdir -p "$LOGDIR"

missing () { ! kpsewhich "$1" >/dev/null 2>&1; }

# --- 1. Make sure every \usepackage resolves ------------------------------

if missing textgreek.sty || missing siunitx.sty; then
    echo "==> Fetching texlive-lang-greek + texlive-science into $FALLBACK_TEXMF"
    mkdir -p "$FALLBACK_TEXMF/debs"
    (
        cd "$FALLBACK_TEXMF/debs"
        rm -f ./*.deb
        apt-get download texlive-lang-greek texlive-science
        for deb in ./*.deb; do dpkg-deb -x "$deb" "$FALLBACK_TEXMF"; done
    )
fi

if missing pythonhighlight.sty; then
    echo "==> Fetching pythonhighlight.sty from CTAN"
    local_tex="$FALLBACK_TEXMF/usr/share/texlive/texmf-dist/tex/latex/local"
    mkdir -p "$local_tex"
    curl -fsSL \
        https://mirrors.ctan.org/macros/latex/contrib/pythonhighlight/pythonhighlight.sty \
        -o "$local_tex/pythonhighlight.sty"
fi

export TEXINPUTS=".:${FALLBACK_TEXMF}//:${TEXINPUTS:-}"

if [ "$deps_only" = yes ]; then
    echo "    dependencies ready"
    exit 0
fi

# --- 2. Build -------------------------------------------------------------

fail () {
    echo "!! $1 failed. Last 40 lines of $2:" >&2
    tail -n 40 "$2" >&2
    exit 1
}

latex_pass () {   # $1 = directory, $2 = jobname
    echo "    pdflatex $2"
    ( cd "$1" && pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" ) \
        >"$LOGDIR/$2.pdflatex.log" 2>&1 || fail "pdflatex $2" "$LOGDIR/$2.pdflatex.log"
}

build () {        # $1 = directory, $2 = jobname, $3 = yes|no (run bibtex)
    echo "==> $1/$2.pdf"
    latex_pass "$1" "$2"
    if [ "$3" = yes ]; then
        echo "    bibtex $2"
        ( cd "$1" && bibtex "$2" ) \
            >"$LOGDIR/$2.bibtex.log" 2>&1 || fail "bibtex $2" "$LOGDIR/$2.bibtex.log"
        latex_pass "$1" "$2"
        latex_pass "$1" "$2"
    fi
}

build . response-letter-to-reviewer no
build . revised_manuscript yes
build SUPP_INFO IAA_SuppInfo yes

# --- 3. Report ------------------------------------------------------------

status=0
echo
for pdf in response-letter-to-reviewer.pdf revised_manuscript.pdf SUPP_INFO/IAA_SuppInfo.pdf; do
    if [ -f "$pdf" ]; then
        pages=$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages:/{ print $2 }')
        printf '  %-38s %s pages\n' "$pdf" "${pages:-?}"
    else
        printf '  %-38s MISSING\n' "$pdf"
        status=1
    fi
done

for log in "$LOGDIR"/revised_manuscript.pdflatex.log "$LOGDIR"/IAA_SuppInfo.pdflatex.log; do
    if grep -qE '^! |undefined|File .* not found' "$log"; then
        echo "  !! unresolved problems in $(basename "$log"):" >&2
        grep -E '^! |undefined|File .* not found' "$log" >&2
        status=1
    fi
done

floats=$(grep -h 'Float too large for page' "$LOGDIR"/*.pdflatex.log | wc -l)
if [ "${floats:-0}" -gt 0 ]; then
    echo "  note: $floats 'Float too large for page' warning(s); see $LOGDIR"
fi

exit "$status"
