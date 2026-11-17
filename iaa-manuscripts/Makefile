# Regenerate the IAA manuscript PDFs.
#
#   make            all three PDFs
#   make letter     response letter only
#   make manuscript manuscript only
#   make supp       supporting information only
#   make clean      remove LaTeX intermediates (keeps the PDFs)
#   make distclean  also remove the generated PDFs
#
# Required LaTeX packages that are not installed system-wide are fetched into
# $(FALLBACK_TEXMF) by ./build_pdfs.sh --deps-only, so no root access is
# needed and nothing is installed globally.

SHELL         := /bin/bash
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:

FALLBACK_TEXMF ?= $(if $(TMPDIR),$(TMPDIR),/tmp)/iaa-manuscripts-texmf
LOGDIR         := $(FALLBACK_TEXMF)/logs
TEXDEPS_STAMP  := $(FALLBACK_TEXMF)/.deps-ok

LETTER   := response-letter-to-reviewer
PAPER    := revised_manuscript
SUPP_DIR := SUPP_INFO
SUPP_JOB := IAA_SuppInfo
SUPP     := $(SUPP_DIR)/$(SUPP_JOB)

LATEX := pdflatex -interaction=nonstopmode -halt-on-error

# Search the fallback tree as well, in case the packages are not installed.
export TEXINPUTS := .:$(FALLBACK_TEXMF)//:$(TEXINPUTS)

CLEAN_EXTS := aux bbl blg log out toc lof lot spl fls synctex.gz

.PHONY: all letter manuscript supp clean distclean

all: letter manuscript supp

letter:     $(LETTER).pdf
manuscript: $(PAPER).pdf
supp:       $(SUPP).pdf

# run a command, keeping its output in $(2) and showing the tail on failure
# $(1) = command, $(2) = logfile
define run
	@$(1) > $(2) 2>&1 || { \
		echo "!! '$(1)' failed. Last 40 lines of $(2):" >&2; \
		tail -n 40 $(2) >&2; exit 1; }
endef

$(LETTER).pdf: $(LETTER).tex | $(TEXDEPS_STAMP) $(LOGDIR)
	@echo "==> $@"
	$(call run,$(LATEX) $(LETTER).tex,$(LOGDIR)/$(LETTER).pdflatex.log)

$(PAPER).pdf: $(PAPER).tex ref.bib $(wildcard FIGURES/*) | $(TEXDEPS_STAMP) $(LOGDIR)
	@echo "==> $@"
	$(call run,$(LATEX) $(PAPER).tex,$(LOGDIR)/$(PAPER).pdflatex.log)
	$(call run,bibtex $(PAPER),$(LOGDIR)/$(PAPER).bibtex.log)
	$(call run,$(LATEX) $(PAPER).tex,$(LOGDIR)/$(PAPER).pdflatex.log)
	$(call run,$(LATEX) $(PAPER).tex,$(LOGDIR)/$(PAPER).pdflatex.log)

$(SUPP).pdf: $(SUPP).tex $(SUPP_DIR)/ref.bib $(wildcard $(SUPP_DIR)/FIGURES/*) | $(TEXDEPS_STAMP) $(LOGDIR)
	@echo "==> $@"
	$(call run,cd $(SUPP_DIR) && $(LATEX) $(SUPP_JOB).tex,$(LOGDIR)/$(SUPP_JOB).pdflatex.log)
	$(call run,cd $(SUPP_DIR) && bibtex $(SUPP_JOB),$(LOGDIR)/$(SUPP_JOB).bibtex.log)
	$(call run,cd $(SUPP_DIR) && $(LATEX) $(SUPP_JOB).tex,$(LOGDIR)/$(SUPP_JOB).pdflatex.log)
	$(call run,cd $(SUPP_DIR) && $(LATEX) $(SUPP_JOB).tex,$(LOGDIR)/$(SUPP_JOB).pdflatex.log)

$(TEXDEPS_STAMP): build_pdfs.sh
	@./build_pdfs.sh --deps-only
	@touch $@

$(LOGDIR):
	@mkdir -p $@

clean:
	@rm -f $(foreach e,$(CLEAN_EXTS),$(LETTER).$(e) $(PAPER).$(e) $(SUPP).$(e))
	@echo "cleaned LaTeX intermediates"

distclean: clean
	@rm -f $(LETTER).pdf $(PAPER).pdf $(SUPP).pdf
	@echo "removed generated PDFs"
