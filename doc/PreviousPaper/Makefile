# Build the paper.  `make` -> paper.pdf ; `make clean` removes LaTeX aux files.
PAPER = paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex   # second pass resolves \cref / figure references

.PHONY: clean
clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).synctex.gz

.PHONY: distclean
distclean: clean
	rm -f $(PAPER).pdf
