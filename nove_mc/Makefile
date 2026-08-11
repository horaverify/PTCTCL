# Build the paper.  `make` -> main.pdf ; `make clean` removes the aux files.
MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex $(wildcard src/*.tex)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean
