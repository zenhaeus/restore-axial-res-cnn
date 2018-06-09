all: presentation.tex
	pdflatex presentation.tex && bibtex presentation.aux && pdflatex presentation.tex && pdflatex presentation.tex

clean:
	$(RM) presentation.pdf
