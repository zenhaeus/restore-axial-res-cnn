all: report.tex
	pdflatex report.tex && bibtex report.aux && pdflatex report.tex && pdflatex report.tex

clean:
	$(RM) report.pdf
