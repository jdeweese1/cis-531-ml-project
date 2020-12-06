all: clean paper tex appendix


paper:
	pandoc --citeproc --bibliography=paper.bib \
	--variable classoption=twocolumn \
	--variable papersize=a4paper \
	--filter pandoc-include \
	-s paper.md -o hate_speech_ml_paper.pdf

appendix:
	pandoc --variable papersize=a4paper \
	--filter pandoc-include \
	-s all_figures.md -o appendix.pdf

tex:
	pandoc --citeproc --bibliography=paper.bib \
	--variable classoption=twocolumn \
	--variable papersize=a4paper \
	--from markdown+simple_tables \
	-s paper.md -o paper.tex

tex2pdf:
	pdflatex paper.tex

clean:
	rm -f *.pdf *.log *.tex
