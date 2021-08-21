#! /bin/bash
pandoc -f markdown -t latex -o PROJ_KirbyD.pdf --toc --number-sections PROJ_KirbyD.md  --bibliography Bibliography.bib --csl harvard.csl -B preamble.latex
