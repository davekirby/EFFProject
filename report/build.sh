#! /bin/bash
pandoc -f markdown -t latex -o preamble.latex preamble.md
pandoc -f markdown -t latex -o PROJ_KirbyD.pdf \
        --toc --number-sections \
        --bibliography Bibliography.bib --csl harvard.csl \
        -B preamble.latex -s \
        --filter pandoc-include \
        PROJ_KirbyD.md  
