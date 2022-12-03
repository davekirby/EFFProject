#! /bin/bash
pandoc -f markdown -t latex -o Project_Proposal.pdf --toc --number-sections Project_Proposal.md  --bibliography Bibliography.bib --csl harvard.csl 
