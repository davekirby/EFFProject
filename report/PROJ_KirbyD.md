---
title:   "Evolving Fuzzy Forests: creating fuzzy inference systems with Genetic Programming"
subtitle: A dissertation submitted in partial fulfilment of the requirements for the MSc in Advanced Computing Technologies (Data Analytics specialisation)
author:
    - David Kirby
    - Department of Computer Science and Information Systems
    - Birkbeck, University of London
date:  Summer Term 2021
---

# List of Figures and Tables

\newpage
# Introduction

There have been a large number of machine learning algorithms developed over the last few decades and combined with the increase in computing power and availability of large volumes of data they have transformed many areas of our lives.   However their use in some areas such as law, medicine or finance is held back because most of the models are unable to explain how they came to a decision and are effectively black boxes.  

One way round this is to have a machine learning algorithm defined as a set of fuzzy IF-THEN rules that can then be represented in a form understandable to non-specialists.  Fuzzy rules were originally hand written for expert systems but writing and debugging rules by hand is time consuming and error prone.  An alternative it to learn the fuzzy rules from data.  Several ways of doing this have been developed, one of the most successful being through the application of genetic programming (GP).


## Background
### Fuzzy Sets and Fuzzy Inference Systems
In 1965 Lofti Zadeh [@Zadeh-Fuzzy-1965] introduced fuzzy logic and fuzzy set theory, which allowed for representing and reasoning about the sets using the kind of imprecise terms used in human language.  For example "the class of tall men" does not have a precise definition, but people have no problem reasoning about it.  Zadeh showed that you could represent these kinds of fuzzy sets by a membership function that maps how strongly an item belongs to the set into a real value between 0 and 1.  

### Genetic Programming


## Project Objective and Overview of Results



# Architecture and Design

# Implementation

# Testing and Evaluation

## Results

# Development Methodology and Schedule

# Conclusion

# References

\newpage
# Appendices

## Appendix I: User Manual

## Appendix II: Source Code

### requirements.txt
```text
!include ../requirements.txt
```
### requirements-dev.txt
```text
!include ../requirements-dev.txt
```
### fuzzygp.py
```python
!include ../evofuzzy/fuzzygp.py
```
### fuzzybase.py
```python
!include ../evofuzzy/fuzzybase.py
```
### fuzzyclassifier.py
```python
!include ../evofuzzy/fuzzyclassifier.py
```
### gymrunner.py
```python
!include ../evofuzzy/gymrunner.py
```
### gymrunner_testbed.ipynb (move and rename?)
????  how to show ipython file?  convert to markdown first with nbconvert.

### classifier_cv.py
```python
!include ../classifier_cv.py
```

### classify_iris.py
```python
!include ../classify_iris.py
```
### classify_segmentation.py
```python
!include ../classify_segmentation.py
```