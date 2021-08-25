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

One solution to this problem is to have a machine learning algorithm defined as a set of fuzzy IF-THEN rules that can then be represented in a form understandable to non-specialists.  Fuzzy rules were originally hand written for expert systems but writing and debugging rules by hand is time consuming and error prone.  An alternative it to learn the fuzzy rules from data.  Several ways of doing this have been developed, one of the most successful being through the application of genetic programming (GP).


## Background
### Fuzzy Sets and Fuzzy Inference Systems
People have no problem dealing on a daily basis with fuzzy and imprecise terms and reasoning about them.  For example "light rain", "heavy rain" and "downpour" have no precise definition and there is no exact point where a change in precipitation moves from one category to another, but people understand and generally agree on what the terms mean.    In 1965 Lofti Zadeh [@Zadeh-Fuzzy-1965] introduced fuzzy logic and fuzzy set theory, which allowed for representing and reasoning about the sets using the kind of imprecise terms used in human language.  Zadeh showed that you could represent these kinds of fuzzy sets by a membership function that maps how strongly an item belongs to the set into a real value between 0 and 1.  Ten years later [@zadehConceptLinguisticVariable1975] he added the idea of mapping fuzzy membership functions to linguistic variables.  For example in english we refer to people as being "short", "average", "tall", "very tall" and so on.  These can be modelled by mapping a linear value (a person's height) to one or more fuzzy sets.  A person can have degrees of membership to several "height" sets, ranging from 0 (not a member of this set) to 1 (a full member of the set).  So a person who is a little above average height may have a membership of 0.8 to the "average height" class and 0.1 to the "tall" class.  Unlike probabilities the membership values do not need to sum to 1 and in most cases will not.  The usual boolean set operators such as AND, OR, UNION, INTERSECTION and NOT can be expressed as mathematical functions on the membership functions.  

Figure 1 shows how a person's height could map onto linguistic variables using triangular mapping functions.  Other shapes than triangles can also be used, such as gaussian or trapezoid.

1[Triangle fuzzy membership](images/fuzzy_height.png)
*Figure 1: fuzzy membership functions for height*

A fuzzy inference system (FIS) uses a set of rules composed of fuzzy sets to map one or more inputs into output values.  For example, a potential set of fuzzy rules for the classification of irises could be:

    IF petal-length IS long AND  petal-width IS wide
        THEN virginica is likely
    IF NOT petal-width IS wide 
      AND (sepal-width is narrow OR petal-length IS fairly long) 
      THEN versicolor is likely
    IF petal-width IS narrow AND petal-length IS short 
        THEN setosa is likely
    etc.


Fuzzy inference systems are commonly used for feedback control systems but can also be used for other purposes, such as classification.  Because the rules are expressed in terms of linguistic variables they can be easy for humans to understand.  However writing rules manually is slow and error prone, so a better approach is to learn the rules from the data.  


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