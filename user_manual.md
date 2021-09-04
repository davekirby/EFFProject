# User Manual for the evofuzzy python package

`evofuzzy` is a python package for using Genetic Programming (GP) to evolve a Fuzzy Inference System (FIS) that can be used for classification and for reinforcement learning in the OpenAI Gym environment.  It is built on the DEAP evolutionary computation framework and the scikit-fuzzy library. No knowledge of these libraries is needed, but some prior knowledge of FIS and GP will be useful when tuning the system.

# Core concepts and terminology

## Fuzzy Variables and Terms
A fuzzy variable maps a linear value into a series of linguistic terms, for example if classifying irises then petal length could be represented by a fuzzy variable with the terms "short", "medium", and "long".  

## Fuzzy Rules, Antecedents and Consequents

A fuzzy rule is an IF - THEN expression that combines fuzzy terms to generate output fuzzy terms.  The IF part can combine input variables called Antecendents through AND, OR and NOT operators.  The THEN part of the rule specifies one or more output terms, or Consequents.  For classification the consequent terms are "likely" and "unlikely" to represent the likelihood of that class.  

For example

```
IF petal_length[short] AND sepal_length[short] THEN [setosa[likely], virginica[unlikely]]
```

## Fuzzy Inference System (FIS)

A Fuzzy Inference System takes a set of fuzzy rules and some input data, evaluates the rules against that data and de-fuzzifies the output to return a crisp (non-fuzzy) result.  This may be a class prediction for classification or actions that a reinforcement learning agent is to perform.

## Genetic Programming 

In the evofuzzy package the fuzzy rules are represented as expression trees.  An individual consists of a list of rules and when the genetic programming system is run an initial population of individuals is randomly generated. The rule set of each individual is evaluated against the data or RL environment and a fitness score calculated.  A new population is then created by selecting individuals according to their fitness and randomly mutating or mating them.  These are then evaluated again and the process repeated for a number of generations controlled by the **n_iter** hyperparameter.


## Selection algorithm

## Mating algorithm

## Mutating algorithm


