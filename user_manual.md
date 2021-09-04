# User Manual for the evofuzzy python package {- .unlisted}

`evofuzzy` is a python package for using Genetic Programming (GP) to evolve a Fuzzy Inference System (FIS) that can be used for classification and for reinforcement learning in the OpenAI Gym environment.  It is built on the DEAP evolutionary computation framework and the scikit-fuzzy library. No knowledge of these libraries is needed, but some prior knowledge of FIS and GP will be useful when tuning the system.

# Core concepts and terminology {- .unlisted}

## Fuzzy Variables and Terms {- .unlisted}
A fuzzy variable maps a linear value into a series of linguistic terms, for example if classifying irises then petal length could be represented by a fuzzy variable with the terms "short", "medium", and "long".  

## Fuzzy Rules, Antecedents and Consequents {- .unlisted}

A fuzzy rule is an IF - THEN expression that combines fuzzy terms to generate output fuzzy terms.  The IF part can combine input variables called Antecendents through AND, OR and NOT operators.  The THEN part of the rule specifies one or more output terms, or Consequents.  For classification the consequent terms are "likely" and "unlikely" to represent the likelihood of that class.  

For example

```
IF petal_length[short] AND sepal_length[short] THEN [setosa[likely], virginica[unlikely]]
```

## Fuzzy Inference System (FIS) {- .unlisted}

A Fuzzy Inference System takes a set of fuzzy rules and some input data, evaluates the rules against that data and de-fuzzifies the output to return a crisp (non-fuzzy) result.  This may be a class prediction for classification or actions that a reinforcement learning agent is to perform.

## Genetic Programming  {- .unlisted}

In the `evofuzzy` package the fuzzy rules are encoded as expression trees and individual consists of a list of rules.

### Creating the population {- .unlisted}

When the genetic programming system is run an initial population of individuals is randomly generated. Hyperparameters that control the generation of the population are:

- `population_size` sets the size of the population to create
- `min_rules` sets the minimum number of rules that an individual may have
- `max_rules` sets the maximum number of rules that an individual may have
- `min_tree_height` is the minimum depth of a generated tree
- `max_tree_height` is the maximum depth of a generated tree

The latter four hyperparameters control the size and complexity of the individuals and hence the bias-variance tradeoff and the execution speed.  If an individual has a few small rules it may not have enough complexity to model the relationships in the data and so underfit.  If it has too many rules or they are too large it may overfit the data.  It will also be harder to interpret and run slower.

### The evolution cycle {- .unlisted}
Once the population hs been created, the rule set of each individual is evaluated against the data or RL environment and a fitness score calculated.  A new population is then created by selecting individuals according to their fitness and randomly mutating or mating them.  These are then evaluated again and the process repeated for a number of generations controlled by the `n_iter` hyperparameter.  


### Selection algorithm {- .unlisted}
evofuzzy uses a double tournament algorithm [@lukeFightingBloatNonparametric2002] to help prevent trees from growing too large (bloat).  The selection is done in two steps:

1. A series of fitness tournaments are held where each round `tournament_size` individuals are selected at random from the population and the fittest is chosen as the winner to go into the next round.
2. a second series of tournaments is held where pairs of candidates from the previous round are selected and the smallest is selected with a probability controlled by the `parsimony_size` hyperparameter.  This is a value between 1 and 2, where 1 means no size selection is done and 2 means the smallest candidate is always selected.  In the paper cited above, values in the range 1.2 to 1.6 were found to work well for their experiments. 

### Mutating algorithm {- .unlisted}
Individuals in the population are selected for mutation with a probability given by the `mutation_prob` hyperparameter.  An individual that is mutated has a single rule from its set of rules selected and either the entire rule is replaced with a newly generated one, or a sub-tree is selected and replaced with a new subtree.  The probability of the entire rule being replaced is controlled by the `whole_rule_prob` hyperparameter.  

### Mating algorithm {- .unlisted}

