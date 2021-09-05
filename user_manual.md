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

Also each cycle the best performing individuals are carried over unmodified to ensure that they are not selected out.  The number to carry over is controlled by the `elite_size` parameter.  Also a number of new individuals given by the `replacements` hyperparameter are created to prevent too much loss of diversity.

By default the FuzzyClassifier will run each individual against the entire training data before creating the next generation.  For a large dataset this will be very slow and wasteful, so there is a hyperparameter `batch_size` that will split the input data into small batches and train/evolve the population against each batch in turn.  This can lead to much faster convergence.  However there is a possibility that an individual may do well against one batch and poorly against another so a good performing individual overall may be weeded out by one poor batch.  To counter this there is a `memory_decay` hyperparameter that controls an exponential weighted moving average of the fitness values over successive evaluations.  This is a value between 0 and 1, where 1 (the default) only remembers the most recent fitness value, and 0 only remembers the first fitness value.

The `batch_size` hyperparameter is ignored by the GymRunner class, but the `memory_decay` hyperparameter is used because the individuals may be evaluated against successive randomly initialised environments so may perform well one time and badly another.  

### Selection algorithm {- .unlisted}
evofuzzy uses a double tournament algorithm [@lukeFightingBloatNonparametric2002] when selecting the next generation, to help prevent trees from growing too large (bloat).  The selection is done in two steps:

1. A series of fitness tournaments are held where in each round `tournament_size` individuals are selected at random from the population and the fittest is chosen as the winner to go into the next round.
2. a second series of tournaments is held where pairs of candidates from the previous round are selected and the smallest is selected with a probability controlled by the `parsimony_size` hyperparameter.  This is a value between 1 and 2, where 1 means no size selection is done and 2 means the smallest candidate is always selected.  In the paper cited above, values in the range 1.2 to 1.6 were found to work well for their experiments. 

### Mutating algorithm {- .unlisted}
Individuals in the population are selected for mutation with a probability given by the `mutation_prob` hyperparameter.  An individual that is mutated has a single rule from its set of rules selected and either the entire rule is replaced with a newly generated one, or a sub-tree is selected and replaced with a new subtree.  The probability of the entire rule being replaced is controlled by the `whole_rule_prob` hyperparameter.  

### Mating algorithm {- .unlisted}

Pairs of individuals in the population may also be selected for mating with a probability given by the `crossover_prob` hyperparameter.  A random rule is selected from each parent and either the entire rules are swapped over with a probability of `whole_rule_prob` or a subtree of each rule is selected and swapped over.

# Using evofuzzy {- .unlisted}

evofuzzy provided two classes - `FuzzyClassifier` for classification and `GymRunner` for reinforcement learning on openAI Gym.  They are both subclasses of `FuzzyBase` so have the following in common.

## Common interface {- .unlisted}

### Hyperparameters {- .unlisted}

Both classes are instantiated with the hyperparameters to use during training.  All the hyperparamers are explained in the previous section, but here is a summary:

**min_tree_height** int

minimum height of tree at creation

**max_tree_height** int

maximum height of a tree at creation

**min_rules** int

minimum number of rules of an individual

**max_rules** int

maximum number of rules of an individual

**population_size** int

the size of the population

**n_iter** int

number of times to iterate over the dataset/environment

**mutation_prob** float 0.0 - 1.0

probability of an individual being mutated

**crossover_prob** float 0.0 - 1.0

probability of a pair of individuals being mated

**whole_rule_prob** float 0.0 - 1.0

probability of entire rules being mutated / mated

**elite_size** int

number of top performers to preserve across generations

**replacements** int

number of new individuals to inject each generation

**tournament_size** int

number of individuals to include in a tournament

**parsimony_size** float 1.0 - 2.0

selection pressure for small size

**batch_size** int or None

number of data points to include each generation (FuzzyClassifier only)

**memory_decay** float 0.0 - 1.0

EWMA weighting for new fitness over previous fitness

**verbose** bool

if True print summary stats while running


### Common methods and properties {- .unlisted}

Both classes have these methods and properties in common:

**save(path_to_file)** 

Save the state of the FuzzyClassifier or GymRunner instance to a file.

**load(path_to_file)**

Restore the state of a FuzzyClassifer or GymRunner from a file previously created with the `save` method.

**best** (property)

Get the current best performing individual.  This is a list of list of DEAP GP primitives.  

**best_str** (property)

Return the fuzzy rules of the best performing individual as a human readable string.

**individual_to_str(individual)**

Convert any individual to a human readable string.

**best_n(n)**
Merge the rules of the top `n` individuals into a single rule set.  This is an experimental feature to combine the predictive power of several top performers into a single entity. 


### Other common features {- .unlisted}

Both classes support writing information while training into a format that can be viewed in TensorBoard, by using the TensorBoardX library (https://tensorboardx.readthedocs.io/en/latest/index.html).  If an instance of the `tensorboardX.SummaryWriter` is passed to the training method (`fit` or `train`) then at the end of each epoch statistics about the current best/average fitness and size is saved, plus a histogram of the fitness and size of the entire population.  The hyperparameters for the run are also saved as a text object.  The user may also use the SummaryWriter to save additional information before or after a run if they wish. 

## The FuzzyClassifer class for classification

The FuzzyClassifier class tries to follow the scikit-learn API as far as possible.  The class has the following methods in addition to those in the previous section:

**fit(X, y, classes, antecedent_terms=None, columns=None, tensorboard_writer=None)**


