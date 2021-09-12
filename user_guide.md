# User Manual for the evofuzzy python package {- .unlisted}

`evofuzzy` is a python package for using Genetic Programming (GP) to evolve a Fuzzy Inference System (FIS) that can be used for classification and for reinforcement learning in the OpenAI Gym environment.  It is built on the DEAP evolutionary computation framework and the scikit-fuzzy library. No knowledge of these libraries is needed, but some prior knowledge of FIS and GP will be useful when tuning the system.

# Core concepts and terminology {- .unlisted}

## Fuzzy Variables and Terms {- .unlisted}
A fuzzy variable maps a linear value into a series of linguistic terms, for example if classifying irises then petal length could be represented by a fuzzy variable with the terms "short", "medium", and "long".  

## Fuzzy Rules, Antecedents and Consequents {- .unlisted}

A fuzzy rule is an IF - THEN expression that combines fuzzy terms to generate output fuzzy terms.  The IF part can combine input variables called Antecedents through AND, OR and NOT operators.  The THEN part of the rule specifies one or more output terms, or Consequents.  For classification the consequent terms are "likely" and "unlikely" to represent the likelihood of that class.  

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

The latter four hyperparameters control the size and complexity of the individuals and hence the bias-variance trade-off and the execution speed.  If an individual has a few small rules it may not have enough complexity to model the relationships in the data and so under-fit.  If it has too many rules or they are too large it may over-fit the data.  It will also be harder to interpret and run slower.

### The evolution cycle {- .unlisted}
Once the population has been created, the rule set of each individual is evaluated against the data or RL environment and a fitness score calculated.  A new population is then created by selecting individuals according to their fitness and randomly mutating or mating them.  These are then evaluated again and the process repeated for a number of generations controlled by the `n_iter` hyperparameter.  

Also each cycle the best performing individuals are carried over unmodified to ensure that they are not selected out.  The number to carry over is controlled by the `elite_size` parameter.  Also a number of new individuals given by the `replacements` hyperparameter are created to prevent too much loss of diversity.

By default the FuzzyClassifier will run each individual against the entire training data before creating the next generation.  For a large dataset this will be very slow and wasteful, so there is a hyperparameter `batch_size` that will split the input data into small batches and train/evolve the population against each batch in turn.  This can lead to much faster convergence.  However there is a possibility that an individual may do well against one batch and poorly against another so a good performing individual overall may be weeded out by one poor batch.  To counter this there is a `memory_decay` hyperparameter that controls an exponential weighted moving average of the fitness values over successive evaluations.  This is a value between 0 and 1, where 1 (the default) only remembers the most recent fitness value, and 0 only remembers the first fitness value.

The `batch_size` hyperparameter is ignored by the GymRunner class, but the `memory_decay` hyperparameter is used because the individuals may be evaluated against successive randomly initialised environments so may perform well one time and badly another.  

### Selection algorithm {- .unlisted}
evofuzzy uses a double tournament algorithm when selecting the next generation, to help prevent trees from growing too large (bloat).  The selection is done in two steps:

1. A series of fitness tournaments are held where in each round `tournament_size` individuals are selected at random from the population and the fittest is chosen as the winner to go into the next round.
2. a second series of tournaments is held where pairs of candidates from the previous round are selected and the smallest is selected with a probability controlled by the `parsimony_size` hyperparameter.  This is a value between 1 and 2, where 1 means no size selection is done and 2 means the smallest candidate is always selected.  Values in the range 1.2 to 1.6 were found to work well for their experiments. 

### Mutating algorithm {- .unlisted}
Individuals in the population are selected for mutation with a probability given by the `mutation_prob` hyperparameter.  An individual that is mutated has a single rule from its set of rules selected and either the entire rule is replaced with a newly generated one, or a sub-tree is selected and replaced with a new sub-tree.  The probability of the entire rule being replaced is controlled by the `whole_rule_prob` hyperparameter.  

### Mating algorithm {- .unlisted}

Pairs of individuals in the population may also be selected for mating with a probability given by the `crossover_prob` hyperparameter.  A random rule is selected from each parent and either the entire rules are swapped over with a probability of `whole_rule_prob` or a sub-tree of each rule is selected and swapped over.

# Using evofuzzy {- .unlisted}

evofuzzy provided two classes - `FuzzyClassifier` for classification and `GymRunner` for reinforcement learning on OpenAI Gym.  They are both subclasses of `FuzzyBase` so have the following in common.


## Common interface {- .unlisted}

### Hyperparameters {- .unlisted}

Both classes are instantiated with the hyperparameters to use during training.  All the hyperparameters are explained in the previous section, but here is a summary:

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

number of data points to include each generation. FuzzyClassifier only, this is ignored by the GymRunner class.

**memory_decay** float 0.0 - 1.0

EWMA weighting for new fitness over previous fitness

**verbose** bool

if True print summary stats while running


### Common methods and properties {- .unlisted}

Both classes have these methods and properties in common:

**save(path_to_file)** 

Save the state of the FuzzyClassifier or GymRunner instance to a file.

**load(path_to_file)**

Restore the state of a FuzzyClassifier or GymRunner from a file previously created with the `save` method.

**best** (property)

Get the current best performing individual.  This is a list of list of DEAP GP primitives.  

**best_str** (property)

Return the fuzzy rules of the best performing individual as a human readable string.

**individual_to_str(individual)**

Convert any individual to a human readable string.

**best_n(n)**
Merge the rules of the top `n` individuals into a single rule set.  This is an experimental feature to combine the predictive power of several top performers into a single entity. 


### Other common features {- .unlisted}

Both classes support writing information while training into a format that can be viewed in TensorBoard, by using the TensorBoardX library (<https://tensorboardx.readthedocs.io/en/latest/index.html>).  If an instance of the `tensorboardX.SummaryWriter` is passed to the training method (`fit` or `train`) then at the end of each iteration statistics about the current best/average fitness and size is saved, plus a histogram of the fitness and size of the entire population.  The hyperparameters for the run are also saved as a text object.  The user may also use the SummaryWriter to save additional information before or after a run if they wish. 


## The FuzzyClassifier class for classification {- .unlisted}

The FuzzyClassifier class tries to follow the scikit-learn API as far as possible.  The class has the following methods in addition to those in the previous section:

**fit(X, y, classes, antecedent_terms=None, columns=None, tensorboard_writer=None)**

Train the classifier on the training data X and y.  The parameters are:

- `X` a pandas DataFrame or numpy-like array of features

- `y` the target data for the classifier

- `classes`:  a dictionary mapping the names of the target class to their values in `y`.  For example, if `y` contains 0, 1, 2 for "setosa, versicolor and virginica respectively then the `classes` parameter should contain `{"setosa": 0, "versicolor": 1, "virginica": 2}`
  **NOTE:** the class names must be valid python identifiers and not python keywords.

- `antecedent_terms`:  an optional dictionary converting feature names to the list of fuzzy terms that will be used for that feature.  For example:
  ```
  {
      'sepal_length': ['short', 'medium', 'long'],
      'sepal_width': ['narrow', 'medium', 'wide'],
      'petal_length': ['short', 'medium', 'long'],
      'petal_width': ['narrow', 'medium', 'wide']
  }
  ```
  This can be used to control the number of terms used for each feature.
  if provided then the keys must match the column names.
  If not provided then the terms will default to "lower", "low", "average", "high", "higher".

- `columns`:  optional feature names to apply to the columns of X.  These must match the keys given in the `antecedent_terms` if provided.
  
  if not provided and X is a pandas DataFrame then the pandas column names will be used.  If not provided and X is a numpy array or similar structure then the column names will default to "column_0", "column_1" etc.

  **NOTE:** the feature names, whether they come from the pandas dataframe or from this parameter, must be valid python identifiers and not python keywords.

- `tensorboard_writer`: an optional `tensorboardX.SummaryWriter` instance to log information for display in TensorBoard.


**predict(X, n=1)**
Predict the target class for the data in X.  

- `X` a DataFrame or numpy array in the same format that the classifier was trained on.
- `n` optional experimental parameter to use the combined top `n` individuals in the population to make the prediction.  By default only the best performer is used.

### Example FuzzyClassifier code: {- .unlisted}
```python
from datetime import datetime
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
from evofuzzy import FuzzyClassifier
import tensorboardX

"""Script for testing the classifier by running it on the iris dataset.
"""

TO_TENSORBOARD = True  # write results and stats to tensorboard?

data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)

train_X, test_X, train_y, test_y = train_test_split(iris, y, test_size=50)


classes = {name: val for (name, val) in zip(data.target_names, range(3))}
antecedent_terms = {
    col: ["very_narrow", "narrow", "medium", "wide", "very_wide"]
    if "width" in col
    else ["very_short", "short", "medium", "long", "very_long"]
    for col in cols
}

if TO_TENSORBOARD:
    logdir = Path(f"tb_logs/iris/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

classifier = FuzzyClassifier(
    population_size=20,
    elite_size=5,
    n_iter=5,
    mutation_prob=0.5,
    crossover_prob=0.5,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=2,
    max_rules=4,
    whole_rule_prob=0.2,
    batch_size=20,
)
classifier.fit(
    train_X,
    train_y,
    classes,
    antecedent_terms=antecedent_terms,
    tensorboard_writer=tensorboard_writer,
)

print(f"Best Rule:  size = {len(classifier.best)}")
print(classifier.best_str)

predictions = classifier.predict(test_X)
confusion = pd.DataFrame(
    data=confusion_matrix(test_y, predictions),
    columns=data.target_names,
    index=data.target_names,
)
print(confusion)
if tensorboard_writer:
    tensorboard_writer.add_text("confusion", confusion.to_markdown())
    tensorboard_writer.close()

```


## The GymRunner class for reinforcement learning {- .unlisted}

The GymRunner class has two methods in addition to the common ones give above.  

**train(env, tensorboard_writer=None, antecedents=None, inf_limit=100.0)**
Train the GymRunner instance to play the OpenAI Gym environment.  The parameters are:

- `env`: the Gym environment created with `gym.make(env_name)`
- `tensorboard_writer`: an optional `tensorboardX.SummaryWriter` instance to log progress to TensorBoard.  
- `antecedents`: an optional list of scikit-fuzzy `Antecedent`s, one for each input variable.  If this is not provided then the antecedents are created automatically from the environment's `observation_space`.  This can be used to give finer control over how the inputs are converted to fuzzy variables, and to give the fuzzy variables meaningful names instead of "obs_0", "obs_1" etc that will be created by default.  See below for the `make_antecedent` helper function.
- `inf_limit`:  Some Gym environments have observation_spaces with lower and upper limits of (-inf, inf) which would cause problems for the fuzzy inference system when the antecedents are created automatically from the observation_space.  This parameter replace +/-inf with +/-`inf_limit`.  It defaults to 100 but that is a quite arbitrary choice so should be set to something appropriate for the environment.  If the `antecedents` parameter is given or the observation space limits are not +/-inf then this parameter has no effect.

**play(env, n=1)**
Show the GymRunner playing the environment.  

- `env`: the Gym environment created with `gym.make(env_name)`
- `n`: experimental parameter to combine the top `n` individuals into a single agent.  By default only the top scoring individual in the population is used.  

This method returns the total reward the agent accrued from playing the environment.


### Helper function {- .unlisted}

**make_antecedent( name, min, max, terms=None)**
This function can be used to create the values for the `antecedents` parameter to the `train` method. 
The parameters are:

- `name`: str the name to give the antecedent.  This must be a usable as a valid python identifier.
- `min`: the minimum value for the antecedent.
- `max`: the maximum value for the antecedent.
- `terms`: an optional list of names for the fuzzy terms. If not provided then they will default to "lower", "low", "average", "high", "higher".


### Example GymRunner code: {- .unlisted}

```python
from datetime import datetime
from pathlib import Path
import tensorboardX
import gym
from evofuzzy import GymRunner
from evofuzzy.fuzzygp import make_antecedent

tensorboard_dir = "tb_logs/cartpole-v0"
if tensorboard_dir:
    logdir = Path(f"{tensorboard_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

env = gym.make("CartPole-v1")
runner = GymRunner(
    population_size=50,
    elite_size=1,
    n_iter=10,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=1,
    max_tree_height=3,
    max_rules=4,
    whole_rule_prob=0.2,
)

antecedents = [
    make_antecedent("position", -2.4, 2.4),
    make_antecedent("velocity", -1, 1),
    make_antecedent("angle", -0.25, 0.25),
    make_antecedent("angular_velocity", -2, 2),
]

runner.train(env, tensorboard_writer, antecedents)
print(runner.best_str)
reward = runner.play(env)
print("Reward:", reward)
```

