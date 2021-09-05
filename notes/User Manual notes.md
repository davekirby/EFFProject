#report 

Write user guide on how to use the library.

# Context
intro to GP and FIS?  Maybe copy from the body of the report and cut down.  Or copy back into the report.

Need to give enough information to make the hyperparams and other params understandable. 



# General (both classifier and RL)

- explain all the hyperparameters

| hyperparameter  | type            | description                                                              |
| --------------- | --------------- | ------------------------------------------------------------------------ |
| min_tree_height | int             | minimum height of tree at creation                                       |
| max_tree_height | int             | maximum height of a tree at creation                                     |
| min_rules       | int             | minimum number of rules of an individual                                 |
| max_rules       | int             | maximum number of rules of an individual                                 |
| population_size | int             | the size of the population                                               |
| n_iter          | int             | number of times to iterate over the dataset/environment                  |
| mutation_prob   | float 0.0 - 1.0 | probability of an individual being mutated                               |
| crossover_prob  | float 0.0 - 1.0 | probability of a pair of individuals being mated                         |
| whole_rule_prob | float 0.0 - 1.0 | probability of entire rules being mutated / mated                        |
| elite_size      | int             | number of top performers to preserve across generations                  |
| replacements    | int             | number of new individuals to inject each generation                      |
| tournament_size | int             | number of individuals to include in a tournament                         |
| parsimony_size  | float 1.0 - 2.0 | selection pressure for small size                                        |
| batch_size      | int or None     | number of data points to include each generation\n(FuzzyClassifier only) |
| memory_decay    | float 0.0 - 1.0 | EWMA weighting for new fitness over previous fitness                     |
| verbose         | bool            | if True print summary stats while running                                |
|                 |                 |                                                                          |

- tensorboard support
    - TensorboardX writer
- fit method
- predict method
- methods to get best rule
- saving and loading
- can use alternative mate and mutate methods by subclassing
- can also override the execute method

# Classifier


# GymRunner

# General advice
- refer to GP handbook.
- start small and increase size (rules, pop etc) as needed
 

