#report 

# Scikit-fuzzy library
Summarise how to use skfuzzy.
- fuzzy variables and terms
- Rules
- Antecedents
    - can be combined with |, & ~ operators (or operator.and_, or_, invert)
    - can be grouped with brackets
    - automf method to create the membership terms
- Consequents
    - A FuzzyVariable, like Antecendent, but can include a defuzzify method used to turn the fuzzy value back into a crisp value
    - can have a list of consequents that are triggered
    - can be weighted (not used in EFF)
    - automf method to create the membership terms
- ControlSystem
- ControlSystemSimulator
    - add the input values with `sim.inputs(dict)`
    - call compute()
    - get results from `output[cons_name]` for each consequent


# DEAP library
summarise how to use DEAP:
- use of toolbox to register functions
- use of creator to create types/classes
- population creation
- mate and mutate functions
- evaluate
    - returns a tuple
- algorithms - top level loop for running the process
- handling stats and logging
- 

## deap.gp
- strong and loose typed trees
-trees are held as lists of operators in reverse polish order
- primitive sets
    - Types:
        - Primitive
        - Terminal
            - can be a constant, or parameterless function
        - EphemeralConstant
        - input parameters are created automatically as terminals
- gp.compile(tree, pset)
    - compiles a tree to a python function

# fuzzygp.py
- function for creating consequents as EphemeralConstants
- function for creating a primitive set for handling rules, antecedents, consequents
    - creates primitives for Rule, operator.and|or|invert, consequents
- functions for generating rules and rule sets
- function to register the creators with toolbox for a given set of ant and cons.
- function ea_with_elitism_and_replacement
    - run the training loop on data and evolve the population
    - optionally batch data by passing slices to the toolbox.evalute function
    - elitism - keep the top performers from one batch to the next
    - replacement - replace worse performers with newly generated individuals
        - prevent loss of diversity
    - handles printing stats and writing data to tensorboard
    - 
- prune rules
    - remove redundancy in rules to stop them growing too big

# Fuzzybase.py
## FuzzyBase class
- Base class for classifier and gym runner
- stores hyperparams in the init method
- initialise method
    - logs hyperparameters to tensorboard if enabled
    - creates toolbox and initialises it with functions needed by the ea function
        -  create pset (primitiveSet)
        -  select method 
            -  use double tournament to keep rule size down
            -  mate & mutate -> forward to methods on the class
            -  register statistics object used for reporting
-  execute method
    -  handle cold/warm start
    -  run the ea function
    -  write results to tensorboard
-  \_mate method
    -  either swap entire rules or subtrees
- \_mutate method
- pick a rule at random and either replace completely with newly generated one, or replace a random subtree

## Other Functions
- methods/properties to return the best individual or string representation of it
Also functions to create antecedents and binary consequents

# fuzzyclassifier.py
## FuzzyClassifier class
- fit method
    - takes X and y to train on (plus other params)
    - convert X to pandas dataframe if necessary
    - creates antecedents and consequents
    - registers evaluate function bound to X and y
    - handles batching
    - calls FuzzyBase.execute
- predict method
    - compiles rules of best individual

## Functions
- make_predictions
    - filter dataframe to remove unused columns
    - create skfuzzy control system from rules + simulator
    - for each row
        - feed row into the control system as a dict
        - compute the results
        - convert the output into a prediction - pick the consequent with the highest firing value

# gymrunner.py

## GymRunner class
- train method
    - create antecedents from the environment if not supplied
    - create consequents from the environment
    - call base initialise method
    - register evaluate method bound to the environment
    - base.execute
- play method
    - call evaluate method with flag set to display the play
- evaluate method
    - compile rules for an individual
    - create skfuzzy ControlSystem and Simulator from the rules
    - repeatedly play the gym environment until 'done'
        - evaluate action based on current observation and fuzzy rules
        - keep tally of the reward
        -
    - return reward as the fitness

- evaluate_action
    - feed observations into simulator
    - interpret the output either as floats or an integer category
- evaluate_discrete_actions
- evaluate_continuous_actions

# Other Files
## classifier_cv.py
Contains cross_validate function to perform 5-fold cross validation on a fuzzyclassifier and log the results to tensorboard, along with the confusion matrix.

## classify_iris.py, classify_segmentation.py
call classify_cv.cross_validate on the iris data set and segmentation data set.

## gymrunner_testbed.ipynb
Notebook for running different gym environments and logging the result to tensorboard.


# Design Decisions
Document design decisions and why they were made, and perhaps give alternative options.

## Improvements
- pruning
- replace losers
- batching
    - trade-off of more evaluations but faster convergence
- multiprocessing - 4-5 x speedup on 8 cores
- EWMA of the fitness

## Issues and problems
- interface between skfuzzy antecedents/consequents and deap primitives - had to add a `__repr__` to get the right string format
- prune columns not covered by any rule to stop skfuzzy barfing
- saving population not working due to classes being created on the fly
- 
