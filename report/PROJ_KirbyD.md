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

![Triangle fuzzy membership](images/fuzzy_height.png)
*Figure 1: fuzzy membership functions for adult human height*

A fuzzy inference system (FIS) uses a set of rules composed of fuzzy sets to map one or more inputs into output values.  For example, a potential set of fuzzy rules for the classification of irises could be:

    IF petal-length IS long AND  petal-width IS wide
        THEN virginica is likely
    IF NOT petal-width IS wide 
      AND (sepal-width is narrow OR petal-length IS long) 
      THEN versicolor is likely
    IF petal-width IS narrow AND petal-length IS short 
        THEN setosa is likely
    etc.


Fuzzy inference systems are commonly used for feedback control systems but can also be used for other purposes, such as classification.  Because the rules are expressed in terms of linguistic variables they can be easy for humans to understand.  However writing rules manually is slow and error prone, so a better approach is to learn the rules from the data.  


### Genetic Programming

Genetic programming (GP) is part of a family of algorithms inspired by the process of natural evolution,  known collectively as Evolutionary Computing (EC).   These algorithms include genetic algorithms,  evolutionary strategies,  particle swarm optimisation,  differential evolution and many others.  All EC algorithms work by creating a population that randomly samples possible solutions in the problem space, then combines attributes of the best individuals to create a new population. They generally hold the following features in common and differ in implementation and emphasis:

-  codification of a potential solution to a problem into some data structure.  This  is often referred to as their chromosome or genotype
-  a population of individuals whose chromosomes are  initially randomly generated
 - a fitness function that evaluates an individual's chromosome To see how well it fits as a solution to the problem
- a method of selecting individuals from the population according to their fitness
- mutation -  the chromosome of a selected individual may be randomly modified
- crossover  or mating -  parts of the chromosomes of two selected individuals are swapped over to make new individuals

The evolutionary computing algorithms generally work as follows:
```
an initial population is randomly generated
for N generations:
    evaluate each individual's fitness

    randomly select individuals with a bias towards fitter individuals

    randomly leave them unchanged, mutated or mated with another selected individual 
        selected individuals are replaced by the offspring
```


Evolutionary computing algorithms have been applied to many problem domains, and are particularly useful for solving problems that are mathematically intractable, for example because the derivative cannot be calculated for back propagation or the problem is NP-complete.  They are also useful for multi-objective problems where they can find a set of solutions that are Pareto optimal.  

Most EC algorithms represent the chromosome as a fixed-length list of values.  For example Genetic Algorithms (GA) [@Holland:1975] represents the chromosome as a bit string, with mutation done by flipping a randomly selected bit and crossover done by swapping the bits of two individuals that are between random start and end points.   Evolutionary Strategy uses a list of floating point values for the chromosome and mutates by adding a value from a gaussian distribution, with crossover not being used at all.   Using a fixed length linear representation works for some problems, but there are limitations on what it can encode.

Genetic Programming [@Koza92geneticprogramming] avoids this limitation by encoding the chromosome as a tree that can vary in size as it evolves.  The tree is usually used to represent a computer program or mathematical expression, but it can also be used to represent other complex structures such as circuit diagrams [@DesignAnalogCircuits].  

To mutate a tree a random node is selected and the subtree from that node is replaced by a new randomly generated tree.  Crossover is done by selecting a random node in each parent and swapping them over to create two new individuals.   Figure 2 shows an example of two trees representing the expressions $sin(x+3)$ and $sqrt(log(y * 10))$.  After crossover the two offsprings represent the expressions $sin(x + log(y*10))$ and $sqrt(3)$.  


![Crossover in GP](images/crossover.png)
*Figure 2: Example of crossover in Genetic Programming*


# Project Objectives and Results Summary

## Objectives

The objective of the project was to implement a library for evolving fuzzy logic rules through Genetic Programming with application in two domains:

1. Classification of data sets

2. Reinforcement Learning with the OpenAI Gym RL platform.

A second objective was to explore how well the generated fuzzy Logic system performed for different tasks and investigate ways of improving its performance.

## Results Summary

All the objectives of the project were achieved.  The main entry points to the code are a class called FuzzyClassifier for classification of datasets and one called GymRunner to learn to play openAI Gym environments.


### FuzzyClassifier

A FuzzyClassifier class was created that could learn a set of fuzzy rules from a dataset that could then be used to predict the classification of unseen data.  It was implemented in the style of scikit-learn classifiers, with hyperparameters specified in the  `__init__` method, the  `fit` method trains the classifier and the `predict` method makes predictions.   

It was found that the classifier worked well on small data sets with two or three classes, but for large datasets the training time could be prohibitive and the accuracy was found to be poor when tried on a data set with seven classes.  However this could be an artifact of the particular dataset used since the No Free Lunch theorem [@wolpertNoFreeLunch1997] says that no algorithm is suitable for all data sets, so the poor performance could be an artifact of that particular dataset rather than the number of classes.  Further work would be needed to determine if it performs as badly on other multi-class problems.

A unique feature of the classifier is the ability to show the top performing rule set as human readable text.  For example on the Wisconsin breast cancer dataset, the following set of rules generated during 5-fold cross validation scored 94.8% accuracy in predicting benign and malignant outcomes on the test data:

```
IF NOT-Single_Epi_Cell_Size[low] THEN [malignant[likely], benign[unlikely]] 
IF Cell_Shape_Uniformity[low] THEN [malignant[likely], benign[likely]] 
IF Cell_Shape_Uniformity[very_low] THEN [benign[likely], malignant[unlikely]] 
IF Mitoses[high] THEN benign[unlikely] 
IF Bare_Nuclei[very_high] THEN benign[unlikely] 
IF Cell_Size_Uniformity[high] AND Single_Epi_Cell_Size[high] THEN malignant[unlikely]
```

### GymRunner 

A class called GymRunner was created that could learn to play some reinforcement learning environments in openAI Gym.  It was found to be able to master some simple environments such as cartpole (balancing a 2D pole on a moveable cart) in as few as ten generations.  For LunarLander, a more complex environment, it took several hundred generations to be able to repeatably get a good score.  A video demonstration of it playing LunarLander is available at https://youtu.be/Oo6hulwqr9M.

More information about the exploration of the FuzzyClassifier and GymRunner performance against different data sets and gym environments can be found in the Testing and Evaluation section later in this report, along with steps taken to improve the performance.

# Architecture and Design

## Third Party Libraries

The following third party libraries were chosen for use in the project:

### DEAP for genetic programming

DEAP [@DEAP_JMLR2012] is the defacto standard library for evolutionary computation in Python, supporting a wide range of algorithms such as Genetic Algorithms, Genetic Programming, Particle Swarm Optimisation, and Evolution Strategy.  Other evolutionary libraries that were investigated were found to be toy projects not intended for production use, did not support Genetic Programming, or were tailored to a specific application of GP such as symbolic regression.

DEAP is a toolbox of components for implementing evolutionary systems, rather than out-of-the-box ready made algorithms such as those provided by frameworks such as scikit-learn.  This makes the learning curve steeper, but gives great flexibility in the problems it can be applied to.  

The main components of DEAP are:

1. `deap.creator.create` function for creating new types.  This is used to define the custom classes used by the application, for example the type of an individual.

2. `deap.base.Toolbox` is used to register parameterised functions.  For example
   ```python
   toolbox = deap.base.Toolbox()
   toolbox.register("select", deap.tools.selTournament, tournsize=3)
   ```
   creates the attribute "select" on the Toolbox instance that is the `deap.tools.selTournament` function with its `tournsize` parameter bound to the value 3.  When the Toolbox instance is passed to the main evolution algorithm it will expect certain functions to be defined on the toolbox for it to call.  Other functions that need to be registered typically include "mate", "mutate" and "evaluate" as well as "select" but it depends on the algorithm being used.

3. A library of functions for different ways of mating, mutating and selecting individuals in the population and for running different kinds of evolution algorithms. Using the toolbox these can be combined like lego bricks to produce a huge variety of evolutionary computing solutions.
   
4. The Genetic Programming module.  This is the most complex part of DEAP and is explained in more detail in the next section.

#### The deap.gp module
The `deap.gp` module contains classes and functions for supporting Genetic Programming.  The core components are:

1. The `PrimitiveTree` class encapsulates the tree structure that the genetic programming operations act on.   The tree of primitives is stored in a python list in depth-first order.  Because the the arity of every primitive is known, the tree can be reconstructed from the list when it is compiled.  The `PrimitiveTree` class has methods for manipulating the tree and a `__str__` method converts the tree into the equivalent python code, to be used by the `compile` function. 

2. The `PrimitiveSet` and `TypedPrimitiveSet` classes are used to register the type of nodes that a go into a tree structure.  There are three types:
   - `Primitive`s are functions that take a fixed non-zero number of arguments and return a single result.  These form the non-leaf nodes of the tree.
   - `Terminal`s are either constants or functions with no arguments that form the leaves of the tree.  Terminals that are functions are executed every time the compiled tree is run.
   - `EphemeralConstant`s are Terminal functions that are executed once when they are first created and after that always return the same value.  These are used for example to generate a random value that is then used as a constant.
  
   A `PrimitiveSet` assumes that the parameter and return types of the primitives and terminals are compatible.   A `TypedPrimitiveSet` requires all the types to be defined when they are registered, and will only build trees where the parameter and return types match.

3. The `compile` function takes an individual tree of primitives and compiles it to a python function.   It does this by first converting it to a string containing a lambda function and then calling `eval` on the string.  Because the function is compiled from a string using `eval` it is necessary that all the primitives and terminals have `__repr__` methods that will result in that object being created when executed.

4. support functions for creating, mutating and mating trees of primitives.

Full documentation for DEAP can be found at https://deap.readthedocs.io/en/master/index.html.


### Scikit-fuzzy for the fuzzy inference system
Several python fuzzy logic libraries were evaluated for the low-level fuzzy inference engine implementation, including:

* fuzzylite python version - https://fuzzylite.com/; https://github.com/fuzzylite/pyfuzzylite
* fuzzylogic - https://github.com/amogorkon/fuzzylogic
* FuzzyLogicToolbox - https://github.com/Luferov/FuzzyLogicToolBox
* FuzzyPy - https://github.com/alsprogrammer/PythonFuzzyLogic
* Simpful - https://github.com/aresio/simpful
* fuzzylab - https://github.com/ITTcs/fuzzylab
* scikit-fuzzy - https://scikit-fuzzy.github.io/scikit-fuzzy/

Several of these were rejected because they were lacking in documentation or unit tests.  Others were rejected because they appeared to be abandoned with no commits for at least two years and in some cases requiring python 2.7 or earlier.

The library chosen was scikit-fuzzy [@warnerJDWarnerScikitfuzzyScikitFuzzy2019], for the following reasons:

* it has reasonable documentation (including docstrings) and unit tests
* it has been updated in the last six months, so is still in active development
* it supports rules having multiple output variables, for example the rule

  `IF NOT-petal_width[average] AND petal_length[long] THEN [setosa[unlikely], versicolor[unlikely]]`

  can set a fuzzy output value for both setosa and versicolor.  Most of the libraries would require this to be written as two rules.
* a rule in scikit-fuzzy is created using python operators "&", "|" and "~" on objects. Although this may be a drawback when creating rules by hand since it is less readable than the text-based approach used by many other libraries, it fits in well with the way that the DEAP gp module defines chromosomes as a tree of python functions and objects.  


Scikit-fuzzy has a low level API that provides functions for creating and manipulating fuzzy variables, and a high-level API that provides fuzzy rules and an inference engine that evaluate the rules against the user's data.   The high-level components of scikit-fuzzy are:

1. The `Antecedent` class represents an input fuzzy variable.  It is created with a name string and a "universe of discourse" - a numpy linspace array that defines the range of values it can take.
2. The  `Consequent` class represents an output fuzzy variable and is defined in the same way as an `Antecedent` with the addition of a defuzzification method.   Several defuzzification methods are available such as centroid, bisector and mean of maximum.
3. Once a fuzzy variable has been created it needs to have terms defined on it.  A term is a membership function over part of the universe of discourse.  The `Antecedent` and `Consequent` classes have an `automf` function that will automatically create overlapping triangular membership functions over the universe of discourse.  The user may provide a list of names for the terms, or default names may be used.  In the above example "petal_width" would be an `Antecedent` and the terms could be "very_narrow",  "narrow" "average", "wide", "very_wide".  Similarly "setosa" would be a `Consequent` with terms "likely" and "unlikely".
4. `Antecedent` terms can combined to create a new compound term using python operators "&", "|" and "~" on objects, for the AND, OR and NOT fuzzy operators.  
5. A `Rule` class is created with an `Antecedent` term expression and a list of `Consequent` instances.
  For example the above rule would be created with:
  ```python
  Rule(
      ~petal_width['average'] & petal_length['long'], 
      [setosa['unlikely'], versicolor['unlikely']]
  )
  ```
6. To run the Fuzzy Inference System a `ControlSystem` instance is created a with a list of rules, and a `ControlSystemSimulator` is created with the control system as a parameter.  The input data is passed into the `ControlSystemSimulator.input` method and `ControlSystemSimulator.calculate` is called.  The result for each consequent is then available in `ControlSystemSimulator.output["consequent-name"]`.  


### OpenAI Gym Reinforcement Learning Platform
Gym [@brockman2016openai] is a framework for reinforcement learning and a collection of environments for training RL agents.  It provides a simple and flexible API that agents can use to explore and interact with an environment.  The environments supported include classical control problems, 2D and 3D physics engines and classic Atari console video games.  It has become a standard platform for doing RL in python and a range of compatible third party environments are also available.  

The core parts of the API are:

1. The environments are registered with the gym and can be created by passing the environment name to the `gym.make` method, e.g. 
`env = gym.make("CartPole-v1")`
2. the `env.observation_space` defines what an agent can "see" at each step and the `env.action_space` defines what an agent can do at each step.  An agent can interrogate these spaces to determine the range of possible inputs and outputs.
3. The `env.reset()` method set the environment to its initial state and returns an observation of that state. 
4. The `env.step(action)` takes an action provided by the agent and returns a tuple of `(observation, reward, done, info)`.  Where
  - `observation` is information about the updated state of the environment
  - `reward` is the reward (positive or negative) for taking that step
  - `done` is a boolean flag indicating if the run is completed
  - `info` may contain diagnostic information specific to an environment, and should not be used by the agent for learning
5. `env.render()` will display the current environment (e.g. one frame of an atari game) and can be used to create a video display of the agent in action.  This is usually omitted during training to speed up the process.

The observation_space and action_space are subclasses of `gym.spaces.Space` and will be one of several types.  The most common are:
1. `Discrete(n)` - the observation or action is a single integer in a range 0-n.
2. `Box(low, high, shape, dtype)` a numpy array of the given shape and dtype where the values are bounded by the high and low values.  This may be a simple linear array or multidimensional.  For example the atari games often have an observation space of `Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)`, where the Box is a 3-dimensional representation of the screen pixels.

Other types of observational space exist, such as dicts or tuples of spaces, but they are rarely used in practice.

### Other third party libraries
Other third-party libraries used are:

1. tensorboardX (https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) is used to optionally write performance data in a format that can be displayed by TensorBoard (https://www.tensorflow.org/tensorboard/)

2. Numpy for general numerical and array manipulation

3. Pandas for handing classification on pandas dataframes.

3. scikit-learn in order to make the FuzzyClassifier class a compatible scikit-learn classifier.

## Design Overview

The code for the project is in a python package `evofuzzy` which contains four modules.

* fuzzygp.py has the low level functions for handling the genetic programming operations and the conversion between DEAP PrimitiveTree instances and scikit-fuzzy rules. The two main entry points are:
  * `register_primitiveset_and_creators` function creates a `TypedPrimitiveSet` initialised with all the primitives and registers it with a deap toolbox, along with functions for creating individuals and complete populations.
  * `ea_with_elitism_and_replacement` is the main evolve / evaluate loop, loosely based on the `eaSimple` function provided by DEAP.   It takes a population and a toolbox instance and will iterate for a fixed number of generations, evaluating each individual with the `toolbox.evaluate` function then mutating and mating them according to their fitness.  It also performs several other services including:
    * handling small batches of data for the classifier
    * parallelising the evaluation of the population members over multiple cores
    * recording statistics about the population, such as mean and best fitness using the DEAP Statistics class.
    * optionally saving the statistics and other information in a format that can be read by TensorBoard.
    * elitism - preserving the best individuals from one generation to the next
    * replacing the worst performing individuals with newly generated ones, to prevent loss of diversity

* fuzzybase.py contains one class, `FuzzyBase`.  This is the base class for the classifier and reinforcement learning implementations and has methods for initialisation of the hyperparameters and logging of statistics, and for executing the `fuzzygp.ea_with_elitism_and_replacement` function to run the evolve/evaluate loop.  It also has methods for saving and loading the state of the class and properties for returning the best performer from the current population, both as a PrimitiveTree and as a human-readable string.  

* fuzzyclassifier.py has a `FuzzyClassifier` class that inherits from `FuzzyBase` and does classification in a manner consistent with scikit-learn classifiers.  it has two public methods in addition to those in the base class:
  * `fit` takes X and y parameters for the training data and ground truth classes and trains the classifier.
  * `predict` takes a X in the same format as it was trained on and predicts the classes.

* gymrunner.py has a `GymRunner` class that inherits from `FuzzyBase` and has two public methods in addition to those in the base class:
  * `train` takes an openAI Gym environment and trains a population of fuzzy rules to play it
  * `play` takes an openAI Gym environment and plays it with the current top performer, rendering a video of it in action. 

![Design Overview class diagram](images/architecture.png)
*Figure 3: Public API of the evofuzzy package*


# Implementation

This section goes through the steps that were taken to implement the evofuzzy package and the decisions made along the way.  For instructions on how to use the final version see the User Guide in Appendix I.

The implementation was done iteratively over several stages.  The classifier code was developed first then support for reinforcement learning was added.  The development process alternated between exploratory programming in jupyter notebooks and test driven development using a combination of unit tests written with pytest and test scripts running against small data sets such as Fisher's iris dataset or simple gym environments such as cartpole. 

## Stage 1:  Classification with hand-coded fuzzy rules

The first stage focused purely on using fuzzy rules for classification.   Rules for classifying the iris dataset were written by hand after inspecting the distribution of the data.  Only two of the features were used, "petal_width" and "petal_length", each with three terms.  The rules that were written were

```python 
rule1 = ctrl.Rule(petal_width['narrow'] & petal_length['short'], setosa['likely'])
rule2 = ctrl.Rule(petal_width['medium'] & petal_length['medium'], versicolor['likely'])
rule3 = ctrl.Rule(petal_width['wide'] & petal_length['long'], virginica['likely'])
```

A function was written that read each row of a pandas DataFrame containing the iris data and passed it to the fuzzy controller.  The rules were applied to the features and calculated the activation strength of each of the consequents.  The argmax of the activation strengths were used to pick the predicted class.  

This method achieved an accuracy of 86.66%, showing that classification with simple fuzzy rules was viable.  A `FuzzyClassifier` class was created to encapsulate this functionality in a `predict` method.

## Stage 2: Encoding fuzzy rules in DEAP GP trees

The second stage was to create a DEAP primitive set that could be used to generate and manipulate fuzzy rules.  Normally when using the `deap.gp` module, calling `compile` on an expression tree would either produce a function that would then be called with the user data as parameters, or if the expression tree was defined without parameters then to call it directly.  For this project what was needed was for the `deap.gp.compile` function to return a scikit-fuzzy `Rule` instance.   Fortunately in python functions and classes are somewhat interchangeable in that both are callable objects, and calling a class object creates an instance of that object.   

The `evofuzzy.fuzzygp._make_primitive_set` function does the work of defining the primitive set that is at the heart of the evofuzzy code.  This takes a list of scikit-fuzzy `Antecedent` instances and a list of `Consequent` instances and returns a primitive set with the types and functions registered on it for creating fuzzy rules.

A `PrimitiveSetTyped` instance was created that takes no parameters and has the type of the return value set to the `skfuzzy.control.Rule` class and the following primitives are added to it:

- The terms of each of the `Antecedent` instances were added as terminal nodes of type `Term`.  
- 
- the `Rule` class is added as a primitive that has a `Term` and a `list` as parameters, and returns a `Rule` instance.

- functions for the "and" "or" and "not" operators were added, using `operator.and_`, `operator.or_` and `operator.invert` from the standard library.  Each of these are defined as taking two `Term` instances as parameters and returning a new `Term`.  

- for the consequents an ephemeral constant was used that creates a list with a single consequent term selected at random from those available.  Since DEAP can only handle a fixed number of parameters for each primitive this was found to be the simplest way to handle having a list as parameter.  A class is used for the ephemeral constant that has a `__str__` method to return the list as a string that can be compiled as valid python code.  

- an identity function was added as a primitive that takes a list and returns the same list unchanged.  This is so that if mating or mutating tries to modify the consequents list, the only option is to pass it through the identity function, since this is the only function that takes that type.

Once the `_make_primitive_set` function was completed a function was added to create a rule from the primitive set.  The `deap.gp.genGrow` function was used, which generates a random tree from the primitives where the branches may be of different lengths.  Another function was added to create a `RuleSet` that consist of a random number of rules.  The `RuleSet` class is a subclass of `list` that has the `__len__` method overridden to return the sum of the length of all the contained rules. This is necessary because the DEAP library uses `len(individual)` when controlling bloat.  A `length` property was added to return the number of rules.  The `register_primitive_set_and_creators` function was written to create the primitive set and register it on the DEAP toolbox, along with the functions for creating individuals and populations. 


## Stage 3: Adding rule generation to the classifier

The `FuzzyClassifier` class with hand-coded rules created in stage 1 was updated to generate random rules using the functions created in stage 2.  The `__init__` method was added that took the hyperparameters for controlling the tree height and number of rules.  Since the FuzzyClassifier is intended to be compatible with scikit-learn, the `__init__` method is only used for assigning hyperparameters to local variables of the same name, as required in the scikit-learn developer's guidelines (https://scikit-learn.org/stable/developers/develop.html#instantiation).


The `fit(X, y, ...)` method was added that created the DEAP toolbox and registered the primitive set and creation functions.  A helper function was added to generate the scikit-fuzzy `Antecedent` objects.  The upper and lower limits for the fuzzy variable are taken from the min and max values in the X data, while the variable names and terms are either taken from a dictionary passed in by the user or derived from the column names & default terms.  Another helper function was added to create the `Consequent` objects that represent the target classes.   Originally these had a single fuzzy term "likely" that ranged from 0 to 1, but later an "unlikely" term was added that was the inverse function.  This enabled more expressive rules to be created.  

## Stage 4: Adding learning from data to the classifier

At this point the classifier would generate a random rule set and use it to predict the classes, but this clearly performs no better than random guessing.  The next step was to add learning from the training data by creating a population of individuals and implementing the evaluate-evolve cycle.  To do this the core genetic programming operators were added as methods to the class and also registered on the toolbox:

- `evaluate` evaluates an individual by compiling its primitive trees into scikit-learn fuzzy rules and executing them against each row of the input data in turn. The resulting predictions over the entire data set are compared with the actual values in `y` and the accuracy score returned as the fitness value for that individual.  
- `mate` takes two individuals and mates them by randomly selecting a rule from each and swapping over randomly selected subtrees.
- `mutate` takes one individual and mutates it by randomly selecting a rule from it and replacing a subtree with a newly created subtree

A function was implemented in the `fuzzygp` module to run the evaluate-mutate loop - originally called `eaSimpleWithElitism` and based on the version in [@wirsanskyHandsonGeneticAlgorithms2020], it was later rewritten as features were added and renamed to `ea_with_elitism_and_replacement` to conform to the PEP8 naming convention.  The function loops round a fixed number of times given by the `n_iter` hyperparameter.  Each time round the loop it:

- evaluates each member of the population against the training data to calculate their fitness
- creates a new generation by selecting members based on their fitness then randomly mutating or mating them.

The function also optionally prints statistics about the population each generation, including the best and average fitness and sizes.

The original version of the function implemented elitism - preserving the best members of the population between generations - using the DEAP `HallOfFame` class.  This was later changed in favour of sorting the population by fitness each iteration and working with slices of the list, which significantly simplified the code.


## Stage 5: Improving the classifier

At this point I had a successful working classifier - the first attempt at classifying the iris dataset got an accuracy of 88% with a population of 20 over 20 generations- slightly better than I got with my hand-written rules.  However there was plenty of room for improvement - the classifier was very slow, taking around 50 seconds to train on the 150 iris data points, and was slow to converge.

Improvements to the initial classifier were added over several iterations:

### Parallelising the evaluation with multiprocessing

Evolutionary algorithms are "embarrasingly parallel" so a significant speedup was obtained by using a `multiprocessing.Pool` to evaluate the population in parallel with a pool of workers.  The runtime for training on the iris dataset went from about 50 seconds down to 12 seconds on an 8-core linux desktop, for a 4-fold speedup.

### Supporting rules with multiple consequents

The initial implementation only allowed for a single `Consequent` term in a rule, which limited their expressiveness and meant more rules were required to cover all the possible output classes.  The code for creating an ephemeral constant was modified to create a consequent list containing a random selection of the available consequents.  The number of consequents to include was randomly selected to be from 1 to half the number of available consequents, rounded up.


### Using Double Tournaments to reduce bloat

A recurring problem in genetic programming is that the trees tend to grow over successive generations as subtrees are replaced through mutation and mating, a problem known as bloat in the GP literature.  Having lots of large trees are slower evaluate and slower to converge to good solution, since the search space is correspondingly larger.

To reduce bloat the standard tournament selection was replaced with a double tournament as described in [@lukeFightingBloatNonparametric2002].  This selects individuals for the next generation through two rounds of tournaments:

1. A series of fitness tournaments are held where in each round `tournament_size` individuals are selected at random from the population and the fittest is chosen as the winner to go into the next round.
2. a second series of tournaments is held where pairs of candidates from the previous round are selected and the smallest is selected with a probability controlled by the `parsimony_size` hyperparameter.  This is a value between 1 and 2, where 1 means no size selection is done and 2 means the smallest candidate is always selected.  In the paper cited above, values in the range 1.2 to 1.6 were found to work well for their experiments. 


### Adding whole-rule mating and mutating 

At this point mating and mutating only happen on a single branch of one of the rules that make up an individual, which may potentially only make a small change to its fitness.  To enable larger changes to take place the ability to mate or mutate at the level of complete rules was added.  When an individual is selected for mutating there is a probability controlled by the `whole_rule_prob` hyperparameter that an entire rule will be replaced by a newly generated rule.  Similarly when two individuals are selected for mating there is the same probability that they will swap entire rules.  

In practice this has not been found to make much difference to the performance on the data sets that have been studied.

### Adding new individuals to reduce diversity loss

Another problem common in evolutionary algorithms is loss of diversity, where a moderately good genotype outperforms the others and spreads through the population, resulting in convergence on a suboptimal local maxima.   To avoid this each generation a number of new individuals are created and added to the population.  The number to add is controlled by the `replacements` hyperparameter.

### Adding support for TensorBoard

To assist with evaluation and tuning of hyperparameters I added support for writing information to disk in a format that can be displayed by TensorBoard (https://www.tensorflow.org/tensorboard/).  I used the tensorboardX library (https://tensorboardx.readthedocs.io/) to write the data.  The `fit` function was extended to take an optional `tensorboardX.SummaryWriter` instance and this was used to save:
- at the start of a training run:
  - the hyperparameters used for training
- after each epoch:
  - the highest and average fitness of the population
  - the smallest and average size of the individuals
  - the fitness of the entire population as histogram data
  - the size of the entire population as histogram data
  - the number of rules each individual has as histogram data
- at the end of training:
  - the rules of the best individual as human-readable text
  - the size of the best individual

tensorboardX has no dependency on TensorBoard, so it is not necessary to install tensorboard to save the data, only to view it afterwards.

Figures 4 shows an examples of TensorBoard comparing several runs of the classifier on the iris dataset.

![Example TensorBoard display of scalar values](images/tensorboard_1.png)
*Figure 4: example TensorBoard display of scalar values*

Figures 5 shows an examples of TensorBoard displaying histograms of how the fitness and sizes of the entire population changes over the 20 epochs.  

![Example TensorBoard display of histograms of population fitness and sizes](images/tensorboard_2.png)
*Figure 5: example TensorBoard display of histograms of population fitness and sizes*

Figure 6 shows the TensorBoard display of the rules for the best individual after a training run.

![Example TensorBoard display of best individual](images/tensorboard_3.png)
*Figure 6: example TensorBoard display of best individual*


### Adding rule pruning

It was noticed that rules were often created with redundant terms, for example (in pseudocode) "IF NOT(NOT(X)) THEN ...", "IF X AND X THEN ..." and "IF X OR X THEN..." could all be replaced with "IF X THEN ..." without changing the meaning of the expression.  This redundancy was unnecessary bloat that slowed down execution of the rules and contributed nothing to the fitness.  To aleviate this  `_prune_rule` and `_prune_population` functions were added that searched for this kind of redundancy and remove it.  The population is then pruned just before it is evaluated.

### Adding "unlikely" term to consequents

The rules at this stage can only assert that a target class is "likely" which hampers their expressiveness.  Adding a second "unlikely" term to the consequents that was the negation of the "likely" term enabled.  The rules could now express statements such as 

`IF petal_width[wide] THEN [versicolor[likely], setosa[unlikely]]`

![Setosa consequent](images/setosa_consequent.png)
*Figure 7: Setosa consequent fuzzy mapping with "unlikely" term*


### Adding mini-batch learning

Although at this point the classifier learns from the data, it only updates the population after each complete pass through the training data.  This is means that for large datasets the convergence on a good solution is extremely slow.  To resolve this mini-batch learning was added, controlled by an optional `batch_size` hyperparameter.  The implementation was done by converting the `batch_size` into a list of python `slice` objects and passing the list to the  `ea_with_elitism_and_replacement` function.  This list is iterated over in the main loop and each individual is evaluated against the current slice of the data then the next generation is evolved.  The data and ground truth arrays are shuffled before `ea_with_elitism_and_replacement` is called in case the data is organised in order of the class values - that would have resulted the population being trained on batches where the output classes are all the same leading to poor generalisation.

This code change results in much faster convergence since there are far more opportunities for learning.  Previously if the data set had 1000 data points then over 10 epochs the population would have evolved 10 times.  With the batch size set to 100 then it would have evolved 100 times.

An epoch is still considered a complete pass through the data, so may now consist of many generations.  The output of the statistics, both to tensorboard and through print statements, still only happened at the end of each complete epoch to keep the output to a manageable level. 

Figure 8 shows a comparison of the best and mean fitness and size when classifying the data without batching (grey line) and with a batch size of 20 (red line).  It can be seen that with batching the population has reached a better solution after five epochs than the run without batching took afer 20 epochs.  The size of the individuals is also smaller.  The run was done over 100 of the iris data points and the remaining 50 were used for measuring the performance on unseen data.  In this case the version without batching had an accuracy of 78% while the version with batching had an accuracy of 96%.

![Batching comparison](images/batching_comparison.png)
*Figure 8: iris classification with and without batching*

## Stage 6 Adding the GymRunner class for reinforcement learning






### Adding EWMA of fitness values




# Evaluation and Tuning

## Results

# Development Methodology and Schedule

# Conclusion

# References

\newpage
# Appendices

## Appendix I: User Guide

!include ../user_manual.md

\newpage
## Appendix II: Source Code

### requirements.txt
```text
!include ../requirements.txt
```

### requirements-dev.txt
```text
!include ../requirements-dev.txt
```

\newpage
### evofuzzy/fuzzygp.py
```python
!include ../evofuzzy/fuzzygp.py
```

\newpage
### evofuzzy/fuzzybase.py
```python
!include ../evofuzzy/fuzzybase.py
```

\newpage
### evofuzzy/fuzzyclassifier.py
```python
!include ../evofuzzy/fuzzyclassifier.py
```

\newpage
### evofuzzy/gymrunner.py
```python
!include ../evofuzzy/gymrunner.py
```

\newpage
### run_cartpole.py
```python
!include ../run_cartpole.py
```

\newpage
### classifier_cv.py
```python
!include ../classifier_cv.py
```

\newpage
### classify_iris.py
```python
!include ../classify_iris.py
```

\newpage
### classify_segmentation.py
```python
!include ../classify_segmentation.py
```

\newpage
### classify_cancer.py
```python
!include ../classify_cancer.py
```
