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
   creates the attribute "select" on the Toolbox instance that is the `deap.tools.selTounament` function with its `tournsize` parameter bound to the value 3.  When the Toolbox instance is passed to the main evolution algorithm it will expect certain functions to be defined on the toolbox for it to call.  Other functions that need to be registered typically include "mate", "mutate" and "evaluate" as well as "select" but it depends on the algorithm being used.

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

The library chosen was scikit-fuzzy, for the following reasons:

* it has reasonable documentation (including docstrings) and unit tests
* it has been updated in the last six months, so is still in active development
* it supports rules having multiple output variables, for example the rule

  `IF NOT-petal_width[average] AND petal_length[long] THEN [setosa[unlikely], versicolor[unlikely]]`

  can set a fuzzy value for both setosa and versicolor.  Most of the libraries would require this to be written as two rules.
* a rule in scikit-fuzzy is created using python operators "&", "|" and "~" on objects, for the AND, OR and NOT fuzzy operators.  
  For example the above rule would be created in code by instantiating FuzzyVariable objects for petal_width, petal_length, setosa and versicolor and defining the fuzzy terms on them ('long', 'unlikely' etc), then the rule can be defined as 

  ```python
  Rule(
      ~petal_width['average'] & petal_length['long'], 
      [setosa['unlikely'], versicolor['unlikely']
  )
  ```

  Although this may be a drawback when creating rules by hand, it fits in well with the way that the DEAP gp module defines chromosomes as a tree of python functions and objects.  

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

* fuzzyclassifier.py has a `FuzzyClassifier` class that inherits from `FuzzyBase` does classification in a manner consistent with scikit-learn classifiers.  it has two public methods in addition to those in the base class:
  * `fit` takes X and y parameters for the training data and ground truth classes and trains the classifier.
  * `predict` takes a X in the same format as it was trained on and predicts the classes.

* gymrunner.py has a `GymRunner` class that inherits from `FuzzyBase` and has two public methods in addition to those in the base class:
  * `train` takes an openAI Gym environment and trains a population of fuzzy rules to play it
  * `play` takes an openAI Gym environment and plays it with the current top performer, rendering a video of it in action. 

![Design Overview class diagram](images/architecture.png)
*Figure 3: Public API of the evofuzzy package*


# Implementation

# Testing and Evaluation

## Results

# Development Methodology and Schedule

# Conclusion

# References

\newpage
# Appendices

## Appendix I: User Manual

!include ../user_manual.md

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