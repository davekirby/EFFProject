#report 

# Testing strategy
- testing process
    - notebooks for experimenting with the deap and skfuzzy libraries
    - unit tests for low level code
        - test first where possible (sometimes)
    - run classifier on iris dataset
    - run gymrunner against pendulum environment

## Additional tests?
Unit tests are pretty sparse
- rename the file to make it more generic
- add tests for the GP side of things
- add tests for the gymrunner??
- mock out lower level functionality?
- 

# Evaluation
- 5-fold cross validation for classifier 
- save data to tensorboard for later inspection
    - timestamp and dataset/environment in the filename
    - fitness scores (best and average)  
        - accuracy for classifier, total reward for RL
    - sizes size of individuals
    - rule counts
    - hyperparameter settings
    - confusion matrix for classifier
    - text version of best individual
    - histogram for each iteration
        - population fitness
        - rule count
        - population sizes

# Performance
- bloody slow for classification
    - replace skfuzzy with a pure C/C++ library?
    - 89% of time spent in the fuzzy library
- can get stuck in local minima (like most ML libraries)
- speed tolerable for RL
- 

# Results
- show typical results for different environments and classification datasets.

## Classifier
- iris
- segmentation
- wisconsin cancer

## gym
- cartpole
- mountaincar
- mountaincar continuous
- pendulum
- acrobot?
- lunar lander continuous






