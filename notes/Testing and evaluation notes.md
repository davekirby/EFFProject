#report 

# Testing strategy
- testing process
    - notebooks for experimenting with the deap and skfuzzy libraries
    - unit tests for low level code
        - test first where possible (sometimes)
    - run classifier on iris dataset
    - run gymrunner against pendulum environment


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

