
# Thoughts on improvements and bug fixes

FuzzyClassifier move classes, antecedent terms, columns to the init method
- will keep it more in line with sklearn


In predict, optionally take a number of members to use for the prediction
        - generate prediction for of the n top members in turn and take the one with the most votes.  As a tiebreaker take the one that the highest rated member went for
        - maybe do the same for GymRunner in play mode
        - Thought:  perhaps an easy way of combining several top individuals is to concatentate all the rules together.  Also maybe weight them by updating the consequents to add the fitness value of the individual as the consequent weight. (done, except for weight)

or batch and RL, have a weighted exponential average of the previous scores, so if it does well on one batch/session and badly on the next, the previous success will be remembered to some extent. (done)

Add a pre-processing function to gymrunner that can transform the observations to make them more usable.

for classifier, make the metric to use configurable, e.g. F1, AUC etc.  
Also Return predict_proba.  - normalize the strengths of the consequents.


## use the batch_size to optionally control max iterations of the gym.

## Allow consequents to be passed to train method?



## Tasks

- [x] #task Move all `fit` parameters to init method, except X and y ✅ 2021-09-08
- [ ] #task Add more unit tests
- [ ] #task try other classifier datasets
- [ ] #task figure a way of weighting rules by fitness?  
    - how about ratio of current / best fitness?  best will always score 1.
    - needs to handle +ve & -ve values.
        - subtract next-fit fitness and scale by the best 
    - normalize with mean & std of entire population, then shift up to make all +ve.




