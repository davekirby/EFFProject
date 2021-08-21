
# Thoughts on improvements and bug fixes

- FuzzyClassifier move  classes, antecedent terms, columns to the init method
    - will keep it more in line with sklearn
    - In predict, optionally take a number of members to use for the prediction
        - generate prediction for of the n top members in turn and take the one with the most votes.  As a tiebreaker take the one that the highest rated member went for
        - maybe do the same for GymRunner in play mode

- for batch and RL, have a weighted exponential average of the previous scores, so if it does well on one batch/session and badly on the next, the previous success will be remembered to some extent.


