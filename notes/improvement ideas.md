
# Thoughts on improvements and bug fixes

FuzzyClassifier move classes, antecedent terms, columns to the init method
- will keep it more in line with sklearn

In predict, optionally take a number of members to use for the prediction
        - generate prediction for of the n top members in turn and take the one with the most votes.  As a tiebreaker take the one that the highest rated member went for
        - maybe do the same for GymRunner in play mode
        - Thought:  perhaps an easy way of combining several top individuals is to concatentate all the rules together.  Also maybe weight them by updating the consequents to add the fitness value of the individual as the consequent weight.

or batch and RL, have a weighted exponential average of the previous scores, so if it does well on one batch/session and badly on the next, the previous success will be remembered to some extent. (done)

