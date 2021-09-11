#report 

It works!
- useful for small datasets that need explainability
    - e.g. medical diagnosis

Viable for RL in simple environments.    Not yet tested in larger environments.

Drawbacks:
- a lot of hyperparameters to tune
    - sloooow
    - may be slow to converge for some problems
    - may not learn at all for some!
    - 

Improvements
- classifier
    - add predict_proba
    - add configurable metric
    - show rule triggering for a given input datapoint
    - 
- GymRunner
    - early stopping
    - 
- both
    - weighted values for top-n predictors
    - demes
    - 