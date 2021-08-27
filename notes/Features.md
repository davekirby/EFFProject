What are the main features of the project?

Using Fuzzy Logic and GP

Common features:
- can save and load the classifer/gymrunner
    - could also use pickle / joblib directly
- can define antecedents and terms explicitly or have them derived directly from pandas dataframe column names
    - explict is more readable
- control over a lot of hyperparameters
- support for using tensorboard to record statistics via the tensorboardX package.
- can print rule sets in a human readable format
    - example
- can warm start - carry on training an existing classifier
- rule pruning to keep size down

Fuzzy rule classifier that learns through genetic programming.
- compatible with sklearn
    - fit/predict methods
    - optional batch processing

Reinforcement Learning agent with policy defined through fuzzy logic rules and trained through genetic programming.
- currently supports 1D boxes and Discrete spaces. 
