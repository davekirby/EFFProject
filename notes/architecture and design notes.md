#report 

# Architecture
- make classifier sklearn compatible
    - https://scikit-learn.org/stable/developers/develop.html
    - https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    - to make it compatible will need to move some the parameters from fit to init methods.
        - antecedent_terms
        - columns
        - tensorboard_writer
    - add unit test with https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html

 
- reuse code between classifier and RL as much as possible
    - common base class
- use of DEAP and skfuzzy
    - reasons for choice
- use of openai gym

# Choice of libraries
- reasons for the choices
## DEAP overview


## skfuzzy overview
## Gym overview
## TensorboardX overview



## Architecture overview / diagram
- evofuzzy - top level package
    - fuzzygp.py
        - implement the core code for GP and fuzzy logic
            - creating primitive set
            - generating individuals
            - running the GP main loop
                - batches input data, if necessary
                - evaluates population
                    - compiles individual to a list of sklearn rules
                    - calls back to toolbox.evaluate to run rules against the input data
                - multiprocessing to share work across all processors
            - prune rules to keep the size down
            - log results and optionally write to tensorboard
- fuzzybase.py
    - base class for classifier and gymrunner with common functionality
        - FuzzyBase class. 
            - Methods:
                - `__init__`
                - execute
                - initialise
                - mate & mutate
                - properties & methods to get best performer and convert to str
            - all the config & hyperparameters
    - helper functions
        - create antecedents and consequents

- fuzzyclassifier.py
    - FuzzyClassifier class
        - fit method
        - predict method
        - evaluate method

- gymrunner.py
    - GymRunner class
        - train method
        - play method
        - evaluate method
    - helper functions for making antecedents and consequents from a gym environment

Test scripts
 - classifier_cv.py
     - cross_validate function to do 5-fold CV of a fuzzy classifier
 - classify_iris.py
 - classify_segmentation.py
 - run_cartpole.py
 - run_mountain_car.py

Architecture diagrams?
- package/class
- update activity diagram from proposal

---

braindump:  What needs to go in this section?   
- Architectural decisions
    - choice of libraries
    - choice of python?
- overview of how it all holds together
    - system diagram?
    - how to diagram the functions and their interaction with classes?
        - experiment with plantuml
- overview of their key attributes?
    - maybe in the design section?
    - do I need separate sections for architecture and design?



