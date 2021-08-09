# Objective
Re-implement the GP engine to do reinforcement learning against OpenAI Gym.  

## Tasks
- [x] #task Install Gym ✅ 2021-08-05
- [x] #task implement interface between GP engine and Gym ✅ 2021-08-09
    - [x] refactor to reuse code
- [x] #task Choose simple reinforcement task ✅ 2021-08-09
- [x] #task get GP working with the reinforcement task ✅ 2021-08-09
- [x] #task print the best rule at the end of the run ✅ 2021-08-09
- [x] #task add a method to render the best rule in action and show it after the last run ✅ 2021-08-09
- [ ] #task generalise GymRunner to more environments


## Class design 
(from [[2021-08-06]])
Class  GymRunner
`__init__` method:  
- same parameters as FuzzyClassifier
- also do all the toolkit initialisation that fit does?

`run` method
- parameters:
    - gym env
    - antecents
    - consequents
    - tensorboard_writer

`_evaluate` method
- implements the gym run loop and returns the score

`play` method
- run the best individual and render the result

## Refactorings:
- [x] create a base class
- [x] move the init method to the base class
- [x] in the fit method, move common code together
- [x] refactor out common code into a helper method
- [x] move helper method to base class
- [x] move base class to a new file
- [x] maybe split into smaller functions?
- [x] add GymRunner subclass


