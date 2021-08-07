# Objective
Re-implement the GP engine to do reinforcement learning against OpenAI Gym.  

## Tasks
- [x] #task Install Gym ✅ 2021-08-05
- [ ] #task implement interface between GP engine and Gym
    - [ ] refactor to reuse code
- [ ] #task Choose simple reinforcement task
- [ ] #task get GP working with the reinforcement task


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


## Refactorings:
- [ ] create a base class
- [ ] move the init method to the base class
- [ ] in the fit method, move common code together
- [ ] refactor out common code into a helper method
- [ ] move helper method to base class
- [ ] maybe split into smaller functions?
- [ ] add GymRunner subclass


