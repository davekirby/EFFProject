# Objective
In the FuzzyClassifier fit method, implement the GP evolution engine.

## Tasks
- [x] #task add eaSimpleWithElitism function ✅ 2021-06-20
- [x] #task add hyperparameters for running the GP ✅ 2021-06-27
- [x] #task Update chromosome fitness value from prediction ✅ 2021-06-27
    - register 'evaluate' method in the toolbox - delegate to a method?
- [x] #task add methods for tuning GP and register them in the toolbox ✅ 2021-06-27
- [x] #task call eaSimpleWithElitism from fit method ✅ 2021-06-27
- [x] #task add HOF and stats ✅ 2021-06-27
- [ ] #task add plot method to FuzzyClassifier?
- [x] #task add methods to get the rules and to get a human readable version of the rules. ✅ 2021-06-27
- [x] #task Add method/attribute to expose the logbook ✅ 2021-06-27
- [ ] #task add docstrings
- [ ] #task add unit tests
- [x] #task add parallel processing ✅ 2021-06-27
- [x] #task update selector to minimise chromosome size ✅ 2021-06-27
- [x] #task add min and average size statistics ✅ 2021-06-27
- [x] #task add mate & mutate at the level of whole rules ✅ 2021-06-30
- [x] #task generate multiple consequents for a rule ✅ 2021-06-27
- [ ] #task try different datasets and evaluate the classifier


## Runtime performance improvements:
Currently with 20 chromosomes and 20 generations, it takes 35-52s to run on the iris dataset.  Going to parallelise that.  
... using a multiprocessing.Pool cut the time down to 10-12 seconds.    Not bad, but not 8 times faster with 8 cpus.  

## Variable length consequents list
Instead of having a single consequent per rule, going to implement a variable number of consequents.  It makes sense to only allow one of each consequent max, so will aim to do that and see what effect it has.



##  Limiting chromosome size
The deap selDoubleTournament function that selects for size as well as fitness is hard coded to use len(individual) for the size calculation.  This is not what I want - I can either override the list `__len__` method to return the cumulative size, or I will have to re-implement the whole function.  Overriding `__len__` may have unforseen side effects, but is the simpler option so will give it a go.  
...I have added the ListOfList class and double tournaments - it seems to work OK, but not sure how much effect it has on the size TBH.  
Also added min and average size statistics so I can see what effect it has.  
I should surface the hyperparameters for controlling it to init method.  

## replacing worst performers.  
I have added code to replace the N worst performing individuals with newly generated ones to prevent loss of diversity.  It seems to work well for keeping a range of ruleset sizes - before it was rapidly converging to a single size.  This happens if you randomly select values from a small range with replacement.  Also with tournament selection of k individuals the worst k-1 individuals will never get selected and the kth worst is highly unlikely to.   The only cost is computing the fitness of the new individuals.  


