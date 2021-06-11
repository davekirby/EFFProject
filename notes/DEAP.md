# Notes on using the DEAP library

## Main classes and functions

### `creator.create(name, base_class, **kwargs)`
construct a new class as a subclass of base class and put it in the creator module. 
Not sure of the advantage over creating a class the normal way, TBH.
Normally used for creating the Fitness and Individual classes.  
The individual has a fitness attribute that holds a subclass of base.Fitness.

### `base.Toolbox & toolbox.register(alias, func, *args, **kwargs`
create a partial function and save it as an attribute on the toolbox instance.  
This is so you can pass a toolbox to the various kinds of GA algorithms and it can access the necessary functions from it.  That means you need to create functions with the right names.

Common toolbox functions:
- select
- mutate
- mate
- evaluate - function that takes an individual and returns the fitness.
  in this case will be responsible for generating the fuzzy rules and executing the FIS against the training data.  
- population - usually just a list of Individuals

Most of these are usually based on classes/functions in tools, e.g. 
- selTournament for tournament selection
- cxOnePoint for single point crossover mating
- mutFlipBit for flip bit mutation
There are lots of others - RTFD.



