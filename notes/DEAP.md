# Notes on using the DEAP library

## Main classes and functions

### `creator.create(name, base_class, **kwargs)`
construct a new class as a subclass of base class and put it in the creator module. 
Not sure of the advantage over creating a class the normal way, TBH.
Normally used for creating the Fitness and Individual classes.  
The individual has a fitness attribute that holds a subclass of base.Fitness.

### `base.Toolbox & toolbox.register(alias, func, *args, **kwargs)`
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

## GP in DEAP - implementation notes
- create a PrimitiveSetTyped instance
- populate it with ephemeralConstants, terminals and primitives.
- The repr of things that you add must be executable to create the thing. 
- similarly when adding a terminal the name you give it must be resolvable to the thing.  If accessing an item in a collection it is necessary to add the collection to the primitiveSetTyped.context dictionary.  This holds the namespace that the repr will be executed in.
-  The individual is then represented by a gp.PrimitiveTree that the mutate and mate operators can work on. 
-  there does not seem to be an easy way to create union types like you can in the typing module.  
-  since DEAP will try to build a tree to the height given by the limits, it will fail if there is only a terminal of a given type since it cannot build on that.  To get over this I added a noop primitive that just returned its argument and declared it of that type.
- I managed to create a list of rules as a chromosome - had to explicitly convert the list into a PrimitiveTree when I did it.
- each rule needs to be converted to a fuzzy rule by calling gp.compile on it before it can be used for prediction.  



