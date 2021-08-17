# Scikit-fuzzy library
Summarise how to use skfuzzy.

# DEAP library
summarise how to use DEAP:
- use of toolbox to register functions
- use of creator to create types/classes
- population creation
- mate and mutate functions
- evaluate
    - returns a tuple
- algorithms - top level loop for running the process
- handling stats and logging
- 

## deap.gp
- strong and loose typed trees
- primitive sets
    - Types:
        - Primitive
        - Terminal
            - can be a constant, or parameterless function
        - EphemeralConstant
        - input parameters are created automatically as terminals
- gp.compile(tree, pset)
    - compiles a tree to a python function
-trees are held as lists of operators in reverse polish order


