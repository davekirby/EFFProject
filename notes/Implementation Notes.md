# Scikit-fuzzy library
Summarise how to use skfuzzy.
- fuzzy variables and terms
- Rules
- Antecedents
    - can be combined with |, & ~ operators (or operator.and_, or_, invert)
    - can be grouped with brackets
    - automf method to create the membership terms
- Consequents
    - A FuzzyVariable, like Antecendent, but can include a defuzzify method used to turn the fuzzy value back into a crisp value
    - can have a list of consequents that are triggered
    - can be weighted (not used in EFF)
    - automf method to create the membership terms
- ControlSystem
- ControlSystemSimulator
    - add the input values with `sim.inputs(dict)`
    - call compute()
    - get results from `output[name]`



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



