#report 

# Overview of Fuzzy Inference Systems
- Zadeh 1965 Fuzzy Sets
- Zadeh 1975 Linguistic Variables
- Mamdani vs Takagi-Sugeno FIS

FIS:
- take numerical input
- convert to fuzzy sets
- feed into the rules - rules trigger their outputs with different strengths
- outputs are defuzzified to give a crisp result.
- mamdani defuzzification
- Takagi-Sugeno - output is a function

Expand on the details of how it works - all a bit vague at the moment.

# Overview of Genetic Programming
- Koza 92
- represent as trees
- demonstrate mutation and crossover
    - diagrams?
    - steal from wikipedia?  Should be OK as log as I cite.
        - damn, wikipedia images are animated

Expand on the details of how it works - all a bit vague at the moment.


**Dictation:**
 genetic programming was introduced by JohnKoza in 1992 with his book of the same name. it is based on the ideas of genetic algorithms and evolutionary computing which were made popular by John Holland and others in the 1960s.

 genetic programming differs from standard  genetic algorithms by representing a computer program by a tree of primitives that's can be then executed to generate the result.

Genetic  programs evolve through a mixture of  mutation and crossover.  Mutations are done by selecting a random subtree and replacing it with a new randomly generated subtree.

 crossover is done by selecting the two potential parents and picking a random subtree on each and swapping them over. The newly created individuals are evaluated  and a fitness value generated for each one.
 
---
**Braindump**
what do I want to say about GP?  
Go into details about how it works and the options available.   
Intro to Evolutionary algorithms in general.   Difference between GA and GP.   Other models - PSO, differential, evolution strategy etc.  Also more esoteric such as wolf pack & golden beetle.  
Commonalities:
- codification of an individual (genotype)
- population of individuals
- fitness function
- selection
- mutation - randomly change part of an individual's chromosome
-  crossover - mate two individuals to produce a hybrid

variations in above produce different members of the EC family.  

GP uses trees for the genotype.
Bloat control - 


# Overview of RL

