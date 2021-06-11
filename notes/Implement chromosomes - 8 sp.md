# Thoughts on chromosome implementation
- use [[DEAP]] library for GP.
- use typed GP to control how the tree is built.

Should a chromosome consist of a set of rules or a single rule?  
- If it is a set of rules then how to control varying the size of the rule set?  
- if a single rule, then how to measure the fitness?  And how to decide what rules to include?
- 