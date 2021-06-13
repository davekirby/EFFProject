# Thoughts on chromosome implementation
- use [[DEAP]] library for GP.
- use typed GP to control how the tree is built.

Should a chromosome consist of a set of rules or a single rule?  
- If it is a set of rules then how to control varying the size of the rule set?  
- if a single rule, then how to measure the fitness?  And how to decide what rules to include?

Conclusions from [[2021-06-12]] thoughts:
to define PrimitiveSetTyped:
- add a terminal for each fuzzy term of each Antecedent
- add a terminal for each fuzzy term of each Consequent
- add a Rule class that takes (Antecedent expression, consequent expression)
- add AND, OR and NOT that take and return Antecendent expressions
- Initially use a single term consequent.  Then experiment with using an ephemeral constant that generates a weighted consequent or a list of (possibly weighted) consequents.

I want to keep the individual rules short, so have a small height limit.

In the future I may experiment with having multi-stage mutate & mate operators, i.e. that do tree operations on part of the chromosome and list operations on other parts.  e.g. model the consequents and the set of rules as a variable length list, with tree M&M operations on the antecedent part of each rule.  

Add elitism from ch4 of the hands-on GA book - https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/blob/master/Chapter04/elitism.py

