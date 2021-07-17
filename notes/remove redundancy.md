What would it take to remove redundancies in rules?
1. list possible optimisations
2. figure out how to implement them in an efficient manner

Possible optimisations:
NOT(NOT(X)) -> replace with X

AND( X, X) - replace with X
OR(X, X) - replace with X

NOT(AND(NOT(X), NOT(Y))) -> replace with OR(X, Y)
NOT(OR(NOT(X), NOT(Y)) -> replace with AND(X, Y)

When would be the best time to do this?
- every time a new rule is created or modified?
- at the end, and only when displaying the best rule?

Optimising the rules during the run may speed things up, but may limit diversity.  If the optimisation itself is slow it may reduce any speed gains from having shorter rules.    On the other hand it may help keep the rule size from growing too much.
I guess the only way to find out which is best is to try it and see. 

# Implementation notes
The PrimitiveSet is a list subclassed with some helper functions:
- `__setitem__` does validation that what is being inserted is a valid subtree.
- searchSubtree returns the subtree at a given node.  This will be the most useful.
It would be useful to have a function that give a binary node, returns the two subtrees.  However I think searchSubtree could be used for that - get first subtree and use that to find the top of the second subtree and call searchSubtree again to get that.  
Thought:  Do a scan through the tree once and build up an index of where the subtrees are for each top node.  But that will fall apart if I start removing nodes.  Will have to sleep on this.
It should be OK if I only delete nodes in front of the current position.

Initial algorithm pseudocode:
```python
pos = 0
while pos < len(rule):
  if rule[pos] == NOT and rule[pos+1] == NOT:
    del rule[pos+1]
    continue
  if rule[pos] in (AND, OR):
    lhs = rule.searchSubtree(pos+1)
    rhs = rule.searchSubtree(lhs.stop)
    if lhs == rhs:
       rule[pos:rhs.stop] = rule[lhs]
       continue
    pos += 1 
```
Can add test for NOT(AND(NOT(X), NOT(Y))) etc later.
-> replace with OR(X, Y)  and vice versa.
I can use the primitiveTree methods for converting to & from strings to create the unit tests.  Will work on that tomorrow.

Implementation added.  For the iris rules it does not make much difference, but increasing the rule size parameters does result in significant reduction of 30-40 primitives per generation in a population of 50.  

