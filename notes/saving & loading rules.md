Problem:
Saving and loading of rules is currently broken.
Works OK if it is the same instance of FuzzyBase, but not if a new instance is created or a whole new process.
I created an 'adding-save-and-load' branch a while ago.  Maybe have a play around on there.   

Merged my latest code to that branch.
There is a catch-22.  It fails to unpickle because it needs the Individual class defined in creator, but the other objects to create that class are in the pickle too.  

Possible solutions:
- pickle two files, one with the creator classes and one with the individuals.  Unpickle the creator classes first and put them in the creator module before unpickling the rest.  
- use marshal in combination with pickle to serialise them from the same file in turn. 
- pickle both to strings then pickle a list with both strings.  That seems the simplest.
- rewrite the whole thing so that the creator stuff is done at the top level.  Not really viable for what I want though.
- convert all the Individual instances to something else before pickling, then convert back when unpickle.

**21:38** WTF?  I moved the creator.create calls to the top level as it says in the DEAP bug reports I have seen for this issue, but now it fails when trying to distribute the population with multiprocessing - before I even get to the save & load part.  
Hmmm... it passes when the save & load test is run on its own, so I think it breaks if you create a new classifier.
Ah... it was because in another test I do creator.create so the class had changed - that confused it.
Yay! It works.  Will merge back to main branch.

