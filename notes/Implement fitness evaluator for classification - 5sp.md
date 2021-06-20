# Objective
Add the evaluation functionality to the classifier.
- add hyperparameters to the init method
- create population in the fit method
    - get the Antecedent names from the DataFrame column names
    - add optional parameters for naming terms
        - a dict mapping column name to list of terms
        - use default for missing columns
        - add optional columns parameter to supply column names if not a dataframe
        - add a parameter for the number of default terms
    - refactor out a method for running a ruleset against a dataframe and returning predictions
        - run the method from the predict method using best ruleset
    - add methods to get the rules and to get a human readable version of the rules.
    - Add method/attribute to expose the logbook and maybe display a graph?
 
 Issues:  passing in the toolkit to either init or fit does not really fit in with the sklearn design pattern, but is needed to make some of the GP functions configurable.  Alternatively I could have the functions as method on the class, and the user subclasses the Classifier to override them.  Maybe get it working first then look at improving the design.  I think I like the subclassing idea since I can have reasonable default implementation 
 
