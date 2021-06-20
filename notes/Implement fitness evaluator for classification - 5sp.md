# Objective
Add the evaluation functionality to the classifier.
- [x] #task add hyperparameters to the init method ✅ 2021-06-20
- create population in the fit method
    - get the Antecedent names from the DataFrame column names
    - [x] #task add optional parameters for naming terms ✅ 2021-06-20
        - a dict mapping column name to list of terms
        - use default for missing columns
        - add optional columns parameter to supply column names if not a dataframe
        - add a parameter for the number of default terms
    - [x] #task refactor out a method for running a ruleset against a dataframe and returning predictions ✅ 2021-06-20
        - run the method from the predict method using best ruleset


 Issues:  passing in the toolkit to either init or fit does not really fit in with the sklearn design pattern, but is needed to make some of the GP functions configurable.  Alternatively I could have the functions as method on the class, and the user subclasses the Classifier to override them.  Maybe get it working first then look at improving the design.  I think I like the subclassing idea since I can have reasonable default implementation 
 
**2021-06-20** I have got working initialising the FuzzyClassifier to generate rules from a chromosome.
Next I will add the genetic programming part to the classifier.   I think this comes under the implement GP engine story.
