# Objective
Create a proof-of-concept inference system to act as a classifier for a simple dataset such as iris.   Initially the rules will be hand written.  
Thoughts:
- should be designed to fit into scikit-learn pipeline.  
- take a pandas dataframe with the data and maybe a dict of the fuzzy set names for each input and output variable
- takes a fuzzy rule set and applies it to the data
- use argmax of the defuzzified output of each class to make a prediction
- should be able to use standard sklearn metrics on the result.

It should be possible to do most of this directly with the skfuzzy library high level interface. 

- [ ] #task experiment with skfuzzy
    - [x] create a notebook in jupyter lab
- [ ] #task create simple iris classifier with hand-coded rules

So what rules do I want to implement?
Here are the ranges of values:

| param        | min | max |
| ------------ | --- | --- |
| sepal length | 4.3 | 7.9 |
| sepal width  | 2.0 | 4.4 |
| petal length | 1.0 | 6.9 |
| petal width  | 0.1 | 2.5 | 

Use 5 set names:  very short, short, medium, long, very long.
Does skfuzzy have hedges?  Doesn't look like it, but I think they are simple to implement as a function on the numpy arrays.  Maybe.  

IF petal width is narrow and petal length is short then cls is 0 (setosa)
IF petal width is medium and petal length is medium then cls is 1 (veriscolor)
IF petal width is wide and petal length is long then cls is 2 (virginica)

Maybe just use 3 terms for each axis then.  

Puzzling how to model the consequents:
1. have separate variables for setosa, versicolor and and verginica with a single term 'likely'
2. have one term 'category' with separate terms for each type
3. separate variables as 1 but have terms 'unlikely', 'possible', 'likely'

I will go with 1 for now and see what happens.  

Bugger, first pass threw a ValueError in the defuzzifier because there was zero area to defuzzify.  