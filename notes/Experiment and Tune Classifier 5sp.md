# Objective
Try the classifier with different data sets and evaluate its performance.  Try different hyperparameters to see what effect they have.  
Record the results so that they can be analysed and included in the report.

# Methods
How best to record the results?
I can run the experiments in a notebook and commit to git after each run.  However that make it hard to compare across runs.  
I should do that, but also look at logging the results in some kind of database for analysis.   Maybe use tensorboard?  
Some useful links on using Tensorboard without tensorflow:
- https://neptune.ai/blog/tensorboard-tutorial
- https://towardsdatascience.com/a-quickstart-guide-to-tensorboard-fb1ade69bbcf
- https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

I don't know if you can install it without tensorboard though.  
There is also https://neptune.ai/ which is free for personal use.  I presume you have to store everything in the cloud.  
Tensorboard alternatives:
https://neptune.ai/blog/the-best-tensorboard-alternatives
(by Neptune, so probably biased)
and blog by thoughtworks - probably less biased:
https://www.thoughtworks.com/radar/tools/experiment-tracking-tools-for-machine-learning
Comet is another option: https://www.comet.ml

also may be useful:  https://dvc.org/ for versioning large data using git.  

Found https://github.com/lanpa/tensorboardX - a tensorflow-free writer for tensorboard.   Docs at https://tensorboardx.readthedocs.io/en/latest/index.html.  Going to go with this since it is reasonably up to date and maintained.


# Tasks 
- [x] #task set up tensorboard (or other) and extend DEAP logging code to use it ✅ 2021-07-11
- [x] #task add more configurable hyperparameters ✅ 2021-07-11
    - parsimony and fitness_size for double tornament
    - linspace for fuzzy variables?
- [x] #task create notebook for running tests & committing to git ✅ 2021-08-05
    - [ ] reload gp module each time?
    - or write a python script?
- [ ] #task pick some datasets to try out
- [x] #task change code to do mini-batch learning for performance improvement ✅ 2021-08-05
- [x] #task add function to [[remove redundancy]] in rules ✅ 2021-08-05


#  Resources for finding datasets:
https://machinelearningmastery.com/standard-machine-learning-datasets/
https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
https://www.openml.org/home (loadable directly in sklearn)
https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/

## Candidate data set
https://archive.ics.uci.edu/ml/datasets/Statlog+%28Image+Segmentation%29
 - 2310 rows
 - 19 features
 - 7 classes - balanced, 330 of each

Results from statlog experiments:
http://www.is.umk.pl/~duch/projects/projects/datasets-stat.html#Image
Best error on the test set is about 0.02, or 98% accuracy.  A high bar!  Most algorithms are in the 95-96% accuracy range.
Please cite: [UCI](http://archive.ics.uci.edu/ml/citation_policy.html)
This looks like the same dataset on openML: https://www.openml.org/d/40984

#[[Experiments for Segmentation Classifer]]
