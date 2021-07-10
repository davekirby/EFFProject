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
- [ ] #task set up tensorboard (or other) and extend DEAP logging code to use it
- [ ] #task add more hyperparameters
- [ ] #task create notebook for running tests & committing to git
    - [ ] reload gp module each time
- [ ] #task pick some datasets to try out
- [ ] #task change code to do mini-batch learning for performance improvement

