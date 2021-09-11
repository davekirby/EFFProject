from sklearn.datasets import load_iris
import pandas as pd

from classifier_cv import cross_validate, HyperParams

"""Script for doing 5-fold cross-validation on the iris dataset.
"""

tensorboard_dir = "tb_logs/iris_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=20,
    elite_size=3,
    n_iter=10,
    mutation_prob=0.5,
    crossover_prob=0.5,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=3,
    max_rules=5,
    whole_rule_prob=0.2,
    batch_size=10,
)

data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)

classes = {name: val for (name, val) in zip(data.target_names, range(3))}
antecendent_terms = {
    col: ["v.narrow", "narrow", "medium", "wide", "v.wide"]
    if "width" in col
    else ["v.short", "short", "medium", "long", "v.long"]
    for col in cols
}

cross_validate(
    iris,
    y,
    hyperparams,
    antecendent_terms,
    classes,
    tensorboard_dir,
    number_of_predictors=1,
)
