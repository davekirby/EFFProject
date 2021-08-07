from sklearn.datasets import load_iris
import pandas as pd

from classifier_cv import cross_validate, HyperParams

"""Script for testing the classifier by running it on the iris dataset.
"""

tensorboard_dir = "tb_logs/iris_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=50,
    hall_of_fame_size=10,
    max_generation=5,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=1,
    max_tree_height=3,
    max_rules=4,
    whole_rule_prob=0.2,
    tree_height_limit=5,
    batch_size=30,
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

cross_validate(iris, y, hyperparams, antecendent_terms, classes, tensorboard_dir)
