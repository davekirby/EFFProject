import pandas as pd
from sklearn.datasets import fetch_openml

from classifier_cv import cross_validate, HyperParams

"""Script for doing 5-fold cross-validation on the Wisconsin Cancer dataset.
"""

tensorboard_dir = "tb_logs/cancer_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=50,
    elite_size=3,
    n_iter=5,
    mutation_prob=0.5,
    crossover_prob=0.5,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=4,
    max_rules=7,
    whole_rule_prob=0.1,
    batch_size=50,
)

data, y = fetch_openml(data_id=15, as_frame=True, return_X_y=True)

# the Bare_Nuclei column has some null values, so drop them
na_mask = data["Bare_Nuclei"].notna()
data = data[na_mask]
y = y[na_mask]

classes = {name: name for name in y.dtype.categories}

antecendent_terms = {
    col: ["very_low", "low", "medium", "high", "very_high"] for col in data.columns
}

cross_validate(
    data,
    y,
    hyperparams,
    antecendent_terms,
    classes,
    tensorboard_dir,
    train_test_swap=False,
    number_of_predictors=1,
)
