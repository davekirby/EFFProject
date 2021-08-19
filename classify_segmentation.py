import pandas as pd
from sklearn.datasets import fetch_openml

from classifier_cv import cross_validate, HyperParams

"""Script for cross-validating the image segmentation dataset.
"""

tensorboard_dir = "tb_logs/segment_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=100,
    hall_of_fame_size=5,
    max_generation=5,
    mutation_prob=0.8,
    crossover_prob=0.3,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=12,
    max_rules=15,
    whole_rule_prob=0.1,
    tree_height_limit=5,
    batch_size=10,
    forgetting=0.6
)

data, y = fetch_openml(data_id=40984, as_frame=True, return_X_y=True)

data.columns = [c.replace(".", "_") for c in data.columns]

classes = {name: name for name in y.dtype.categories}

antecendent_terms = {col: ["low", "medium", "high"] for col in data.columns}

cross_validate(
    data,
    y,
    hyperparams,
    antecendent_terms,
    classes,
    tensorboard_dir,
    train_test_swap=False,
)
