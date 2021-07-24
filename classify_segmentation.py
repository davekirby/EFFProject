import pandas as pd
from sklearn.datasets import fetch_openml

from classifier_cv import cross_validate, HyperParams

"""Script for cross-validating the image segmentation dataset.
"""

tensorboard_dir = "tb_logs/segment_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=50,
    hall_of_fame_size=10,
    max_generation=5,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=2,
    max_tree_height=4,
    min_rules=7,
    max_rules=10,
    whole_rule_prob=0.2,
    tree_height_limit=6,
    batch_size=100,
)

data, y = fetch_openml(data_id=40984, as_frame=True, return_X_y=True)

data.columns = [c.replace(".", "_") for c in data.columns]

classes = {name: name for name in y.dtype.categories}

cross_validate(data, y, hyperparams, None, classes, tensorboard_dir)
