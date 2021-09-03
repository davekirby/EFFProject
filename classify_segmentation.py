import pandas as pd
from sklearn.datasets import fetch_openml

from classifier_cv import cross_validate, HyperParams

"""Script for cross-validating the image segmentation dataset.
"""

tensorboard_dir = "tb_logs/segment_cv/"
# tensorboard_dir = None

hyperparams = HyperParams(
    population_size=100,
    elite_size=3,
    max_generation=2,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=8,
    max_rules=12,
    whole_rule_prob=0.1,
    tree_height_limit=5,
    batch_size=10,
    forgetting=0.4,
)

data, y = fetch_openml(data_id=40984, as_frame=True, return_X_y=True)

data.columns = [c.replace(".", "_") for c in data.columns]

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
)
