"""
Cross-validation function for the FuzzyClassifier class.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import tensorboardX
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from evofuzzy import fuzzyclassifier


class HyperParams(NamedTuple):
    min_tree_height: int = 2
    max_tree_height: int = 4
    min_rules: int = 2
    max_rules: int = 5
    population_size: int = 100
    max_generation: int = 50
    mutation_prob: float = 0.1
    crossover_prob: float = 0.9
    whole_rule_prob: float = 0.1
    tree_height_limit: int = 10
    hall_of_fame_size: int = 5
    mutation_min_height: int = 0
    mutation_max_height: int = 2
    replacements: int = 5
    tournament_size: int = 5
    parsimony_size: float = 1.9
    batch_size: Optional[int] = None


def cross_validate(
    train_x,
    train_y,
    hyperparams,
    antecendent_terms,
    classes,
    tensorboard_dir,
    train_test_swap=False,
):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    if tensorboard_dir and tensorboard_dir[-1] == "/":
        tensorboard_dir = tensorboard_dir[:-1]

    for (i, (train_idx, test_idx)) in enumerate(kfold.split(train_x, train_y)):
        if train_test_swap:
            train_idx, test_idx = test_idx, train_idx
        if tensorboard_dir:
            logdir = Path(
                f"{tensorboard_dir}/{i}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            logdir.mkdir(parents=True, exist_ok=True)
            tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
        else:
            tensorboard_writer = None

        classifier = fuzzyclassifier.FuzzyClassifier(
            min_tree_height=hyperparams.min_tree_height,
            max_tree_height=hyperparams.max_tree_height,
            min_rules=hyperparams.min_rules,
            max_rules=hyperparams.max_rules,
            population_size=hyperparams.population_size,
            max_generation=hyperparams.max_generation,
            mutation_prob=hyperparams.mutation_prob,
            crossover_prob=hyperparams.crossover_prob,
            whole_rule_prob=hyperparams.whole_rule_prob,
            tree_height_limit=hyperparams.tree_height_limit,
            hall_of_fame_size=hyperparams.hall_of_fame_size,
            mutation_min_height=hyperparams.mutation_min_height,
            mutation_max_height=hyperparams.mutation_max_height,
            replacements=hyperparams.replacements,
            tournament_size=hyperparams.tournament_size,
            parsimony_size=hyperparams.parsimony_size,
            batch_size=hyperparams.batch_size,
        )
        classifier.fit(
            train_x.iloc[train_idx],
            train_y.iloc[train_idx],
            classes,
            antecedent_terms=antecendent_terms,
            tensorboard_writer=tensorboard_writer,
        )

        print(f"Best Rule:  size = {len(classifier.best)}")
        print(classifier.best_str)
        print(
            "Final length of rules sets",
            dict(Counter(x.length for x in classifier.population_)),
        )
        predictions = classifier.predict(train_x.iloc[test_idx])
        actual = train_y.iloc[test_idx]
        accuracy = str(sum(actual == predictions) / len(actual))
        target_names = classes.keys()
        confusion = pd.DataFrame(
            data=confusion_matrix(actual, predictions),
            columns=target_names,
            index=target_names,
        )
        print("Test accuracy:", accuracy)
        print(confusion)
        if tensorboard_writer:
            tensorboard_writer.add_text("cv_accuracy", accuracy)
            tensorboard_writer.add_text("confusion", confusion.to_markdown())
            tensorboard_writer.close()
