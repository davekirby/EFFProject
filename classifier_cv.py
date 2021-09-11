"""
Cross-validation function for the FuzzyClassifier class.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import NamedTuple, Optional, Iterable, List, Dict

import pandas as pd
import tensorboardX
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from evofuzzy import FuzzyClassifier


class HyperParams(NamedTuple):
    """NamedTuple holding the hyperparameters for the classifier."""

    min_tree_height: int = 2
    max_tree_height: int = 4
    min_rules: int = 2
    max_rules: int = 5
    population_size: int = 100
    n_iter: int = 50
    mutation_prob: float = 0.1
    crossover_prob: float = 0.9
    whole_rule_prob: float = 0.1
    elite_size: int = 5
    replacements: int = 5
    tournament_size: int = 5
    parsimony_size: float = 1.9
    batch_size: Optional[int] = None
    memory_decay: float = 1


def cross_validate(
    train_x: pd.DataFrame,
    train_y: Iterable,
    hyperparams: HyperParams,
    antecendent_terms: Dict[str : List[str]],
    classes: List[str],
    tensorboard_dir: Optional[str],
    train_test_swap: bool = False,
    number_of_predictors: int = 1,
):
    """Do 5-fold cross validation on training data and training target classes.  Print
    out the results and optionally write them to tensorboard directory.

    :param train_x: DataFrame of training data
    :param train_y: target classes
    :param hyperparams: HyperParams object
    :param antecendent_terms: mapping from antecendent names to list of terms to use
    :param classes: list of target class names
    :param tensorboard_dir: directory to write tensorboard info to (may be None)
    :param train_test_swap: If True then for each fold train on 1/5 the data and test on 4/5
    :param number_of_predictors: number of predictors to use for testing
    :return: None
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    if tensorboard_dir and tensorboard_dir[-1] == "/":
        tensorboard_dir = tensorboard_dir[:-1]

    results = []
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

        classifier = FuzzyClassifier(
            min_tree_height=hyperparams.min_tree_height,
            max_tree_height=hyperparams.max_tree_height,
            min_rules=hyperparams.min_rules,
            max_rules=hyperparams.max_rules,
            population_size=hyperparams.population_size,
            n_iter=hyperparams.n_iter,
            mutation_prob=hyperparams.mutation_prob,
            crossover_prob=hyperparams.crossover_prob,
            whole_rule_prob=hyperparams.whole_rule_prob,
            elite_size=hyperparams.elite_size,
            replacements=hyperparams.replacements,
            tournament_size=hyperparams.tournament_size,
            parsimony_size=hyperparams.parsimony_size,
            batch_size=hyperparams.batch_size,
            memory_decay=hyperparams.memory_decay,
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
        predictions = classifier.predict(train_x.iloc[test_idx], n=number_of_predictors)
        actual = train_y.iloc[test_idx]
        accuracy = sum(actual == predictions) / len(actual)
        results.append(accuracy)
        target_names = classes.keys()
        confusion = pd.DataFrame(
            data=confusion_matrix(actual, predictions),
            columns=target_names,
            index=target_names,
        )
        print("Test accuracy:", accuracy)
        print(confusion)
        if tensorboard_writer:
            tensorboard_writer.add_text("cv_accuracy", str(accuracy))
            tensorboard_writer.add_text("confusion", confusion.to_markdown())
            tensorboard_writer.close()

    print("Accuracy: ", results)
    print(f" -  average {fmean(results)}, std {stdev(results)}")
