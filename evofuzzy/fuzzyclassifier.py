from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase
from .fuzzygp import make_antecedents, make_binary_consequents


class FuzzyClassifier(FuzzyBase, BaseEstimator, ClassifierMixin):
    """Class to create a fuzzy rule classifier"""

    def fit(
        self,
        X,
        y,
        classes: Dict[str, Any],
        antecedent_terms: Optional[Dict[str, List[str]]] = None,
        columns: Optional[List[str]] = None,
        tensorboard_writer=None,
    ):

        X, y = shuffle(X, y)
        self.classes_ = classes

        if not columns and not isinstance(X, pd.DataFrame):
            columns = [f"column_{i}" for i in range(X.shape[1])]

        self.columns_ = columns

        if self.columns_:
            X = self.convert_dataframe(X)

        self.antecedents_ = make_antecedents(X, antecedent_terms)
        self.consequents_ = make_binary_consequents(classes.keys())

        self._initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, X=X, y=y)

        slices = list(_batch_slices(len(X), self.batch_size))

        self.execute(slices, tensorboard_writer)
        return self

    def convert_dataframe(self, X):
        # if columns is provided then assume either X is a numpy array or the user
        # want to rename the dataframe columns
        if isinstance(X, pd.DataFrame):
            X.columns = self.columns_
        else:
            X = pd.DataFrame(data=X, columns=self.columns_)
        return X

    def predict(self, X: pd.DataFrame, n=1):
        individual = self.best_n(n)
        if self.columns_:
            X = self.convert_dataframe(X)

        rules = [self.toolbox_.compile(rule) for rule in individual]
        return _make_predictions(X, rules, self.classes_)

    def _evaluate(self, individual, batch_slice, X, y):
        X = X.iloc[batch_slice]
        y = y.iloc[batch_slice]
        rules = [self.toolbox_.compile(rule) for rule in individual]
        predictions = _make_predictions(X, rules, self.classes_)
        return (accuracy_score(y, predictions),)


def _make_predictions(
    X: pd.DataFrame, rules: List[ctrl.Rule], classes: Dict[str, Any]
) -> List[Any]:
    """Apply fuzzy rules to data in a pandas dataframe and
    predict the target class.
    :param X: Pandas dataframe with the data.  Column names must match the antecedent
              names in the rules.
    :param rules:  list of fuzzy rules
    :param classes: dict mapping rule consequent names to target class values

    :returns: list of class predictions
    """
    antecedents = {
        term.parent.label for rule in rules for term in rule.antecedent_terms
    }
    columns = [col for col in X.columns if col in antecedents]
    X = X[columns]
    controller = ctrl.ControlSystem(rules)
    classifier = ctrl.ControlSystemSimulation(controller)
    prediction = []
    class_names = list(classes.keys())
    class_vals = list(classes.values())
    for row in X.itertuples(index=False):
        classifier.inputs(row._asdict())
        classifier.compute()
        class_idx = np.argmax([classifier.output.get(name, 0) for name in class_names])
        class_val = class_vals[class_idx]
        prediction.append(class_val)
    return prediction


def _batch_slices(max_size, batch_size=None):
    """generate slices to split an array-like object into smaller batches.
    If batch size is not given then yield a slice that covers the whole thing.
    """
    if batch_size is None:
        yield slice(0, max_size)
    else:
        range_iter = iter(range(0, max_size, batch_size))
        start = next(range_iter)
        for end in range_iter:
            yield slice(start, end)
            start = end
        if start != max_size:
            yield slice(start, max_size)
