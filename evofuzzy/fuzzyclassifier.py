import random
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from deap import gp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from skfuzzy import control as ctrl
import skfuzzy as fuzz

from .fuzzybase import FuzzyBase


def _make_antecedents(
    X: pd.DataFrame, antecedent_terms: Dict[str, List[str]]
) -> List[ctrl.Antecedent]:
    if antecedent_terms is None:
        antecedent_terms = {}
    mins = X.min()
    maxes = X.max()
    antecedents = []
    for column in X.columns:
        antecedent = ctrl.Antecedent(
            np.linspace(mins[column], maxes[column], 11), column
        )
        terms = antecedent_terms.get(column, None)
        if terms:
            antecedent.automf(names=terms)
        else:
            antecedent.automf(variable_type="quant")
        antecedents.append(antecedent)
    return antecedents


def _make_consequents(classes: Dict[str, Any]) -> List[ctrl.Consequent]:
    consequents = []
    for cls in classes:
        cons = ctrl.Consequent(np.linspace(0, 1, 10), cls, "som")
        cons["likely"] = fuzz.trimf(cons.universe, (0.0, 1.0, 1.0))
        cons["unlikely"] = fuzz.trimf(cons.universe, (0.0, 0.0, 1.0))
        consequents.append(cons)
    return consequents


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

        if columns:
            # if columns is provided then assume either X is a numpy array or the user
            # want to rename the dataframe columns
            X = pd.DataFrame(data=X, columns=columns)

        self.antecedents_ = _make_antecedents(X, antecedent_terms)
        self.consequents_ = _make_consequents(classes)

        self.initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, X=X, y=y)

        slices = list(batches_slices(len(X), self.batch_size))

        return self.execute(slices, tensorboard_writer)

    def predict(self, X: pd.DataFrame):
        individual = self.hof_[0]
        rules = [self.toolbox_.compile(rule) for rule in individual]
        return _make_predictions(X, rules, self.classes_)

    def _evaluate(self, individual, batch_slice, X, y):
        X = X.iloc[batch_slice]
        y = y.iloc[batch_slice]
        rules = [self.toolbox_.compile(rule) for rule in individual]
        predictions = _make_predictions(X, rules, self.classes_)
        return (accuracy_score(y, predictions),)

    def _mate(self, ind1, ind2):
        rule1_idx = random.randint(0, ind1.length - 1)
        rule2_idx = random.randint(0, ind2.length - 1)
        if random.random() < self.whole_rule_prob:
            # swap entire rules over
            rule2 = ind1[rule1_idx]
            rule1 = ind2[rule2_idx]
        else:
            rule1, rule2 = gp.cxOnePoint(ind1[rule1_idx], ind2[rule2_idx])
        ind1[rule1_idx] = rule1
        ind2[rule2_idx] = rule2
        return ind1, ind2

    def _mutate(self, individual):
        rule_idx = random.randint(0, individual.length - 1)
        if random.random() < self.whole_rule_prob:
            rule = self.toolbox_.expr()
        else:
            (rule,) = gp.mutUniform(
                individual[rule_idx], expr=self.toolbox_.expr, pset=self.pset_
            )
        individual[rule_idx] = rule
        return (individual,)

    @property
    def best(self):
        return self.hof_[0]

    def best_size(self, *args):
        return len(self.best)

    @property
    def best_strs(self):
        return [str(self.toolbox_.compile(r)).splitlines()[0] for r in self.best]


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


def batches_slices(max_size, batch_size=None):
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
