from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from deap import base, creator, gp
from sklearn.base import BaseEstimator, ClassifierMixin
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from .fuzzygp import Config, registerCreators


def _make_antecedents(X: pd.DataFrame, antecedent_terms):
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


def _make_consequents(classes):
    consequents = []
    for cls in classes:
        cons = ctrl.Consequent(np.linspace(0, 1, 10), cls, "som")
        cons["likely"] = fuzz.trimf(cons.universe, (0.0, 1.0, 1.0))
        consequents.append(cons)
    return consequents


class FuzzyClassifier(BaseEstimator, ClassifierMixin):
    """Class to create a fuzzy rule classifier"""

    def __init__(
        self,
        min_tree_height: int = 2,
        max_tree_height: int = 4,
        min_rules: int = 2,
        max_rules: int = 5,
    ):
        self.min_tree_height = min_tree_height
        self.max_tree_height = max_tree_height
        self.min_rules = min_rules
        self.max_rules = max_rules

    def fit(
        self,
        X,
        y,
        classes: Dict[str, Any],
        antecedent_terms: Optional[Dict[str, List[str]]] = None,
        columns: Optional[List[str]] = None,
    ):
        self.classes_ = classes
        self.toolbox_ = base.Toolbox()
        self.config_ = Config(
            self.min_tree_height, self.max_tree_height, self.min_rules, self.max_rules
        )

        if columns:
            # if columns is provided then assume either X is a numpy array or the user
            # want to rename the dataframe columns
            X = pd.DataFrame(data=X, columns=columns)

        if not hasattr(creator, "RuleSetFitness"):
            creator.create("RuleSetFitness", base.Fitness, weights=(-1.0,))
        self.antecedents_ = _make_antecedents(X, antecedent_terms)
        self.consequents_ = _make_consequents(classes)

        self.pset_ = registerCreators(
            self.toolbox_, self.config_, self.antecedents_, self.consequents_
        )

        return self

    def predict(self, X: pd.DataFrame):
        # create a random rule set for now
        individual = self.toolbox_.individualCreator()
        rules = [self.toolbox_.compile(rule) for rule in individual]
        return _make_predictions(X, rules, self.classes_)


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
