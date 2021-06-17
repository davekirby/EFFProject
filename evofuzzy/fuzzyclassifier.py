import numpy as np
from typing import Dict, List, Any
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from skfuzzy import control as ctrl


class FuzzyClassifier(BaseEstimator, ClassifierMixin):
    """Class to create a fuzzy rule classifier"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(
        self, X: pd.DataFrame, rules: List[ctrl.Rule], classes: Dict[str, Any]
    ) -> List[Any]:
        """Apply fuzzy rules to data in a pandas dataframe and
        predict the target class.
        :param X: Pandas dataframe with the data.  Column names must match the antecedent
                  names in the rules.
        :param rules:  list of fuzzy rules
        :poram classes: dict mapping rule consequent names to class values

        :returns: list of class predictions
        """
        controller = ctrl.ControlSystem(rules)
        classifier = ctrl.ControlSystemSimulation(controller)
        prediction = []
        class_names = list(classes.keys())
        class_vals = list(classes.values())
        for row in X.itertuples(index=False):
            classifier.inputs(row._asdict())
            classifier.compute()
            class_idx = np.argmax([classifier.output[name] for name in class_names])
            class_val = class_vals[class_idx]
            prediction.append(class_val)
        return prediction
