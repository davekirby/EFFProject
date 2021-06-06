import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from skfuzzy import control as ctrl

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X, rules, classes):
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

