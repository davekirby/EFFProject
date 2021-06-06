from evofuzzy.classifier import Classifier
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np
import pandas as pd

def test_pandas_simple():
    classifier = Classifier()
    # build a simple dataframe and set of rules
    size = ctrl.Antecedent(np.linspace(0, 10, 11), 'size')
    size.automf(names='small medium large'.split())
    elephant = ctrl.Consequent(np.linspace(0, 1, 10), 'elephant', 'som')
    mouse = ctrl.Consequent(np.linspace(0, 1, 10), 'mouse', 'som')
    for consequent in [mouse, elephant]:
        consequent['likely'] = fuzz.trimf(consequent.universe, (0., 1., 1.))
    rules = [
        ctrl.Rule(size['large'], elephant['likely']),
        ctrl.Rule(size['small'], mouse['likely'])
    ]
    data = pd.DataFrame(
        [[1], [9]],
        columns=['size']
    )
    prediction = classifier.predict(data, rules, classes=dict(mouse=0, elephant=1))
    assert prediction == [0, 1]
