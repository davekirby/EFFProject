import skfuzzy as fuzz
import skfuzzy.control as ctrl
from deap import creator, base, tools
import numpy as np
import pandas as pd

from evofuzzy.fuzzyclassifier import FuzzyClassifier
from evofuzzy.fuzzygp import Config, registerCreators


def test_pandas_simple():
    classifier = FuzzyClassifier()
    # build a simple dataframe and set of rules
    size, elephant, mouse = _create_antecedents_and_consequents()
    rules = [
        ctrl.Rule(size["large"], elephant["likely"]),
        ctrl.Rule(size["small"], mouse["likely"]),
    ]
    data = pd.DataFrame([[1], [9]], columns=["size"])
    prediction = classifier.predict(data, rules, classes=dict(mouse=0, elephant=1))
    assert prediction == [0, 1]


def _create_antecedents_and_consequents():
    size = ctrl.Antecedent(np.linspace(0, 10, 11), "size")
    size.automf(names="small medium large".split())
    elephant = ctrl.Consequent(np.linspace(0, 1, 10), "elephant", "som")
    mouse = ctrl.Consequent(np.linspace(0, 1, 10), "mouse", "som")
    for consequent in [mouse, elephant]:
        consequent["likely"] = fuzz.trimf(consequent.universe, (0.0, 1.0, 1.0))
    return size, elephant, mouse


def test_instance_creator():
    size, elephant, mouse = _create_antecedents_and_consequents()
    config = Config(min_tree_height=4, max_tree_height=4, min_rules=3, max_rules=3)
    toolbox = base.Toolbox()
    creator.create("RuleSetFitness", base.Fitness, weights=(-1.,))
    registerCreators(toolbox, config, [size], [elephant, mouse])
    individual = toolbox.individualCreator()
    assert len(individual) == 3
    for i in individual:
        assert i.height == 4
        rule = toolbox.compile(i)
        assert isinstance(rule, ctrl.Rule)
        print(rule)
