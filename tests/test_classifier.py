from importlib import reload
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from deap import creator, base
from deap.gp import PrimitiveTree
import numpy as np
import pandas as pd
import pytest

from evofuzzy.fuzzyclassifier import FuzzyClassifier, _make_predictions
from evofuzzy.fuzzygp import (
    CreatorConfig,
    register_primitiveset_and_creators,
    _make_primitive_set,
)
from evofuzzy.fuzzygp import _prune_rule


def test_pandas_classifier():
    # build a simple dataframe and set of rules
    size, elephant, mouse = _create_antecedents_and_consequents()
    rules = [
        ctrl.Rule(size["large"], elephant["likely"]),
        ctrl.Rule(size["small"], mouse["likely"]),
    ]
    data = pd.DataFrame([[1], [9]], columns=["size"])
    prediction = _make_predictions(data, rules, classes=dict(mouse=0, elephant=1))
    assert prediction == [0, 1]


def test_classifier_unknown_feature():
    """Test that the classifer can handle a feature that is not covered in the
    rule set.
    """
    size, elephant, mouse = _create_antecedents_and_consequents()
    rules = [
        ctrl.Rule(size["large"], elephant["likely"]),
        ctrl.Rule(size["small"], mouse["likely"]),
    ]
    data = pd.DataFrame([[1, 2], [9, 10]], columns=["size", "bogus"])
    prediction = _make_predictions(data, rules, classes=dict(mouse=0, elephant=1))
    assert prediction == [0, 1]


def test_classifier_unknown_class():
    """Test that the classifer can handle a feature that is not covered in the
    rule set.
    """
    size, elephant, mouse = _create_antecedents_and_consequents()
    rules = [
        ctrl.Rule(size["large"], elephant["likely"]),
        ctrl.Rule(size["small"], mouse["likely"]),
    ]
    data = pd.DataFrame([[1], [9]], columns=["size"])
    prediction = _make_predictions(
        data, rules, classes=dict(mouse=0, human=1, elephant=2)
    )
    assert prediction == [0, 2]


def _create_antecedents_and_consequents():
    """Create Antecedent and Consequent instances for testing.
    There is one Antecendent 'size' which can have the terms 'small', 'medium' or 'large'
    and two Consequents, 'mouse' and 'elephant',  which can have the term 'likely'.
    """
    size = ctrl.Antecedent(np.linspace(0, 10, 11), "size")
    size.automf(names="small medium large".split())
    elephant = ctrl.Consequent(np.linspace(0, 1, 10), "elephant", "som")
    mouse = ctrl.Consequent(np.linspace(0, 1, 10), "mouse", "som")
    for consequent in [mouse, elephant]:
        consequent["likely"] = fuzz.trimf(consequent.universe, (0.0, 1.0, 1.0))
    return size, elephant, mouse


def test_instance_creator():
    size, elephant, mouse = _create_antecedents_and_consequents()
    rules_size = 3
    height = 4
    config = CreatorConfig(
        min_tree_height=height,
        max_tree_height=height,
        min_rules=rules_size,
        max_rules=rules_size,
    )
    toolbox = base.Toolbox()
    register_primitiveset_and_creators(toolbox, config, [size], [elephant, mouse])
    individual = toolbox.individualCreator()
    assert individual.length == rules_size
    for i in individual:
        assert i.height == height
        rule = toolbox.compile(i)
        assert isinstance(rule, ctrl.Rule)


pset = None


def test_save_and_load():
    classifier = FuzzyClassifier(
        population_size=3,
        n_iter=2,
    )
    filename = "test.pkl"
    X = pd.DataFrame({"size": [10, 1]})
    y = pd.Series([1, 0])
    classes = {"mouse": 0, "elephant": 1}
    classifier.fit(X, y, classes)
    result = classifier.predict(X)
    classifier.save(filename)
    del classifier
    classifier = FuzzyClassifier()
    classifier.load(filename)
    assert classifier.predict(X) == result


@pytest.mark.parametrize(
    "input, output",
    [
        (
            "Rule(invert(invert(or_(size['medium'], size['large']))), [])",
            "Rule(or_(size['medium'], size['large']), [])",
        ),
        (
            "Rule(invert(invert(invert(or_(size['medium'], size['large'])))), [])",
            "Rule(invert(or_(size['medium'], size['large'])), [])",
        ),
        (
            "Rule(invert(invert(invert(invert(or_(size['medium'], size['large']))))), [])",
            "Rule(or_(size['medium'], size['large']), [])",
        ),
        (
            "Rule(or_(size['medium'], size['medium']), [])",
            "Rule(size['medium'], [])",
        ),
        (
            "Rule(and_(size['medium'], size['medium']), [])",
            "Rule(size['medium'], [])",
        ),
        (
            "Rule(and_(size['medium'], size['large']), [])",
            "Rule(and_(size['medium'], size['large']), [])",
        ),
        (
            "Rule(invert(invert(and_(size['medium'], size['medium']))), [])",
            "Rule(size['medium'], [])",
        ),
    ],
)
def test_rule_pruner(input, output):
    global pset
    if pset is None:
        # create the test pset if it does not exist - this is never
        # modified so can be shared between tests
        size, elephant, mouse = _create_antecedents_and_consequents()
        pset = _make_primitive_set([size], [mouse, elephant])

    rule = PrimitiveTree.from_string(input, pset)
    _prune_rule(rule)
    assert str(rule) == output
