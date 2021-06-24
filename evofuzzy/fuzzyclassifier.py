import operator
import random
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from .fuzzygp import Config, registerCreators, eaSimpleWithElitism


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
        population_size: int = 100,
        max_generation: int = 50,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.9,
        tree_height_limit: int = 10,
        hall_of_fame_size: int = 5,
        mutation_min_height: int = 0,
        mutation_max_height: int = 2,
    ):
        self.min_tree_height = min_tree_height
        self.max_tree_height = max_tree_height
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.population_size = population_size
        self.max_generation = max_generation
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tree_height_limit = tree_height_limit
        self.hall_of_fame_size = hall_of_fame_size
        self.mutation_min_height = mutation_min_height
        self.mutation_max_height = mutation_max_height

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

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, X=X, y=y)
        self.toolbox_.register("select", tools.selTournament, tournsize=3)
        self.toolbox_.register("mate", self._mate)
        self.toolbox_.register("expr_mut", gp.genGrow, min_=self.mutation_min_height,
            max_=self.mutation_max_height)


        self.toolbox_.register("mutate", self._mutate)
        self.toolbox_.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.tree_height_limit
            ),
        )
        self.toolbox_.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.tree_height_limit
            ),
        )

        self.hof_ = tools.HallOfFame(self.hall_of_fame_size)
        self.stats_ = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats_.register("min", np.min)
        self.stats_.register("avg", np.mean)
        population = self.toolbox_.populationCreator(n=self.population_size)

        self.population_, self.logbook_ = eaSimpleWithElitism(
            population, self.toolbox_, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
            ngen=self.max_generation,
            stats=self.stats_,
            halloffame=self.hof_, verbose=True
        )
        return self

    def predict(self, X: pd.DataFrame):
        individual = self.hof_[0]
        rules = [self.toolbox_.compile(rule) for rule in individual]
        return _make_predictions(X, rules, self.classes_)

    def _evaluate(self, individual, X, y):
        rules = [self.toolbox_.compile(rule) for rule in individual]
        predictions = _make_predictions(X, rules, self.classes_)
        return accuracy_score(y, predictions)


    def _mate(self, ind1, ind2):
        rule1 = random.choice(ind1)
        rule2 = random.choice(ind2)
        return gp.cxOnePoint(rule1, rule2)


    def _mutate(self, individual):
        rule = random.choice(individual)
        return gp.mutUniform(individual, expr=self.toolbox_.expr, pset=self.pset_)

    @property
    def best(self):
        return self.hof_[0]

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
