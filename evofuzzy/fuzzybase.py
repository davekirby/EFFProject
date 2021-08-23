import random
from typing import Optional, Dict, List, Any, Iterable

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from deap import base, creator, tools, gp
from skfuzzy import control as ctrl

from evofuzzy.fuzzygp import (
    ea_with_elitism_and_replacement,
    CreatorConfig,
    registerCreators, RuleSet,
)


class FuzzyBase:
    """Common base class for FuzzyClassifier and GymRunner"""

    always_evaluate_ = False
    population_ = None

    def __init__(
        self,
        min_tree_height: int = 2,
        max_tree_height: int = 4,
        min_rules: int = 2,
        max_rules: int = 5,
        population_size: int = 100,
        max_generation: int = 20,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.9,
        whole_rule_prob: float = 0.1,
        tree_height_limit: int = 10,
        hall_of_fame_size: int = 5,
        mutation_min_height: int = 0,
        mutation_max_height: int = 2,
        replacements: int = 5,
        tournament_size: int = 5,
        parsimony_size: float = 1.7,
        batch_size: Optional[int] = None,
        forgetting: float = 1,
    ):
        self.min_tree_height = min_tree_height
        self.max_tree_height = max_tree_height
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.population_size = population_size
        self.max_generation = max_generation
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.whole_rule_prob = whole_rule_prob
        self.tree_height_limit = tree_height_limit
        self.hall_of_fame_size = hall_of_fame_size
        self.mutation_min_height = mutation_min_height
        self.mutation_max_height = mutation_max_height
        self.replacements = replacements
        self.tournament_size = tournament_size
        self.parsimony_size = parsimony_size
        self.batch_size = batch_size
        self.forgetting = forgetting

    def initialise(self, tensorboard_writer):
        if tensorboard_writer:
            hparams = "\n".join(
                f"* {k}: {v}" for (k, v) in self.__dict__.items() if not k.endswith("_")
            )
            tensorboard_writer.add_text("hparams", hparams)
        self.toolbox_ = base.Toolbox()
        self.config_ = CreatorConfig(
            self.min_tree_height, self.max_tree_height, self.min_rules, self.max_rules
        )
        self.pset_ = registerCreators(
            self.toolbox_, self.config_, self.antecedents_, self.consequents_
        )
        self.toolbox_.register(
            "select",
            tools.selDoubleTournament,
            fitness_size=self.tournament_size,
            parsimony_size=self.parsimony_size,
            fitness_first=True,
        )
        self.toolbox_.register("mate", self._mate)
        self.toolbox_.register("mutate", self._mutate)

        self.fitness_stats_ = tools.Statistics(get_fitness_values)
        self.fitness_stats_.register("max", np.max)
        self.fitness_stats_.register("avg", np.mean)
        self.size_stats_ = tools.Statistics(len)
        self.size_stats_.register("min", np.min)
        self.size_stats_.register("avg", np.mean)
        self.size_stats_.register("best", self.best_size)
        self.stats_ = tools.MultiStatistics(
            fitness=self.fitness_stats_, size=self.size_stats_
        )

    def execute(self, slices, tensorboard_writer, warm_start=False):
        if not warm_start or self.population_ is None:
            self.population_ = self.toolbox_.populationCreator(n=self.population_size)

        self.population_, self.logbook_ = ea_with_elitism_and_replacement(
            self.population_,
            self.toolbox_,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.max_generation,
            replacement_size=self.replacements,
            stats=self.stats_,
            tensorboard_writer=tensorboard_writer,
            hof_size=self.hall_of_fame_size,
            verbose=True,
            slices=slices,
            always_evalute=self.always_evaluate_,
            forgetting=self.forgetting,
        )
        if tensorboard_writer:
            tensorboard_writer.add_text("best_ruleset", self.best_str)
            tensorboard_writer.add_text("size_of_best_ruleset", str(self.best_size()))

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
        return self.population_[-1] if self.population_ else None

    def best_size(self, *args):
        return len(self.best) if self.best else 0

    @property
    def best_str(self):
        return self.individual_to_str(self.best) if self.best else "Unevaluated"

    def individual_to_str(self, individual):
        return "\n".join(
            str(self.toolbox_.compile(r)).splitlines()[0] for r in individual
        )

    def best_n(self, n=1):
        """Create a new rule set that combines the top n individuals"""
        if n == 1:
            return self.best
        rules = RuleSet()
        for individual in self.population_[-n:]:
            rules.extend(individual)
        return rules

def get_fitness_values(ind):
    return ind.fitness.values


def make_antecedent(name, min, max, terms=None, inf_limit=None):
    if inf_limit is not None and min == -np.inf:
        min = -inf_limit
    if inf_limit is not None and max == np.inf:
        max = inf_limit
    antecedent = ctrl.Antecedent(np.linspace(min, max, 11), name)
    if terms:
        antecedent.automf(names=terms)
    else:
        antecedent.automf(variable_type="quant")
    return antecedent


def make_antecedents(
    X: pd.DataFrame, antecedent_terms: Dict[str, List[str]]
) -> List[ctrl.Antecedent]:
    if antecedent_terms is None:
        antecedent_terms = {}
    mins = X.min()
    maxes = X.max()
    antecedents = []
    for column in X.columns:
        terms = antecedent_terms.get(column, None)
        antecedent = make_antecedent(column, mins[column], maxes[column], terms)
        antecedents.append(antecedent)
    return antecedents


def make_binary_consequents(classes: Iterable[str]) -> List[ctrl.Consequent]:
    consequents = []
    for cls in classes:
        cons = ctrl.Consequent(np.linspace(0, 1, 10), cls, "som")
        cons["likely"] = fuzz.trimf(cons.universe, (0.0, 1.0, 1.0))
        cons["unlikely"] = fuzz.trimf(cons.universe, (0.0, 0.0, 1.0))
        consequents.append(cons)
    return consequents
