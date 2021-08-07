from typing import Optional

import numpy as np
from deap import base, creator, tools, gp

from evofuzzy.fuzzygp import (
    ea_with_elitism_and_replacement,
    CreatorConfig,
    registerCreators,
)


class FuzzyBase:
    """Common base class for FuzzyClassifier and GymRunner"""

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

    def execute(self, slices, tensorboard_writer):
        population = self.toolbox_.populationCreator(n=self.population_size)
        self.population_, self.logbook_ = ea_with_elitism_and_replacement(
            population,
            self.toolbox_,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.max_generation,
            replacements=self.replacements,
            stats=self.stats_,
            tensorboard_writer=tensorboard_writer,
            halloffame=self.hof_,
            verbose=True,
            slices=slices,
        )
        if tensorboard_writer:
            tensorboard_writer.add_text("best_ruleset", "\n\n".join(self.best_strs))
            tensorboard_writer.add_text("size_of_best_ruleset", str(self.best_size()))
        return self

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
        if not hasattr(creator, "RuleSetFitness"):
            creator.create("RuleSetFitness", base.Fitness, weights=(1.0,))
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
        self.toolbox_.register(
            "expr_mut",
            gp.genGrow,
            min_=self.mutation_min_height,
            max_=self.mutation_max_height,
        )
        self.toolbox_.register("mutate", self._mutate)
        self.hof_ = tools.HallOfFame(self.hall_of_fame_size)
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


def get_fitness_values(ind):
    return ind.fitness.values