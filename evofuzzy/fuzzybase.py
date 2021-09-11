import pickle
from typing import Optional, Iterable

import numpy as np
import tensorboardX
from deap import base, tools

from evofuzzy.fuzzygp import (
    ea_with_elitism_and_replacement,
    CreatorConfig,
    register_primitiveset_and_creators,
    RuleSet,
    mate_rulesets,
    mutate_ruleset,
    get_fitness_values,
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
        n_iter: int = 20,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.9,
        whole_rule_prob: float = 0.1,
        elite_size: int = 5,
        replacements: int = 5,
        tournament_size: int = 5,
        parsimony_size: float = 1.7,
        batch_size: Optional[int] = None,
        memory_decay: float = 1,
        verbose: bool = True,
    ):
        """Initialise hyperparameters"""

        self.min_tree_height = min_tree_height
        self.max_tree_height = max_tree_height
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.population_size = population_size
        self.n_iter = n_iter
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.whole_rule_prob = whole_rule_prob
        self.elite_size = elite_size
        self.replacements = replacements
        self.tournament_size = tournament_size
        self.parsimony_size = parsimony_size
        self.batch_size = batch_size
        self.memory_decay = memory_decay
        self.verbose = verbose

    def _initialise(self, tensorboard_writer: Optional["tensorboardX.SummaryWriter"]):
        """Initialise the toolbox and register functions needed by DEAP on it.
        Also set up the DEAP statistics object and optionally write hyperparameters to TensorBoard.
        :param tensorboard_writer: Object used to write to TensorBoard.
        :return:
        """

        if tensorboard_writer:
            hparams = "\n\n".join(
                f"* {k}: {v}" for (k, v) in self.__dict__.items() if not k.endswith("_")
            )
            tensorboard_writer.add_text("hparams", hparams)

        if hasattr(self, "toolbox_"):
            # already initialised
            return

        self.toolbox_ = base.Toolbox()
        self.config_ = CreatorConfig(
            self.min_tree_height, self.max_tree_height, self.min_rules, self.max_rules
        )
        self.pset_ = register_primitiveset_and_creators(
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

    def execute(
        self,
        slices: Optional[Iterable[slice]],
        tensorboard_writer: Optional["tensorboardX.SummaryWriter"],
    ):
        """Run the fuzzygp main loop.  Optionally write the best rule to TensorBoard.

        :param slices: sequence of slices to use for batching classifier training data.
        :param tensorboard_writer: Object used to write to TensorBoard.
        :return: None
        """
        if self.population_ is None:
            self.population_ = self.toolbox_.populationCreator(n=self.population_size)

        self.population_, self.logbook_ = ea_with_elitism_and_replacement(
            self.population_,
            self.toolbox_,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_iter,
            replacement_size=self.replacements,
            stats=self.stats_,
            tensorboard_writer=tensorboard_writer,
            elite_size=self.elite_size,
            verbose=self.verbose,
            slices=slices,
            always_evaluate=self.always_evaluate_,
            memory_decay=self.memory_decay,
        )
        if tensorboard_writer:
            tensorboard_writer.add_text("best_ruleset", self.best_str)
            tensorboard_writer.add_text("size_of_best_ruleset", str(self.best_size()))

    def _mate(self, ind1: RuleSet, ind2: RuleSet):
        """Do crossover on two individuals"""
        return mate_rulesets(ind1, ind2, self.whole_rule_prob)

    def _mutate(self, individual: RuleSet):
        """Mutate an individual"""
        return mutate_ruleset(self.toolbox_, individual, self.whole_rule_prob)

    @property
    def best(self):
        """Return the best individual"""
        return self.population_[-1] if self.population_ else None

    def best_size(self, *args):
        """Return the size of the best individual.
        N.B. takes additional *args because it is called from the DEAP stats code which
        passes additional arguments.
        """
        return len(self.best) if self.best else 0

    @property
    def best_str(self):
        """Return the string representation of the best individual"""
        return self.individual_to_str(self.best) if self.best else "Unevaluated"

    def individual_to_str(self, individual: RuleSet):
        """Convert any individual to its string representation"""
        return "\n".join(
            str(self.toolbox_.compile(r)).splitlines()[0] for r in individual
        )

    def save(self, filename: str):
        """Save the state of the object as a pickle"""
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f, -1)
        return self

    def load(self, filename: str):
        """Load the state of the object from a pickle created by the save method"""
        with open(filename, "rb") as f:
            self.__dict__ = pickle.load(f)
        return self

    def best_n(self, n: int = 1):
        """Create a new rule set that combines the top n individuals"""
        if n == 1:
            return self.best
        rules = RuleSet()
        for individual in self.population_[-n:]:
            rules.extend(individual)
        return rules
