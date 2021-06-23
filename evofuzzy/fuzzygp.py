import random
from typing import Any, List, NamedTuple

from deap import creator, base, algorithms, gp, tools
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control import Rule, Antecedent, Consequent
from skfuzzy.control.term import Term
import random
import operator
import numpy as np


def ident(x: Any) -> Any:
    """Identity function - returns the parameter unchanged.
    Used to enable terminal/ephemeral values to be created below a tree's minimum depth.
    """
    return x


def _makePrimitiveSet(
    antecendents: List[Antecedent], consequents: List[Consequent]
) -> gp.PrimitiveSetTyped:
    """Create a typed primitive set that can be used to generate fuzzy rules.

    :param antecendents:  The Antecedent instances for the rules
    :param consequents:  The Consequent instances for the rules.

    :return:  The PrimitiveSetTyped with all the rule components registered
    """

    class MakeConsequents:
        """Ephemeral constant that randomly generates consequents for the rules."""

        cons_terms = [
            f"{cons.label}['{name}']"
            for cons in consequents
            for name in cons.terms.keys()
        ]

        def __init__(self):
            self.value = random.choice(self.cons_terms)

        def __repr__(self):
            return f"[{self.value}]"

    pset = gp.PrimitiveSetTyped("Rule", [], Rule)

    for ant in antecendents:
        pset.context[ant.label] = ant
        for name, term in ant.terms.items():
            pset.addTerminal(term, Term, f"{ant.label}['{name}']")

    for cons in consequents:
        pset.context[cons.label] = cons

    pset.addEphemeralConstant("consequents", MakeConsequents, list)
    pset.addPrimitive(Rule, [Term, list], Rule)
    pset.addPrimitive(operator.and_, [Term, Term], Term)
    pset.addPrimitive(operator.or_, [Term, Term], Term)
    pset.addPrimitive(operator.invert, [Term], Term)
    pset.addPrimitive(ident, [list], list)

    return pset


class Config(NamedTuple):
    """Hyperparameter configuration"""

    min_tree_height: int = 2
    max_tree_height: int = 4
    min_rules: int = 2  # minimum number of rules to have in a chromosome
    max_rules: int = 5  # maximum number of rules to have in a chromosome


def registerCreators(
    toolbox: base.Toolbox,
    config: Config,
    antecendents: List[Antecedent],
    consequents: List[Consequent],
):
    """Create a primitive set for fuzzy rules and register an individualCreator and
    populationCreator with the toolbox.
    Prerequisites:
    - RuleSetFitness class registered in creator module
    - antecedents and consequents have had their terms defined

    :param toolbox: deap toolbox
    :param config:  Config instance holding hyperparameters
    :param antecendents: list of fuzzy antecendents used by the rules
    :param consequents: list of fuzzy consequents used by the rules
    :return: None
    """

    def genRuleSet(pset, min_, max_, type_=None):
        rules_len = random.randint(config.min_rules, config.max_rules)
        return [
            gp.PrimitiveTree(gp.genGrow(pset, min_, max_, type_))
            for _ in range(rules_len)
        ]

    pset = _makePrimitiveSet(antecendents, consequents)
    creator.create("Individual", list, fitness=creator.RuleSetFitness, pset=pset)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "expr",
        genRuleSet,
        pset=pset,
        min_=config.min_tree_height,
        max_=config.max_tree_height,
    )
    toolbox.register(
        "individualCreator", tools.initIterate, creator.Individual, toolbox.expr
    )
    toolbox.register(
        "populationCreator", tools.initRepeat, list, toolbox.individualCreator
    )
    return pset


def eaSimpleWithElitism(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=True,
):
    """Modified version of the DEAP eaSimple function to run the evolution process
    while keeping the top performing members in the HallOfFame from one generation to the next.

    Adapted from the book "Hands-On Genetic Algorithms with Python" by Eyal Wirsansky

    :param population: The initial population
    :param toolbox: the deap toolbox with functions registered on it
    :param cxpb: crossover probability 0 <= cxpb <= 1
    :param mutpb: mutation probability 0 <= mupb <= 1
    :param ngen: number of generations to run the evolution for
    :param stats: DEAP Stats instance for recording statistics
    :param halloffame: DEAP HallOfFame instance for recording the top performing individuals
    :param verbose: boolean flag - if True then print stats while running
    :return: final population and logbook

    """

    def evaluate_population(pop):
        # Evaluate the individuals in population pop with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_count = evaluate_population(population)

    if halloffame:
        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0
    else:
        hof_size = 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=invalid_count, **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_count = evaluate_population(offspring)

        if halloffame:
            offspring.extend(halloffame.items)
            halloffame.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=invalid_count, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
