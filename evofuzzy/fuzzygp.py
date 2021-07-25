import heapq
import random
from itertools import repeat
from typing import Any, List, NamedTuple
import operator
from multiprocessing import Pool
from functools import partial

import numpy as np
from deap import creator, base, algorithms, gp, tools
from skfuzzy.control import Rule, Antecedent, Consequent
from skfuzzy.control.term import Term

BIG_INT = 1 << 64


def identity(x: Any) -> Any:
    """Identity function - returns the parameter unchanged.
    Used to enable terminal/ephemeral values to be created below a tree's minimum depth.
    """
    return x


class MakeConsequents:
    """Ephemeral constant that randomly generates consequents for the rules.
    This needs to be a class so that the repr can be defined to return something
    that can be used by toolbox.compile
    """

    def __init__(self, cons_terms):
        max_consequents = len(cons_terms) // 2 + 1
        sample_size = random.randint(1, max_consequents)
        candidates = random.sample(list(cons_terms.values()), sample_size)
        self.values = [random.choice(term) for term in candidates]

    def __repr__(self):
        return f"[{', '.join(value for value in self.values)}]"


def _makePrimitiveSet(
    antecendents: List[Antecedent], consequents: List[Consequent]
) -> gp.PrimitiveSetTyped:
    """Create a typed primitive set that can be used to generate fuzzy rules.

    :param antecendents:  The Antecedent instances for the rules
    :param consequents:  The Consequent instances for the rules.

    :return:  The PrimitiveSetTyped with all the rule components registered
    """
    cons_terms = {
        cons.label: [f"{cons.label}['{name}']" for name in cons.terms.keys()]
        for cons in consequents
    }
    makeConsequents = partial(MakeConsequents, cons_terms)

    pset = gp.PrimitiveSetTyped("Rule", [], Rule)

    for ant in antecendents:
        pset.context[ant.label] = ant
        for name, term in ant.terms.items():
            pset.addTerminal(term, Term, f"{ant.label}['{name}']")

    for cons in consequents:
        pset.context[cons.label] = cons

    # the DEAP gp module caches the consequents function as a module level attribute
    # and fails if it already exists
    if hasattr(gp, "consequents"):
        del gp.consequents
    pset.addEphemeralConstant("consequents", makeConsequents, list)
    pset.addPrimitive(Rule, [Term, list], Rule)
    pset.addPrimitive(operator.and_, [Term, Term], Term)
    pset.addPrimitive(operator.or_, [Term, Term], Term)
    pset.addPrimitive(operator.invert, [Term], Term)
    pset.addPrimitive(identity, [list], list)

    return pset


class CreatorConfig(NamedTuple):
    """Hyperparameter configuration for creating instances"""

    min_tree_height: int = 2
    max_tree_height: int = 4
    min_rules: int = 2  # minimum number of rules to have in a chromosome
    max_rules: int = 5  # maximum number of rules to have in a chromosome


def genRule(pset, min_, max_, type_=None):
    return gp.PrimitiveTree(gp.genGrow(pset, min_, max_, type_))


def genRuleSet(pset, min_, max_, type_=None, config=None):
    rules_len = random.randint(config.min_rules, config.max_rules)
    return [genRule(pset, min_, max_, type_) for _ in range(rules_len)]


class RuleSet(list):
    """Subclass of list that contains lists, used to hold a sets of fuzzy rules.
    len(ruleset) will return the total length of all the contained lists.
    The ruleset.length property will return the length of the top level list.
    """

    def __len__(self):
        return sum(len(item) for item in self)

    @property
    def length(self):
        return super().__len__()


def registerCreators(
    toolbox: base.Toolbox,
    config: CreatorConfig,
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

    pset = _makePrimitiveSet(antecendents, consequents)
    creator.create("Individual", RuleSet, fitness=creator.RuleSetFitness, pset=pset)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "expr",
        genRule,
        pset=pset,
        min_=config.min_tree_height,
        max_=config.max_tree_height,
    )
    toolbox.register(
        "rules_expr",
        genRuleSet,
        pset=pset,
        min_=config.min_tree_height,
        max_=config.max_tree_height,
        config=config,
    )
    toolbox.register(
        "individualCreator", tools.initIterate, creator.Individual, toolbox.rules_expr
    )
    toolbox.register(
        "populationCreator", tools.initRepeat, list, toolbox.individualCreator
    )
    toolbox.register("get_pset", identity, pset)
    return pset


def ea_with_elitism_and_replacement(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    replacements,
    stats=None,
    tensorboard_writer=None,
    halloffame=None,
    verbose=True,
    slices=None,
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
    :param tensorboard_writer: Optional tensorboard SummaryWriter instance to log
            results to tensorboard.
    :param halloffame: DEAP HallOfFame instance for recording the top performing individuals
    :param verbose: boolean flag - if True then print stats while running
    :param slices: optional list of slice objects to run EA on small batches
    :return: final population and logbook

    """

    if slices is None:
        slices = [slice(0, BIG_INT)]

    def evaluate_population(pop, slice):
        # Evaluate the individuals in population pop with an invalid fitness
        invalid_population = [ind for ind in pop if not ind.fitness.valid]
        # prune any rules that have not been evaluated
        prune_population(invalid_population)
        with Pool() as pool:
            fitnesses = pool.starmap(
                toolbox.evaluate, zip(invalid_population, repeat(slice))
            )

        for ind, fit in zip(invalid_population, fitnesses):
            ind.fitness.values = fit
        return len(invalid_population)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "fitness", "size"
    logbook.chapters["fitness"].header = "max", "avg"
    logbook.chapters["size"].header = "min", "avg", "best"

    invalid_count = evaluate_population(population, slices[0])

    if halloffame is not None:
        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0
    else:
        hof_size = 0

    write_stats(
        population, 0, invalid_count, verbose, logbook, stats, tensorboard_writer
    )

    for gen in range(1, ngen):
        invalid_count = 0
        print("Batch: ", end="")
        for idx, slice in enumerate(slices):
            print(idx, end=", ")
            offspring = toolbox.select(population, len(population) - hof_size)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            invalid_count += evaluate_population(offspring, slice)

            # replace the worst performing individuals with newly generated ones
            _replace_worst(toolbox, offspring, replacements)
            invalid_count += evaluate_population(offspring, slice)

            if halloffame:
                offspring.extend(halloffame.items)
                halloffame.update(offspring)

            population[:] = offspring

        print()
        write_stats(
            population, gen, invalid_count, verbose, logbook, stats, tensorboard_writer
        )

    return population, logbook


def write_stats(
    population, generation, invalid_count, verbose, logbook, stats, tensorboard_writer
):
    record = stats.compile(population) if stats else {}
    logbook.record(gen=generation, nevals=invalid_count, **record)
    if verbose:
        print(logbook.stream)
    if tensorboard_writer:
        for (name, val) in record.items():
            if isinstance(val, dict):
                for (subname, subval) in val.items():
                    tensorboard_writer.add_scalar(
                        f"{name}/{subname}", subval, generation
                    )
            else:
                tensorboard_writer.add_scalar(name, val, generation)
        tensorboard_writer.add_histogram(
            "fitnesses", np.array([i.fitness.values[0] for i in population]), generation
        )
        tensorboard_writer.add_histogram(
            "sizes", np.array([len(i) for i in population]), generation
        )
        tensorboard_writer.add_histogram(
            "rule_count", np.array([i.length for i in population]), generation
        )


def _replace_worst(toolbox, population, replacements):
    """Find the worst performers and replace them with new random individuals.
    N.B. the population is updated in-place.

    :param toolbox:
    :param population: list of individuals
    :param replacements: number of individuals to replace
    :return:
    """
    replacement_idx = heapq.nsmallest(
        replacements,
        ((ind.fitness.values, idx) for (idx, ind) in enumerate(population)),
    )
    for (_, idx) in replacement_idx:
        population[idx] = toolbox.individualCreator()


def prune_rule(rule):
    pos = 0
    while pos < len(rule):
        name = rule[pos].name
        # if there are two consecutive inverts then delete them both
        if name == "invert" and rule[pos + 1].name == "invert":
            del rule[pos : pos + 2]
            continue
        if name in ("and_", "or_"):
            # merge duplicate branches
            lhs = rule.searchSubtree(pos + 1)
            rhs = rule.searchSubtree(lhs.stop)
            if rule[lhs] == rule[rhs]:
                rule[pos : rhs.stop] = rule[lhs]
                continue
        pos += 1
    return rule


def prune_population(population):
    for ind in population:
        for rule in ind:
            prune_rule(rule)
