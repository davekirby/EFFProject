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
    """Subclass of list that contains lists, used to hold a set of fuzzy rules.
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
    if not hasattr(creator, "RuleSetFitness"):
        creator.create("RuleSetFitness", base.Fitness, weights=(1.0,))

    if hasattr(creator, "Individual"):
        del creator.Individual
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
    replacement_size=0,
    stats=None,
    tensorboard_writer=None,
    hof_size=1,
    verbose=True,
    slices=None,
    always_evalute=False,
    forgetting=1,
):
    """Modified version of the DEAP eaSimple function to run the evolution process
    while keeping the top performing members in the HallOfFame from one generation to the next.

    :param population: The initial population
    :param toolbox: the deap toolbox with functions registered on it
    :param cxpb: crossover probability 0 <= cxpb <= 1
    :param mutpb: mutation probability 0 <= mupb <= 1
    :param ngen: number of generations to run the evolution for
    :param replacement_size: number of poor performers to replace with new individuals
    :param stats: DEAP Stats instance for recording statistics
    :param tensorboard_writer: Optional tensorboard SummaryWriter instance to log
            results to tensorboard.
    :param hof_size: the number of top performers to carry over to the next generation
    :param verbose: boolean flag - if True then print stats while running
    :param slices: optional list of slice objects to run EA on small batches
    :param always_evalute: flag to force evalutation of fitness
    :param forgetting: value between 0 and 1 to control how much weight to put on previous fitness
            values
    :return: final population and logbook

    """

    if slices is None:
        batched = False
        slices = [slice(0, BIG_INT)]
    else:
        batched = True

    def evaluate_population(population, batch_slice):
        if not always_evalute and not batched:
            # Only evaluate the individuals in the population that have not been evaluated already
            population = [ind for ind in population if not ind.fitness.valid]
        # prune the rules that are going to be evaluated
        prune_population(population)
        with Pool() as pool:
            fitnesses = pool.starmap(
                toolbox.evaluate, zip(population, repeat(batch_slice))
            )

        for ind, fit in zip(population, fitnesses):
            if ind.fitness.valid:
                old_fitness = ind.fitness.values
                new_fit = fit[0] * forgetting + (old_fitness[0] * (1 - forgetting))
                ind.fitness.values = (new_fit,)
            else:
                ind.fitness.values = fit

    logbook = tools.Logbook()
    logbook.header = "gen", "fitness", "size"
    logbook.chapters["fitness"].header = "max", "avg"
    logbook.chapters["size"].header = "min", "avg", "best"

    evaluate_population(population, slices[0])
    population.sort(key=lambda ind: ind.fitness.values)

    write_stats(population, 0, verbose, logbook, stats, tensorboard_writer)

    for gen in range(1, ngen):
        if batched and verbose:
            print("Batch: ", end="")
        for idx, slice_ in enumerate(slices):
            if batched and verbose:
                print(idx, end=", ")

            replacements = toolbox.populationCreator(replacement_size)

            offspring = toolbox.select(
                population[replacement_size:],
                len(population) - (hof_size + replacement_size),
            )
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            if hof_size:
                offspring.extend(population[-hof_size:])
            if replacement_size:
                offspring.extend(replacements)

            population[:] = offspring
            evaluate_population(population, slice_)
            population.sort(key=lambda ind: ind.fitness.values)

        write_stats(population, gen, verbose, logbook, stats, tensorboard_writer)

    return population, logbook


def write_stats(population, generation, verbose, logbook, stats, tensorboard_writer):
    record = stats.compile(population) if stats else {}
    logbook.record(gen=generation, **record)
    if verbose:
        print()
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
