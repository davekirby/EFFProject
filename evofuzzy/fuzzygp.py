import heapq
import random
from typing import Any, List, NamedTuple
import operator
from multiprocessing import Pool
from functools import partial

from deap import creator, base, algorithms, gp, tools
from skfuzzy.control import Rule, Antecedent, Consequent
from skfuzzy.control.term import Term


def ident(x: Any) -> Any:
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

    pset.addEphemeralConstant("consequents", makeConsequents, list)
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


def genRule(pset, min_, max_, type_=None):
    return gp.PrimitiveTree(gp.genGrow(pset, min_, max_, type_))


def genRuleSet(pset, min_, max_, type_=None, config=None):
    rules_len = random.randint(config.min_rules, config.max_rules)
    return [genRule(pset, min_, max_, type_) for _ in range(rules_len)]


class ListOfLists(list):
    """Subclass of list that only contains list.
    len(lol) will return the total length of all the sublists.
    lol.length will return the length of the top level list.
    """

    def __len__(self):
        return sum(len(item) for item in self)

    @property
    def length(self):
        return super().__len__()


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

    pset = _makePrimitiveSet(antecendents, consequents)
    creator.create("Individual", ListOfLists, fitness=creator.RuleSetFitness, pset=pset)
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
    return pset


def eaSimpleWithElitism(
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
    :return: final population and logbook

    """

    def evaluate_population(pop):
        # Evaluate the individuals in population pop with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        with Pool() as pool:
            fitnesses = pool.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "fitness", "size"
    logbook.chapters["fitness"].header = "min", "avg"
    logbook.chapters["size"].header = "min", "avg"

    invalid_count = evaluate_population(population)

    if halloffame is not None:
        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0
    else:
        hof_size = 0

    write_stats(
        population, 0, invalid_count, verbose, logbook, stats, tensorboard_writer
    )

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_count = evaluate_population(offspring)

        # replace the worst performing individuals with newly generated ones
        _replace_worst(toolbox, offspring, replacements)
        invalid_count += evaluate_population(offspring)

        if halloffame:
            offspring.extend(halloffame.items)
            halloffame.update(offspring)

        population[:] = offspring

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
        pass  # TODO: add tensorboard support



def _replace_worst(toolbox, population, replacements):
    """Find the worst performers and replace them with new random individuals.
    N.B. the population is updated in-place.

    :param toolbox:
    :param population: list of individuals
    :param replacements: number of individuals to replace
    :return:
    """
    replacement_idx = heapq.nlargest(
        replacements,
        ((ind.fitness.values, idx) for (idx, ind) in enumerate(population)),
    )
    for (_, idx) in replacement_idx:
        population[idx] = toolbox.individualCreator()
