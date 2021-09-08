import random
from itertools import repeat, count
from typing import Any, List, NamedTuple, Dict, Iterable, Tuple, Optional
import operator
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from deap import creator, base, algorithms, gp, tools
from skfuzzy import control as ctrl
from skfuzzy.control import Rule, Antecedent, Consequent
from skfuzzy.control.term import Term

BIG_INT = 1 << 64

consequents_counter = count()

class RuleSet(list):
    """Subclass of list that contains gp.PrimitiveTree instances, used to hold a set of fuzzy rules.
    len(ruleset) will return the total length of all the contained primitive trees.
    The ruleset.length property will return the length of the top level list.
    """

    def __len__(self):
        return sum(len(item) for item in self)

    @property
    def length(self):
        return super().__len__()


creator.create("RuleSetFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", RuleSet, fitness=creator.RuleSetFitness)


def identity(x: Any) -> Any:
    """Identity function - returns the parameter unchanged.
    Used to enable terminal/ephemeral values to be created below a tree's minimum depth.
    """
    return x


class MakeConsequents:
    """Ephemeral constant that randomly generates consequents for the rules.
    This needs to be a class so that the repr can be defined to return something
    that can be used by toolbox.compile.
    :param cons_terms: dict mapping a skfuzzy Consequent name to a list of "name['term']"
                       strings.
    """

    def __init__(self, cons_terms: Dict[str, str]):
        max_consequents = len(cons_terms) // 2 + 1
        sample_size = random.randint(1, max_consequents)
        candidates = random.sample(list(cons_terms.values()), sample_size)
        self.values = [random.choice(term) for term in candidates]

    def __repr__(self):
        return f"[{', '.join(value for value in self.values)}]"


def _make_primitive_set(
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
    make_consequents = partial(MakeConsequents, cons_terms)

    pset = gp.PrimitiveSetTyped("Rule", [], Rule)

    for ant in antecendents:
        pset.context[ant.label] = ant
        for name, term in ant.terms.items():
            pset.addTerminal(term, Term, f"{ant.label}['{name}']")

    for cons in consequents:
        pset.context[cons.label] = cons

    # The ephemeral constants must have a globally unique name
    consequents_name = f"consequents_{next(consequents_counter)}"
    pset.addEphemeralConstant(consequents_name, make_consequents, list)
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


def _generate_rule(pset: gp.PrimitiveSetTyped, min_: int, max_: int, type_=None):
    """
    Return a randomly generated PrimitiveTree encoding a fuzzy rule.
    :param pset: The PrimitiveSetTyped to draw the node types from
    :param min_: minimum tree height
    :param max_: maximum tree height
    :param type_:
    :return: the generated primitiveTree
    """
    return gp.PrimitiveTree(gp.genGrow(pset, min_, max_, type_))


def _generate_rule_set(
    pset: gp.PrimitiveSetTyped, type_=None, config: CreatorConfig = None
):
    rules_len = random.randint(config.min_rules, config.max_rules)
    return RuleSet(
        _generate_rule(pset, config.min_tree_height, config.max_tree_height, type_)
        for _ in range(rules_len)
    )


def register_primitiveset_and_creators(
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

    :param toolbox: deap.base.Toolbox instance
    :param config:  Config instance holding hyperparameters
    :param antecendents: list of fuzzy antecendents used by the rules
    :param consequents: list of fuzzy consequents used by the rules
    :return: The PrimitiveSet that has been created
    """

    pset = _make_primitive_set(antecendents, consequents)

    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "expr",
        _generate_rule,
        pset=pset,
        min_=config.min_tree_height,
        max_=config.max_tree_height,
    )
    toolbox.register(
        "rules_expr",
        _generate_rule_set,
        pset=pset,
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
    elite_size=1,
    verbose=True,
    slices=None,
    always_evaluate=False,
    memory_decay=1,
):
    """Modified version of the DEAP eaSimple function to run the evolution process
    while keeping the top performing members from one generation to the next and replacing
    poor performers with new individuals.

    :param population: The initial population
    :param toolbox: the deap toolbox with functions registered on it
    :param cxpb: crossover probability 0 <= cxpb <= 1
    :param mutpb: mutation probability 0 <= mupb <= 1
    :param ngen: number of generations to run the evolution for
    :param replacement_size: number of poor performers to replace with new individuals
    :param stats: DEAP Stats instance for recording statistics
    :param tensorboard_writer: Optional tensorboardX SummaryWriter instance to log
            results to tensorboard.
    :param elite_size: the number of top performers to carry over to the next generation
    :param verbose: boolean flag - if True then print stats while running
    :param slices: optional list of slice objects to run EA on small batches
    :param always_evaluate: flag to force evaluation of fitness
    :param memory_decay: value between 0 and 1 to control how much weight to put on previous fitness
            values
    :return: final population and logbook

    """

    if slices is None:
        batched = False
        slices = [slice(0, BIG_INT)]
    else:
        batched = True

    def evaluate_population(population: List[RuleSet], batch_slice: slice):
        if not always_evaluate and not batched:
            # Only evaluate the individuals in the population that have not been evaluated already
            population = [ind for ind in population if not ind.fitness.valid]
        # prune the rules that are going to be evaluated
        _prune_population(population)
        with Pool() as pool:
            fitnesses = pool.starmap(
                toolbox.evaluate, zip(population, repeat(batch_slice))
            )

        for ind, fit in zip(population, fitnesses):
            if ind.fitness.valid:
                old_fitness = ind.fitness.values
                new_fit = fit[0] * memory_decay + (old_fitness[0] * (1 - memory_decay))
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

    for gen in range(1, ngen+1):
        if batched and verbose:
            print("Batch: ", end="")
        for idx, slice_ in enumerate(slices):
            if batched and verbose:
                print(idx, end=", ")

            replacements = toolbox.populationCreator(replacement_size)

            offspring = toolbox.select(
                population[replacement_size:],
                len(population) - (elite_size + replacement_size),
            )
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            if elite_size:
                offspring.extend(population[-elite_size:])
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


def _prune_rule(rule: gp.PrimitiveTree):
    """
    Remove redundancy in a fuzzy rule.  ie:
    - `NOT NOT X` is converted to X
    - `X AND X` is converted to X
    - `X OR X` is converted to X
    rules are modified in-place
    :param rule: the rule to prune
    """
    pos = 0
    while pos < len(rule):
        name = rule[pos].name
        # if there are two consecutive inverts then delete them both
        if name == "invert" and rule[pos + 1].name == "invert":
            del rule[pos: pos + 2]
            continue
        if name in ("and_", "or_"):
            # merge duplicate branches
            lhs = rule.searchSubtree(pos + 1)
            rhs = rule.searchSubtree(lhs.stop)
            if rule[lhs] == rule[rhs]:
                rule[pos : rhs.stop] = rule[lhs]
                continue
        pos += 1


def _prune_population(population: List[RuleSet]):
    """
    Prune all the rules in a population
    :param population: list of RuleSets to prune
    """
    for ind in population:
        for rule in ind:
            _prune_rule(rule)


def mate_rulesets(
    individual_1: RuleSet, individual_2: RuleSet, whole_rule_prob: float
) -> Tuple[RuleSet, RuleSet]:
    """Mate two individuals by randomly selecting two rules, one from each individual then either
      - swapping the rules over completely
      or
      - swapping randomly selected subtrees of the two rules
    N.B. The individuals are modified in-place as well as returned from the function.

    :param individual_1:  First individual
    :param individual_2:  Second individual
    :param whole_rule_prob: probability of swapping entire rules instead of sub-trees
    :return: the modified individuals
    """
    rule1_idx = random.randint(0, individual_1.length - 1)
    rule2_idx = random.randint(0, individual_2.length - 1)
    if random.random() < whole_rule_prob:
        # swap entire rules over
        rule2 = individual_1[rule1_idx]
        rule1 = individual_2[rule2_idx]
    else:
        rule1, rule2 = gp.cxOnePoint(individual_1[rule1_idx], individual_2[rule2_idx])
    individual_1[rule1_idx] = rule1
    individual_2[rule2_idx] = rule2
    return individual_1, individual_2


def mutate_ruleset(
    toolbox: base.Toolbox, individual: RuleSet, whole_rule_prob: float
) -> Tuple[RuleSet]:
    """Mutate an individual by selecting a rule then either:
    - replacing the entire rule with a newly generated one
    or
    - selecting a subtree of the rule and replacing it with a newly generated subtree
    N.B. The individual is modified in-place as well as returned from the function

    :param toolbox: deap.base.Toolbox instance that has been initialised with the
    register_primitiveset_and_creators function.
    :param individual:  the individual to mutate
    :param whole_rule_prob: probability of replacing an entire rule instead of a sub-tree
    :return: the modified individual in a 1-tuple
    """
    rule_idx = random.randint(0, individual.length - 1)
    if random.random() < whole_rule_prob:
        rule = toolbox.expr()
    else:
        (rule,) = gp.mutUniform(
            individual[rule_idx], expr=toolbox.expr, pset=toolbox.get_pset()
        )
    individual[rule_idx] = rule
    return (individual,)


def get_fitness_values(individual: RuleSet) -> Tuple[float]:
    """Return the fitness of an individual.  N.B. in DEAP the fitness is always a tuple

    :param individual:
    :return: Tuple with the fitness value(s) as floats
    """
    return individual.fitness.values


def make_antecedent(
    name: str,
    min: float,
    max: float,
    terms: Optional[List[str]] = None,
    inf_limit: Optional[float] = None,
) -> ctrl.Antecedent:
    """Create a skfuzzy Antecedent object and initialise the terms on it.

    :param name: The name of the antecedent
    :param min: minimum value that the antecedent can take
    :param max:  minimum value that the antecedent can take
    :param terms: optional names of the terms that are defined on the antecedent.
                  If omitted then the default names are used:
                    "
    :param inf_limit: if the min and/or max are +/-inf then replace them with +/- this value.
                      This can happen when taking the min and max from a Gym environment observation

    :return: the created Antecedent instance
    """
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
