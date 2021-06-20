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
