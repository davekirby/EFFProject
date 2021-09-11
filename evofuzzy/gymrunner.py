from itertools import count
from typing import Optional, List, Tuple, Union, Iterable

import gym
import numpy as np
import tensorboardX
from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase
from .fuzzygp import make_antecedent, make_binary_consequents, RuleSet


def _make_box_consequent(name: str, low: float, high: float) -> ctrl.Consequent:
    """Create a Consequent for use with the Box action space.
    :param name: name of the action
    :param low: Minimum value
    :param high: Maximum value
    :return: Consequent
    """
    cons = ctrl.Consequent(np.linspace(low, high, 11), name, "som")
    cons.automf(5, "quant")
    return cons


def _antecedents_from_env(env: gym.Env, inf_limit: float) -> List[ctrl.Antecedent]:
    """Create antecedents from a Box observation space.

    :param env: gym Environment
    :param inf_limit: limits to use if the observation_space says they are "inf"
    :return: list of Antecedents
    """
    observations = env.observation_space
    assert isinstance(
        observations, gym.spaces.Box
    ), "Only Box observation spaces supported"
    assert (
        len(observations.shape) == 1
    ), "Only one dimensional observation spaces supported"
    return [
        make_antecedent(f"obs_{i}", low, high, inf_limit=inf_limit)
        for (i, low, high) in zip(count(), observations.low, observations.high)
    ]


def _consequents_from_env(env: gym.Env) -> Tuple[List[ctrl.Consequent], bool]:
    """Create the consequents from the gym environment's action_space.

    :param env:  Gym environment
    :return: tuple with list of consequents and a flag to indicate if it is a Box or Discrete space
    """
    actions = env.action_space
    if isinstance(actions, gym.spaces.Box):
        assert len(actions.shape) == 1, "Only one dimensional action spaces supported"
        return (
            [
                _make_box_consequent(f"action_{i}", low, high)
                for (i, low, high) in zip(count(), actions.low, actions.high)
            ],
            True,
        )
    assert isinstance(
        actions, gym.spaces.Discrete
    ), "Only Box and Discrete actions supported"
    return make_binary_consequents(f"action_{i}" for i in range(actions.n)), False


class GymRunner(FuzzyBase):
    always_evaluate_ = True

    def train(
        self,
        env: gym.Env,
        tensorboard_writer: Optional["tensorboardX.SummaryWriter"] = None,
        antecedents: Optional[List[ctrl.Antecedent]] = None,
        inf_limit: float = 100.0,
    ):
        """Train the GymRunner against the gym environment.

        :param env: The environment
        :param tensorboard_writer: Optional object to write information to TensorBoard
        :param antecedents: Optional list of Antecendent objects to use
        :param inf_limit: limit to use if the environment limits are "inf"
        :return: None
        """
        if antecedents:
            self.antecedents_ = antecedents
        else:
            self.antecedents_ = _antecedents_from_env(env, inf_limit)

        self.consequents_, self.box_actions_ = _consequents_from_env(env)

        self._initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, env=env)
        self.execute(None, tensorboard_writer)

    def play(self, env: gym.Env, n: int = 1) -> float:
        """Display the best individual playing in the environment

        :param env: the Gym environment to play
        :param n: the number of top performing individuals to use - defaults to 1
        :return: the final reward
        """
        reward = self._evaluate(self.best_n(n), None, env, True)
        env.close()
        return reward[0]

    def _evaluate(
        self, individual: RuleSet, batch: None, env: gym.Env, render: bool = False
    ) -> Tuple[float]:
        """Evaluate one individual by running the gym environment with the
        fuzzy rules represented by the individual as the agent.

        :param individual: The individual to evaluate
        :param batch: Not used - needed for compatibility with FuzzyClassifier
        :param env: gym environment to use
        :param render: if true then render the scene every timestep
        :return: tuple containing the total reward
        """
        total_reward = 0
        rules = [self.toolbox_.compile(rule) for rule in individual]
        antecedents = {
            term.parent.label for rule in rules for term in rule.antecedent_terms
        }
        controller = ctrl.ControlSystem(rules)
        simulator = ctrl.ControlSystemSimulation(controller)

        observation = env.reset()
        while True:
            if render:
                env.render()
            action = self._evaluate_action(observation, simulator, antecedents)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return (total_reward,)

    def _evaluate_action(
        self,
        observation: gym.spaces.Box,
        simulator: ctrl.ControlSystemSimulation,
        antecedents: Iterable[str],
    ) -> Union[int, List[float]]:
        """Run the FIS on the current observation and return the action to take.

        :param observation: a Box observation space
        :param simulator: scikit-fuzzy controller
        :param antecedents: set of antecedent names
        :return: the action to take.  Will be float or int, depending on if the action
        space is Box or Discrete
        """
        obs_vals = {
            ant.label: ob
            for (ant, ob) in zip(self.antecedents_, observation)
            if ant.label in antecedents
        }
        simulator.inputs(obs_vals)
        simulator.compute()
        if self.box_actions_:
            return self._evaluate_continuous_actions(simulator)
        else:
            return self._evaluate_discrete_actions(simulator)

    def _evaluate_discrete_actions(
        self, simulator: ctrl.ControlSystemSimulation
    ) -> int:
        """Get the Consequent values from the simulator and convert them to a Discrete value

        :param simulator: scikit-fuzzy controller
        :return: Discrete value for the action
        """
        action_names = [cons.label for cons in self.consequents_]
        return np.argmax([simulator.output.get(name, 0) for name in action_names])

    def _evaluate_continuous_actions(
        self, simulator: ctrl.ControlSystemSimulation
    ) -> List[float]:
        """Get the Consequent values from the simulator and return them as a list of actions.

        :param simulator: scikit-fuzzy controller
        :return: list of floats for the actions
        """
        return [
            simulator.output.get(f"action_{i}", 0)
            for i in range(len(self.consequents_))
        ]
