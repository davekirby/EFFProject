from itertools import count
from typing import Optional

import gym
import numpy as np
from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase
from .fuzzygp import make_antecedent, make_binary_consequents


def _make_box_consequent(name, low, high):
    cons = ctrl.Consequent(np.linspace(low, high, 11), name, "som")
    cons.automf(5, "quant")
    return cons


def _antecedents_from_env(env: gym.Env, inf_limit: Optional[float] = None):
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


def consequents_from_env(env: gym.Env):
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

    def train(self, env, tensorboard_writer, antecedents=None, inf_limit=None):
        if antecedents:
            self.antecedents_ = antecedents
        else:
            self.antecedents_ = _antecedents_from_env(env, inf_limit)

        self.consequents_, self.box_actions_ = consequents_from_env(env)

        self.initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, env=env)
        self.execute(None, tensorboard_writer)

    def play(self, env, n=1):
        """Display the best individual playing in the environment

        :param env: the Gym environment to play
        :param n: the number of top performing individuals to use - defaults to 1
        :return: the total reward
        """
        reward = self._evaluate(self.best_n(n), None, env, True)
        env.close()
        return reward[0]

    def _evaluate(self, individual, batch, env, render=False):
        """Evaluate one individual by running the gym environment with the
        fuzzy rules represented by the individual as the agent.
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

    def _evaluate_action(self, observation, simulator, antecedents):
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

    def _evaluate_discrete_actions(self, simulator):
        action_names = [cons.label for cons in self.consequents_]
        return np.argmax([simulator.output.get(name, 0) for name in action_names])

    def _evaluate_continuous_actions(self, simulator):
        return [
            simulator.output.get(f"action_{i}", 0)
            for i in range(len(self.consequents_))
        ]
