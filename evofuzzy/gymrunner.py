import numpy as np
from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase, make_consequents


class GymRunner(FuzzyBase):
    always_evaluate_ = True
    def train(self, env, antecendents, actions, tensorboard_writer):
        self.antecedents_ = antecendents
        self.consequents_ = make_consequents(actions)
        self.actions_ = actions

        self.initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, env=env)
        return self.execute(None, tensorboard_writer)

    def play(self, env):
        """Display the best individual playing in the evironment"""
        reward = self._evaluate(self.best, None, env, True)
        print("Finished with reward of", reward[0])

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
        for _ in range(1000):
            if render:
                env.render()
            action = self._evaluate_action(observation, simulator, antecedents)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return (total_reward,)

    def _evaluate_action(self, observation, simulator, antecedents):
        action_names = list(self.actions_.keys())
        action_vals = list(self.actions_.values())
        obs_vals = {
            ant.label: ob
            for (ant, ob) in zip(self.antecedents_, observation)
            if ant.label in antecedents
        }
        simulator.inputs(obs_vals)
        simulator.compute()
        action_idx = np.argmax([simulator.output.get(name, 0) for name in action_names])
        action_val = action_vals[action_idx]
        return action_val
