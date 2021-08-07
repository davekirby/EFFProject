import numpy as np
from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase, make_consequents


class GymRunner(FuzzyBase):
    def run(self, env, antecendents, actions, tensorboard_writer):
        self.antecedents_ = antecendents
        self.consequents_ = make_consequents(actions)
        self.actions_ = actions

        self.initialise(tensorboard_writer)

        if hasattr(self.toolbox_, "evaluate"):
            del self.toolbox_.evaluate
        self.toolbox_.register("evaluate", self._evaluate, env=env)
        return self.execute(None, tensorboard_writer)

    def _evaluate(self, individual, batch, env):
        """Evaluate one individual by running the gym environment with the
        fuzzy rules represented by the individual as the agent.
        """
        total_reward = 0
        rules = [self.toolbox_.compile(rule) for rule in individual]
        controller = ctrl.ControlSystem(rules)
        simulator = ctrl.ControlSystemSimulation(controller)

        observation = env.reset()
        for _ in range(self.max_generation):
            action = self._evaluate_action(observation, simulator)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def _evaluate_action(self, observation, simulator):
        action_names = list(self.actions_.keys())
        action_vals = list(self.actions_.values())
        obs_vals = {ant.label: ob for (ant, ob) in zip(self.antecedents_, observation)}
        simulator.inputs(obs_vals)
        simulator.compute()
        action_idx = np.argmax([simulator.output.get(name, 0) for name in action_names])
        action_val = action_vals[action_idx]
        return action_val
