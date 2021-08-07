from skfuzzy import control as ctrl

from .fuzzybase import FuzzyBase


class GymRunner(FuzzyBase):

    def run(self, env, antecendents, consequents, tensorboard_writer):
        self.antecedents_ = antecendents
        self.consequents_ = consequents

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
        observation = env.reset()
        for _ in range(self.max_generation):
            action = self._evaluate_action(env, individual)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def _evaluate_action(self, env, individual):
        return env.action_space.sample()

