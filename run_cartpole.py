from datetime import datetime
from pathlib import Path
import tensorboardX
import gym
from evofuzzy.gymrunner import GymRunner
from evofuzzy.fuzzybase import FuzzyBase, make_antecedent

tensorboard_dir = "tb_logs/cartpole-v0"
if tensorboard_dir:
    logdir = Path(f"{tensorboard_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

env = gym.make("CartPole-v1")
runner = GymRunner(
    population_size=50,
    hall_of_fame_size=1,
    max_generation=10,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=1,
    max_tree_height=3,
    max_rules=4,
    whole_rule_prob=0.2,
    tree_height_limit=5,
)

antecedents = [
    make_antecedent("position", -2.4, 2.4),
    make_antecedent("velocity", -1, 1),
    make_antecedent("angle", -0.25, 0.25),
    make_antecedent("angular_velocity", -2, 2),
]

runner.train(env, tensorboard_writer, antecedents)
print(runner.best_str)
runner.play(env)
