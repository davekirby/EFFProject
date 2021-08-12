from datetime import datetime
from pathlib import Path
import tensorboardX
import gym
from evofuzzy.gymrunner import GymRunner, Action
from evofuzzy.fuzzybase import FuzzyBase, make_antecedent, make_consequents

tensorboard_dir = "tb_logs/mountaincar-v0"
if tensorboard_dir:
    logdir = Path(f"{tensorboard_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

env = gym.make("MountainCarContinuous-v0")
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
    make_antecedent("obs_1", -1.2, 0.6),
    make_antecedent("obs_2", -0.07, 0.07),
]

action = Action(-1, 1, 5)

runner.train(env, antecedents, action, tensorboard_writer)
print(runner.best_str)
runner.play(env)
