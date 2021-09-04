from datetime import datetime
from pathlib import Path
import tensorboardX
import gym
from evofuzzy import GymRunner

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
    elite_size=1,
    n_iter=10,
    mutation_prob=0.9,
    crossover_prob=0.2,
    min_tree_height=1,
    max_tree_height=3,
    max_rules=4,
    whole_rule_prob=0.2,
    tree_height_limit=5,
)

runner.train(env, tensorboard_writer)
print(runner.best_str)
reward = runner.play(env)
print("Reward:", reward)
