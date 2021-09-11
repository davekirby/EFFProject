from datetime import datetime
from pathlib import Path
import tensorboardX
import gym
from evofuzzy import GymRunner

""" Script to run GymRunner on an environment in an ipython or Jupyter shell. 
To use, set `env_name` in the shell then %run this file.
"""

tensorboard_dir = f"tb_logs/{env_name}"
if tensorboard_dir:
    logdir = Path(f"{tensorboard_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

env = gym.make(env_name)
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
)

runner.train(env, tensorboard_writer)
print(runner.best_str)
reward = runner.play(env)
print("Reward:", reward)
