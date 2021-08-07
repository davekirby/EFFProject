## Links:
http://gym.openai.com/ - home page
https://gym.openai.com/docs/ - docs
https://github.com/openai/gym/wiki/FAQ
https://github.com/openai/gym - github
https://github.com/openai/gym/wiki/Table-of-environments 

## Commands:

env = gym.make("env-name") - create an environment

env.reset() - returns initial observation

env.observation_space.n - the number of possible states

env.observation_space - the shape of the observation space

env.observation_space.high - maximum values

env.observation_space.low - minimum values

env.render() - display current environment state

env.action_space.n - number of possible actions, usually 0 to n-1

`python gym/examples/scripts/list_envs` to list all environments
(assumes gym is checked out in current directory).
Alternatively `print(gym.envs.registry.all())`

`python gym/gym/utils/play.py` or `python -m gym.utils.play`  plays an atari game - defaults to Montezuma's revenge, use `--env <envname>` for others.

## Spaces
used for observations and actions.

### Discrete
Fixed range of non-negative numbers

### Box
An array of n numbers



