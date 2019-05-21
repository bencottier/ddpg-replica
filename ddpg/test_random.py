import random
import numpy as np
import tensorflow as tf
import gym
from run_random import run


if __name__ == '__main__':
    seed = 57
    n = 5

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    env = gym.make('HalfCheetah-v2')
    env.seed(seed)

    run(n, env)
