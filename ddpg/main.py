"""
main.py

author: bencottier
"""
from ddpg import ddpg
import gym


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    # env.reset()
    # done = False
    # while not done:
    #     env.render()
    #     obs, rwd, done, info = env.step(env.action_space.sample()) # take a random action
    # env.close()

    for seed in range(0, 100, 10):
        print("\nNEW EXPERIMENT: SEED {}\n".format(seed))
        ddpg(env_name='HalfCheetah-v2', discount=0.99, batch_size=64, polyak=0.001, 
                epochs=50, steps_per_epoch=5000, logdir='out', seed=seed)
