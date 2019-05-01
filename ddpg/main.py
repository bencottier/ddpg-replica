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

    env = gym.make('HalfCheetah-v2')
    ddpg(env, discount=0.99, batch_size=64, polyak=0.001)
