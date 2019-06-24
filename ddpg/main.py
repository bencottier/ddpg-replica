"""
main.py

author: bencottier
"""
from ddpg import ddpg
import core
import gym
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--exp_variant', default=None)
    parser.add_argument('--logdir', default='out')
    parser.add_argument('--seeds', type=int, default=0, nargs='*')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--polyak', type=float, default=0.001)
    parser.add_argument('--exploration_steps', type=int, default=0)
    parser.add_argument('--rand_proc', default='ou')
    parser.add_argument('--rand_proc_kwargs', type=json.loads, default=dict())
    args = parser.parse_args()

    seeds = args.seeds if isinstance(args.seeds, list) else [args.seeds]
    rand_proc_dir = {'normal': core.NormalProcess, 
                     'ou': core.OrnsteinUhlenbeckProcess}
    rand_proc = rand_proc_dir[args.rand_proc]

    for seed in seeds:
        print("\nNEW EXPERIMENT: SEED {}\n".format(seed))
        ddpg(env_name=args.env_name, exp_name=args.exp_name, exp_variant=args.exp_variant,
                logdir='out', seed=seed, 
                epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, 
                batch_size=args.batch_size, discount=args.discount,
                polyak=args.polyak, exploration_steps=args.exploration_steps,
                rand_proc=rand_proc, rand_proc_kwargs=args.rand_proc_kwargs)
