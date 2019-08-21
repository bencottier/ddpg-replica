#!/usr/bin/bash

conda activate spinningup

python ~/ddpg-replica/ddpg/main.py HalfCheetah-v2 --exp_name minitest --exp_variant wd_1e-2_normal --seeds 0 --epochs 20 --steps_per_epoch 5000 --exploration_steps 0 --rand_proc normal --rand_proc_kwargs {\"scale\": 0.1}

# python ~/ddpg-replica/ddpg/main.py HalfCheetah-v2 --exp_name benchmark --exp_variant wd_1e-2_normal --seeds 0 10 20 30 40 50 60 70 80 90 --epochs 600 --steps_per_epoch 5000 --exploration_steps 0 --rand_proc normal --rand_proc_kwargs {\"scale\": 0.1}
