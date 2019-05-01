"""
ddpg.py

author: bencottier
"""
from core import mlp_actor_critic, placeholders
import gym
import tensorflow as tf
import numpy as np


def ddpg(env, discount, batch_size, polyak, seed=0, 
        actor_critic=mlp_actor_critic, ac_kwargs=dict()):

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    action_space = env.action_space
    act_dim = action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    x_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(tf.float32, shape=(None, act_dim))
    x2_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    r_ph = tf.placeholder(tf.float32, shape=(None,))
    d_ph = tf.placeholder(tf.float32, shape=(None,))

    with tf.variable_scope('actor-critic'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, action_space, **ac_kwargs)

    with tf.variable_scope('target'):  # scope helps group the vars for target update
        pi_targ, q_targ, q_pi_targ = actor_critic(x2_ph, a_ph, action_space, **ac_kwargs)

    # Use "done" variable to cancel future value when at end of episode
    # The stop_gradient means inputs to the operation will not factor into gradients
    backup = tf.stop_gradient(r_ph + d_ph * discount * q_pi_targ)
    q_loss = tf.reduce_mean((backup - q)**2)
    # From memory. Not quite clear how this relates to equation in paper.
    pi_loss = -tf.reduce_mean(q)

    # Update targets
    # TODO Not sure if correct. Even if correct, it's not how I remember the baseline.
    ac_vars = [v for v in tf.trainable_variables() 
               if ('actor-critic' in v.name and 'dense' in v.name)]
    targ_vars = [v for v in tf.trainable_variables()
               if ('target' in v.name and 'dense' in v.name)]

    # TEST
    # TODO comment out once verified
    # ex_targ_var = targ_vars[0][:3, :3]
    # ex_ac_var = ac_vars[0][:3, :3]
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # ex_targ_var_0 = sess.run(ex_targ_var)
    # ex_ac_var_0 = sess.run(ex_ac_var)
    # print(ex_targ_var_0)
    # print(ex_ac_var_0)

    for i in range(len(targ_vars)):
        targ_vars[i] = targ_vars[i].assign(polyak * ac_vars[i] + (1 - polyak) * targ_vars[i])
    
    # TEST
    # TODO comment out once verified
    # ex_targ_var = targ_vars[0][:3, :3]
    # ex_ac_var = ac_vars[0][:3, :3]
    # ex_targ_var_1 = sess.run(ex_targ_var)
    # ex_targ_var_1_np = polyak * ex_ac_var_0 + (1 - polyak) * ex_targ_var_0
    # print(ex_targ_var_1)
    # print(ex_targ_var_1_np)
    # assert np.allclose(ex_targ_var_1, ex_targ_var_1_np)


