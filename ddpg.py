"""
ddpg.py

author: bencottier
"""
from core import mlp_actor_critic, placeholders
import gym
import tensorflow as tf


def ddpg(env, discount, batch_size, polyak):

    action_space = env.action_space
    act_dim = len(action_space.shape)
    obs_dim = len(env.observation_space.shape)
    env.reset()
    _, rwd, _, _ = env.step(env.action_space.sample())
    rwd_dim = len(rwd.shape)

    [x_ph, a_ph, x2_ph, r_ph, d_ph] = placeholders(obs_dim, act_dim, obs_dim, rwd_dim, ())

    with tf.variable_scope('current'):
        pi, q, q_pi = mlp_actor_critic(x_ph, a_ph, action_space)

    with tf.variable_scope('target'):  # scope helps group the vars for target update
        pi_targ, q_targ, q_pi_targ = mlp_actor_critic(x2_ph, a_ph, action_space)

    # Use "done" variable to cancel future value when at end of episode
    # The stop_gradient means inputs to the operation will not factor into gradients
    backup = tf.stop_gradient(r_ph + d_ph * discount * q_pi_targ)
    q_loss = tf.reduce_mean((backup - q)**2)
    # From memory. Not quite clear how this relates to equation in paper.
    pi_loss = -tf.reduce_mean(q)

    # Update targets
    # TODO Not sure if correct. Even if correct, it's not how I remember the baseline.
    # Also, we only want to update the parameters, not the output values. Seems wrong.
    for v,v_targ in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current'),
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')):             
        tf.assign(v_targ, polyak*v + (1-polyak)*v_targ)

    
