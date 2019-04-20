"""
helpers.py

author: bencottier
"""
import tensorflow as tf
import numpy as np


def mlp(x, hidden_sizes, activation, output_activation=None):
    out = x
    for hidden_size in hidden_sizes[:-1]:
        out = tf.layers.dense(out, hidden_size, activation=activation)
    out = tf.layers.dense(out, hidden_sizes[-1], activation=output_activation)
    return out


def mlp_actor_critic(x, a, action_space, hidden_sizes=[300, 400], activation=tf.nn.relu):
    act_dim = a.shape[1]
    act_limit = action_space.high[0]
    hidden_sizes = list(hidden_sizes)
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, hidden_sizes+[act_dim], activation, tf.nn.tanh)
    with tf.variable_scope('q'):
        # Squeeze needed to avoid unintended TF broadcasting down the line
        q = tf.squeeze(mlp(tf.concat([x, a], axis=1), hidden_sizes+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=1), hidden_sizes+[1], activation, None), axis=1)
    return pi, q, q_pi

