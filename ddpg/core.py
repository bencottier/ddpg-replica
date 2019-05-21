"""
helpers.py

author: bencottier
"""
import tensorflow as tf
import numpy as np


class OrnsteinUhlenbeckProcess(object):
    
    def __init__(self, theta, sigma, mu=0., shape=None, x0=None, dt=.001):
        self.shape = shape
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.sqrtdt = np.sqrt(self.dt)
        self.x0 = np.zeros(self.shape, dtype=np.float32) if x0 is None else x0
        self.reset()

    def sample(self, update=True):
        x = self.x + self.theta * (self.mu - self.x) * self.dt + \
                self.sigma * self.sqrtdt * np.random.normal(size=self.shape)
        if update:
            self.x = x
        return x

    def reset(self):
        self.x = self.x0


def placeholders(*shapes):
    phs = []
    for shape in shapes:
        phs.append(tf.placeholder(tf.float32, shape=shape))
    return phs


def mlp(x, hidden_sizes, activation, output_activation=None):
    """
    Create a multi-layer perceptron network.

    Arguments:
        x: Tensor. Input to the network.
        hidden_sizes: iterable of int. Number of units in each hidden layer.
        activation: callable. Activation function applied at each hidden layer.
        output_activation: callable. Activation function applied at the output layer.

    Returns:
        Output tensor of the network.
    """
    out = x
    for hidden_size in hidden_sizes[:-1]:
        out = tf.layers.dense(out, hidden_size, activation=activation)
    out = tf.layers.dense(out, hidden_sizes[-1], activation=output_activation)
    return out


def mlp_actor_critic(x, a, action_space, hidden_sizes=(400, 300), activation=tf.nn.relu):
    """
    Create MLPs for an actor-critic RL algorithm.

    Arguments:
        x: Tensor. Observation or state.
        a: Tensor. Action.
        action_space: gym.Space. Contains information about the action space.
        hidden_sizes:  iterable of int. Number of units in each hidden layer of the MLPs.
        activation: callable. Activation function applied at each hidden layer of the MLPs.

    Returns:
        Output tensors for
            pi: actor or policy network.
            q: critic or action-value network, taking (x, a) as input.
            q_pi: critic or action-value network, taking (x, pi) as input.
    """
    act_dim = action_space.shape[0]
    act_limit = action_space.high
    hidden_sizes = list(hidden_sizes)
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, hidden_sizes+[act_dim], activation, tf.nn.tanh)
    with tf.variable_scope('q'):
        # Squeeze needed to avoid unintended TF broadcasting down the line
        q = tf.squeeze(mlp(tf.concat([x, a], axis=1), hidden_sizes+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=1), hidden_sizes+[1], activation, None), axis=1)
    return pi, q, q_pi


if __name__ == '__main__':

    # class ActionSpace():
    #     high = [10.0]

    # action_space = ActionSpace()
    # obs_dim = 2
    # act_dim = 3
    # batch_size = 4
    # x_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    # a_ph = tf.placeholder(tf.float32, shape=(None, act_dim))
    # pi, q, q_pi = mlp_actor_critic(x_ph, a_ph, action_space)
    
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # feed_dict = {x_ph: np.random.normal(size=(batch_size, obs_dim)), 
    #              a_ph: np.random.normal(size=(batch_size, act_dim))}
    # pi, q, q_pi = sess.run([pi, q, q_pi], feed_dict=feed_dict)
    # print("Pi(s):", pi)
    # print("Q(s, a):", q)
    # print("Q(s, pi):", q_pi)

    x0 = 0
    process = OrnsteinUhlenbeckProcess(0.15, 0.2, x0=x0)
    n = 100
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i + 1] = process.sample()

    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.show()
