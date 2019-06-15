"""
helpers.py

author: bencottier
"""
import tensorflow as tf
import numpy as np


class StochasticProcess:
    def __init__(self, loc=0., shape=None, x0=None, dt=.001):
        self.loc = loc
        self.shape = shape
        self.x0 = np.zeros(self.shape, dtype=np.float32) if x0 is None else x0
        self.dt = dt
        self.reset()

    def next_value(self):
        return 0.0

    def sample(self, update=True):
        x = self.next_value()
        if update:
            self.x = x
        return x

    def reset(self):
        self.x = self.x0


class StochasticProcessWithScale(StochasticProcess):
    def __init__(self, scale=1.0, *args, **kwargs):
        super(StochasticProcessWithScale, self).__init__(*args, **kwargs)
        self.scale = scale


class NormalProcess(StochasticProcessWithScale):
    """
    Samples from a normal distribution independently at each step.
    """
    def next_value(self):
        return np.random.normal(self.loc, self.scale, self.shape)


class OrnsteinUhlenbeckProcess(StochasticProcessWithScale):
    def __init__(self, theta=0.15, sigma=0.2, *args, **kwargs):
        super(OrnsteinUhlenbeckProcess, self).__init__(scale=sigma, *args, **kwargs)
        self.theta = theta
        self.sqrtdt = np.sqrt(self.dt)

    def next_value(self):
        return self.x + self.theta * (self.loc - self.x) * self.dt + \
                self.scale * self.sqrtdt * np.random.normal(size=self.shape)


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
        q_net = mlp(tf.concat([x, a], axis=1), hidden_sizes+[1], activation, None)
        q = tf.squeeze(q_net, axis=1, name='q')
    with tf.variable_scope('q', reuse=True):
        q_pi_net = mlp(tf.concat([x, pi], axis=1), hidden_sizes+[1], activation, None)
        q_pi = tf.squeeze(q_pi_net, axis=1, name='q_pi')
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
    # process = NormalProcess(scale=0.1, x0=x0)
    n = 100
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i + 1] = process.sample()

    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.show()
