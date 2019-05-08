"""
ddpg.py

author: bencottier
"""
import core
import gym
import tensorflow as tf
import numpy as np


def ddpg(env, discount, batch_size, polyak, num_episode, max_step,
        seed=0, actor_critic=core.mlp_actor_critic, ac_kwargs=dict()):
    # Set random seed
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    # Initialise variables
    action_space = env.action_space
    act_dim = action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    x_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(tf.float32, shape=(None, act_dim))
    x2_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    r_ph = tf.placeholder(tf.float32, shape=(None,))
    d_ph = tf.placeholder(tf.float32, shape=(None,))

    # Initalise random process for action exploration
    process = core.OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2)

    # Build main computation graph
    with tf.variable_scope('actor-critic'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, action_space, **ac_kwargs)

    with tf.variable_scope('target'):  # scope helps group the vars for target update
        _, _, q_pi_targ = actor_critic(x2_ph, a_ph, action_space, **ac_kwargs)

    # Use "done" variable to cancel future value when at end of episode
    # The stop_gradient means inputs to the operation will not factor into gradients
    backup = tf.stop_gradient(r_ph + d_ph * discount * q_pi_targ)
    q_loss = tf.reduce_mean((backup - q)**2)
    pi_loss = -tf.reduce_mean(q_pi)

    # Optimisers
    opt_actor = tf.train.AdamOptimizer(learning_rate=1e-4, name='opt_actor')
    opt_critic = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-2, learning_rate=1e-3,
                                               name='opt_critic')
    # Update critic by minimizing the loss
    critic_minimize = opt_critic.minimize(q_loss)
    # Update the actor policy using the sampled policy gradient
    actor_minimize = opt_actor.minimize(pi_loss)

    # Target variable update
    # TODO Not sure if correct. Even if correct, it's not how I remember the baseline.
    ac_vars = [v for v in tf.trainable_variables() if 'actor-critic' in v.name]
    targ_vars = [v for v in tf.trainable_variables() if 'target' in v.name]
    for i in range(len(targ_vars)):
        targ_vars[i] = targ_vars[i].assign(polyak * ac_vars[i] + (1 - polyak) * targ_vars[i])

    # Replay buffer
    max_buffer_size = 1000000
    buffer = []

    # Start up a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def select_action():
        return pi + tf.convert_to_tensor(process.sample())

    def step():
        """
        Execute a time step in training.
        """
        # Receive initial observation state
        o = env.reset()
        for t in range(max_step):
            # Select action according to the current policy and exploration noise
            a = select_action()
            # Execute action and observe reward and new state
            o2, r, done, _ = env.step(a)
            transition_t = (o, a, r, o2, done)
            # Store transition in buffer
            if len(buffer) < max_buffer_size:
                buffer.append(transition_t)
            else:
                buffer[t % max_buffer_size] = transition_t
            # Sample a random minibatch of transitions from buffer
            transitions = [random.choice(buffer) for _ in range(batch_size)]
            [x_batch, a_batch, r_batch, x2_batch, d_batch] = \
                    [np.array([tr[i] for tr in transitions]) for i in range(5)]
            # Run training ops
            feed_dict = {x_ph: x_batch, a_ph: a_batch, x2_ph: x2_batch, r_ph: r_batch, d_ph: d_batch}
            q_loss, pi_loss = sess.run([critic_minimize, actor_minimize], feed_dict=feed_dict)
            # Target networks update automatically through the graph
            # Advance the stored state
            o = o2
            if done:
                break
        # TODO return stats?

    # Training loop
    for episode in range(num_episode):
        # Initalise random process for action exploration
        process.reset()
        step()
        # TODO Reporting

    env.close()


if __name__ == '__main__':
    # Buffer experimentation

    # max_buffer_size = 1000
    # buffer = np.zeros(shape=(max_buffer_size, 4))
    # batch_size = 4
    # act_dim = 6
    # obs_dim = 17
    # o = np.random.normal(size=obs_dim)
    # for t in range(20):
    #     # Select action according to the current policy and exploration noise
    #     a = np.random.normal(size=act_dim)
    #     # Execute action and observe reward and new state
    #     r = np.random.rand()
    #     o2 = np.random.normal(size=obs_dim)
    #     # Store transition in buffer
    #     buffer[t % max_buffer_size, :] = np.array([o, a, r, o2])
    #     # Sample a random minibatch of transitions from buffer
    #     rand_idx = np.random.choice(np.arange(len(buffer[:np.minimum(t, max_buffer_size)])), size=batch_size)
    #     transitions = buffer[rand_idx]
    #     o = o2

    import random
    max_buffer_size = 1000
    buffer = []
    batch_size = 4
    act_dim = 6
    obs_dim = 17
    o = np.random.normal(size=obs_dim)
    for t in range(20):
        # Select action according to the current policy and exploration noise
        a = np.random.normal(size=act_dim)
        # Execute action and observe reward and new state
        r = np.random.rand()
        o2 = np.random.normal(size=obs_dim)
        # Store transition in buffer
        if len(buffer) < max_buffer_size:
            buffer.append((o, a, r, o2))
        else:
            buffer[t % max_buffer_size] = (o, a, r, o2)
        # Sample a random minibatch of transitions from buffer
        transitions = []
        for _ in range(batch_size):
            transitions.append(random.choice(buffer))
        x_batch = np.array([transition[0] for transition in transitions])
        a_batch = np.array([transition[1] for transition in transitions])
        r_batch = np.array([transition[2] for transition in transitions])
        x2_batch = np.array([transition[3] for transition in transitions])

        o = o2
