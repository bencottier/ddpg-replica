"""
ddpg.py

author: bencottier
"""
import core
import gym
from spinup.utils.logx import Logger, EpochLogger
import tensorflow as tf
import numpy as np
import random
import time


def ddpg(exp_name, env_name, discount=0.99, batch_size=64, polyak=0.001, 
        epochs=200, steps_per_epoch=5000, seed=0, 
        actor_critic=core.mlp_actor_critic, noise_scale=1.0, 
        exploration_steps=None, ac_kwargs=dict(), logdir=None):
    # Do not reuse old graph (in case of persistence over multiple calls)
    tf.reset_default_graph()

    # Create environment
    env = gym.make(env_name)

    # Create loggers
    if logdir is not None:
        time_string = time.strftime('%Y-%m-%d-%H-%M-%S')
        logdir = f'{logdir}/{time_string}_ddpg_{env_name.lower()}_s{seed}'
    epoch_logger = EpochLogger(output_dir=logdir)
    epoch_logger.save_config(locals())

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    env.seed(seed)

    # Initialise variables
    action_space = env.action_space
    act_dim = action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    x_ph = tf.placeholder(tf.float32, shape=(None, obs_dim), name='obs')
    a_ph = tf.placeholder(tf.float32, shape=(None, act_dim), name='act')
    x2_ph = tf.placeholder(tf.float32, shape=(None, obs_dim), name='obs2')
    r_ph = tf.placeholder(tf.float32, shape=(None,), name='rwd')
    d_ph = tf.placeholder(tf.float32, shape=(None,), name='done')

    # Initalise random process for action exploration
    process = core.OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2, shape=(act_dim,))
    if exploration_steps is None:
        exploration_steps = 0.2 * steps_per_epoch

    # Build main computation graph
    with tf.variable_scope('actor-critic'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, action_space, **ac_kwargs)

    with tf.variable_scope('target'):  # scope helps group the vars for target update
        _, _, q_pi_targ = actor_critic(x2_ph, a_ph, action_space, **ac_kwargs)

    # Target variable initialisation
    ac_vars = [v for v in tf.trainable_variables() if 'actor-critic' in v.name]
    targ_vars = [v for v in tf.trainable_variables() if 'target' in v.name]
    targ_init = [targ_vars[i].assign(ac_vars[i]) for i in range(len(targ_vars))]

    # Use "done" variable to cancel future value when at end of episode
    # The stop_gradient means inputs to the operation will not factor into gradients
    backup = tf.stop_gradient(r_ph + (1 - d_ph) * discount * q_pi_targ, name='backup')
    q_loss = tf.reduce_mean((backup - q)**2, name='q_loss')
    pi_loss = -tf.reduce_mean(q_pi, name='pi_loss')

    # Target variable update
    targ_update = [targ_vars[i].assign(polyak * ac_vars[i] + (1 - polyak) * targ_vars[i]) \
            for i in range(len(targ_vars))]

    # Optimisers
    opt_critic = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-2, learning_rate=1e-3,
                                               name='opt_critic')
    opt_actor = tf.train.AdamOptimizer(learning_rate=1e-4, name='opt_actor')
    # Update critic by minimizing the loss
    critic_minimize = opt_critic.minimize(q_loss, name='critic_minimize')
    # Update the actor policy using the sampled policy gradient
    actor_minimize = opt_actor.minimize(pi_loss, name='actor_minimize')

    # Replay buffer
    max_buffer_size = 1000000
    buffer = []

    # Create a file writer for logging
    writer = tf.summary.FileWriter(logdir)
    tf.summary.scalar('q_loss', q_loss)
    merged_summary = tf.summary.merge_all()

    # Start up a session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    sess.run(tf.global_variables_initializer())
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    # Save variables
    input_dict = {'x': x_ph, 'a': a_ph, 'x2': x2_ph, 'r': r_ph, 'd': d_ph}
    output_dict = {'critic_minimize': critic_minimize, 'actor_minimize': actor_minimize, 
            'q_loss': q_loss, 'pi_loss': pi_loss}
    epoch_logger.setup_tf_saver(sess, input_dict, output_dict)

    def select_action(a_pi):
        return a_pi + np.random.normal(loc=0.0, scale=noise_scale, size=(act_dim,)) # process.sample()

    def train_epoch(total_steps):
        """
        Execute an epoch of training.
        """
        t = 0  # episode time step
        ret = 0  # episode return
        process.reset()  # initalise random process for action exploration
        o = env.reset()  # receive initial observation state
        for step in range(steps_per_epoch):
            if step < exploration_steps:
                # Initial exploration
                a = env.action_space.sample()
            else:
                # Select action according to the current policy and exploration noise
                a_pi = np.squeeze(sess.run(pi, feed_dict={x_ph: o.reshape([1, -1])}), axis=0)
                a = select_action(a_pi)
            # Execute action and observe reward and new state
            o2, r, done, _ = env.step(a)
            t += 1
            total_steps += 1
            ret += r
            transition_t = (o, a, r, o2, done)
            # Advance the stored state
            o = o2
            # Check for end of episode
            if done:
                epoch_logger.store(EpRet=ret)
                epoch_logger.store(EpLen=t)
                t = 0
                ret = 0
                process.reset()
                o = env.reset()
            # Store transition in buffer
            if len(buffer) < max_buffer_size:
                buffer.append(transition_t)
            else:
                buffer[total_steps % max_buffer_size] = transition_t
            # Sample a random minibatch of transitions from buffer
            transitions = [random.choice(buffer) for _ in range(batch_size)]
            [x_batch, a_batch, r_batch, x2_batch, d_batch] = \
                    [np.array([tr[i] for tr in transitions]) for i in range(5)]
            # Run training ops
            feed_dict = {x_ph: x_batch, a_ph: a_batch, x2_ph: x2_batch, r_ph: r_batch, d_ph: d_batch}
            if step < exploration_steps:
                ops = [critic_minimize, q_loss]
                _, q_loss_eval = sess.run(ops, feed_dict=feed_dict)
                pi_loss_eval = 0
            else:
                ops = [critic_minimize, actor_minimize, q_loss, pi_loss]
                _, _, q_loss_eval, pi_loss_eval = sess.run(ops, feed_dict=feed_dict)
            # Update the target networks
            sess.run(targ_update)
            # Log stats
            epoch_logger.store(LossQ=q_loss_eval)
            epoch_logger.store(LossPi=pi_loss_eval)
            # Generate summary and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(summ, total_steps)
        return total_steps
                
    def test():
        # Run a few episodes for statistical power
        for _ in range(1):
            t = 0
            ret = 0
            done = False
            o = env.reset()
            while not done:
                # Select action according to the current policy
                a = np.squeeze(sess.run(pi, feed_dict={x_ph: o.reshape([1, -1])}), axis=0)
                # Execute action and observe reward and new state
                o2, r, done, _ = env.step(a)
                t += 1
                ret += r
                # Advance the stored state
                o = o2
            # Log stats
            epoch_logger.store(TestEpRet=ret)
            epoch_logger.store(TestEpLen=t)

    # Initialise target networks
    sess.run(targ_init)

    # Training loop
    t0 = time.time()
    total_steps = 0
    for epoch in range(epochs):
        total_steps = train_epoch(total_steps)
        test()
        # Reporting
        epoch_logger.store(Epoch=epoch+1)
        epoch_logger.store(TotalEnvInteracts=total_steps)
        epoch_logger.store(Time=time.time() - t0)
        epoch_logger.log_tabular('Epoch', average_only=True)
        epoch_logger.log_tabular('EpRet', with_min_and_max=True)
        epoch_logger.log_tabular('TestEpRet', with_min_and_max=True)
        epoch_logger.log_tabular('EpLen', average_only=True)
        epoch_logger.log_tabular('TestEpLen', average_only=True)
        epoch_logger.log_tabular('TotalEnvInteracts', average_only=True)
        epoch_logger.log_tabular('LossQ', average_only=True)
        epoch_logger.log_tabular('LossPi', average_only=True)
        epoch_logger.log_tabular('Time', average_only=True)
        epoch_logger.dump_tabular()
        # Save state of training variables (use itr=ep to not overwrite)
        # TODO: investigate cause of `Warning: could not pickle state_dict.`
        # epoch_logger.save_state({'trainable_variables': tf.trainable_variables()})

    env.close()
    sess.close()


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
