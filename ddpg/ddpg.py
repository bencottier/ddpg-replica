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


def ddpg(env_name, exp_name=None, exp_variant=None, seed=0, epochs=200, steps_per_epoch=5000,
        batch_size=64, discount=0.99, polyak=0.001, weight_decay=1e-2,
        exploration_steps=None, rand_proc=core.OrnsteinUhlenbeckProcess, rand_proc_kwargs=dict(),
        actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), logdir=None):
    # Do not reuse old graph (in case of persistence over multiple calls)
    tf.reset_default_graph()

    # Create environment
    env = gym.make(env_name)
    # env._max_episode_steps = 200  # limit episode length

    # Create loggers
    if exp_name is None:
        exp_name = 'uncategorised'
    if exp_variant is None:
        exp_variant = 'run'
    if logdir is not None:
        time_string = time.strftime('%Y-%m-%d-%H-%M-%S')
        logdir = f'{logdir}/{exp_name}/{time_string}_ddpg_{env_name.lower()}_s{seed}'
    epoch_logger = EpochLogger(output_dir=logdir, exp_name=f'{exp_variant}')
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
    process = rand_proc(shape=(act_dim,), **rand_proc_kwargs)
    if exploration_steps is None:
        exploration_steps = 0
    elif exploration_steps < 0:
        exploration_steps = 0.2 * steps_per_epoch

    # Build main computation graph
    with tf.variable_scope('actor-critic'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, action_space, **ac_kwargs)

    with tf.variable_scope('target'):  # scope helps group the vars for target update
        pi_targ, q_targ, q_pi_targ = actor_critic(x2_ph, a_ph, action_space, **ac_kwargs)

    # Target variable initialisation
    ac_vars = [v for v in tf.trainable_variables() if 'actor-critic' in v.name]
    q_vars = [v for v in ac_vars if 'q' in v.name]
    pi_vars = [v for v in ac_vars if 'pi' in v.name]
    targ_vars = [v for v in tf.trainable_variables() if 'target' in v.name]
    targ_init = [targ_vars[i].assign(ac_vars[i]) for i in range(len(targ_vars))]

    # Use "done" variable to cancel future value when at end of episode
    # The stop_gradient means inputs to the operation will not factor into gradients
    backup = tf.stop_gradient(r_ph + discount * q_pi_targ, name='backup')
    regulariser = tf.reduce_sum([tf.nn.l2_loss(v) for v in q_vars if 'kernel' in v.name])
    q_loss = tf.reduce_mean((backup - q)**2, name='q_loss') + weight_decay * regulariser
    pi_loss = -tf.reduce_mean(q_pi, name='pi_loss')

    # Target variable update
    targ_update = [targ_vars[i].assign(polyak * ac_vars[i] + (1 - polyak) * targ_vars[i]) \
            for i in range(len(targ_vars))]

    # Optimisers
    opt_critic = tf.train.AdamOptimizer(learning_rate=1e-3, name='opt_critic')
    opt_actor = tf.train.AdamOptimizer(learning_rate=1e-4, name='opt_actor')
    # Update critic by minimizing the loss
    critic_minimize = opt_critic.minimize(q_loss, var_list=q_vars, name='critic_minimize')
    # Update the actor policy using the sampled policy gradient
    actor_minimize = opt_actor.minimize(pi_loss, var_list=pi_vars, name='actor_minimize')

    # Replay buffer
    max_buffer_size = 1000000
    buffer = []

    # Create a file writer for logging
    writer = tf.summary.FileWriter(logdir)
    tf.summary.scalar('critic/loss', q_loss)
    tf.summary.scalar('critic/value0', q[0])
    tf.summary.scalar('critic/target0', q_targ[0])
    merged_summary = tf.summary.merge_all()

    # Start up a session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)  # slower but ensures reproducibility
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
        a = a_pi + process.sample()
        # Note: AFAIK clipping was not used in original DDPG
        return np.clip(a, action_space.low, action_space.high)

    def train_epoch(total_steps):
        """
        Execute an epoch of training.
        """
        t = 0  # episode time step
        ret = 0  # episode return
        process.reset()  # initalise random process for action exploration
        o = env.reset()  # receive initial observation state
        for step in range(steps_per_epoch):
            # env.render()
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
                ops = [critic_minimize, q_loss, pi_loss]
                _, q_loss_eval, pi_loss_eval = sess.run(ops, feed_dict=feed_dict)
            else:
                ops = [critic_minimize, actor_minimize, q_loss, pi_loss]
                _, _, q_loss_eval, pi_loss_eval = sess.run(ops, feed_dict=feed_dict)
            q_eval, q_targ_eval = sess.run([q, q_targ], feed_dict=feed_dict)
            # Update the target networks
            sess.run(targ_update)
            # Log stats
            epoch_logger.store(QVals=q_eval)
            epoch_logger.store(LossQ=q_loss_eval)
            epoch_logger.store(LossPi=pi_loss_eval)
            # Generate summary and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(summ, total_steps)
        return total_steps

    def test():
        # Run a few episodes for statistical power
        for _ in range(5):
            t = 0
            ret = 0
            done = False
            o = env.reset()
            while not done:
                # env.render()
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
        epoch_logger.log_tabular('QVals', with_min_and_max=True)
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
    pass
