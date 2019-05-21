import random
import numpy as np
import tensorflow as tf
import gym


def run(n, env):
    print([random.random() for i in range(n)])

    print(np.random.normal(size=n))

    sess = tf.Session()
    x = tf.random.normal(shape=(n,))
    print(sess.run(x))
    with tf.variable_scope('foo'):
        x2 = tf.placeholder(dtype=tf.float32, shape=(None, n))
        x3 = tf.layers.dense(x2, 3)
    sess.run(tf.global_variables_initializer())
    print(sess.run(x3, feed_dict={x2: np.random.normal(size=(2, n))}))
    
    print(env.reset())
