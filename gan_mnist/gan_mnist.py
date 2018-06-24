import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import os

if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists('./out'):
    os.mkdir('./out')


def get_logger(filepath,level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filepath)
    handler.setLevel(logging.INFO)

    # create a logging format
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def random_data(row,column):
    return np.random.uniform(-1., 1., size=[row, column])

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

#鉴别网络weights
d_w1 = weight_variable([784,128])
d_b1 = bias_variable([128])

d_w2 = weight_variable([128,1])
d_b2 = bias_variable([1])

param_d = [d_w1, d_w2, d_b1, d_b2]

#生成网络weights
g_w1 = weight_variable([100,128])
g_b1 = bias_variable([128])

g_w2 = weight_variable([128,784])
g_b2 = bias_variable([784])

param_g = [g_w1, g_w2, g_b1, g_b2]

#鉴别网络
def d_network(x):
    d1 = tf.nn.relu(tf.matmul(x,d_w1)+d_b1)
    d_out = tf.matmul(d1,d_w2)+d_b2
    return tf.nn.sigmoid(d_out)

#生成网络
def g_network(x):
    g1 = tf.nn.relu(tf.matmul(x,g_w1)+g_b1)
    g_out = tf.matmul(g1,g_w2)+g_b2
    return tf.nn.sigmoid(g_out)


x = tf.placeholder(tf.float32,shape=[None,784])
z = tf.placeholder(tf.float32,shape=[None,100])

d_out_real = d_network(x)

g_out = g_network(z)
d_out_fake = d_network(g_out)

d_loss = -tf.reduce_mean(tf.log(d_out_real) + tf.log(1. - d_out_fake))
g_loss = -tf.reduce_mean(tf.log(d_out_fake))

d_optimizer = tf.train.AdamOptimizer().minimize(d_loss,var_list=param_d)
g_optimizer = tf.train.AdamOptimizer().minimize(g_loss,var_list=param_g)

batch_size = 256
max_step = 1000000
mnist = input_data.read_data_sets('../mnist', one_hot=True)
logger = get_logger("./log/info.log")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("training")
    i=0
    for step in range(max_step):
        batch_real,_ = mnist.train.next_batch(batch_size)
        _,d_loss_train = sess.run([d_optimizer, d_loss],feed_dict={x:batch_real, z:random_data(batch_size,100)})
        _,g_loss_train = sess.run([g_optimizer, g_loss],feed_dict={z:random_data(batch_size,100)})


        if step % 1000 == 0:
            samples = sess.run(g_out, feed_dict={z: random_data(16, 100)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(4)), bbox_inches='tight')
            i += 1
            plt.close(fig)

            logger.info("step %s: d_loss is %s, gan_loss is %s"%(step,d_loss_train,g_loss_train))
            print("step %s: d_loss is %s, g_loss is %s"%(step,d_loss_train,g_loss_train))
