import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from tensorflow.examples.tutorials.mnist import input_data
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

g_w1 = weight_variable([110,128])
g_b1 = bias_variable([128])
g_w2 = weight_variable([128,784])
g_b2 = bias_variable([784])

g_param = [g_w1,g_w2,g_b1,g_b2]

d_w1 = weight_variable([794,128])
d_b1 = bias_variable([128])
d_w2 = weight_variable([128,1])
d_b2 = bias_variable([1])

d_param = [d_w1,d_w2,d_b1,d_b2]

def d_network(x,y):
    #加入condition
    condition_x = tf.concat(values=[x,y],axis=1)

    d_1 = tf.nn.relu(tf.matmul(condition_x,d_w1)+d_b2)
    d_out = tf.matmul(d_1,d_w2)+d_b2

    return d_out

def g_network(x,y):
    #加入condition
    condition_x = tf.concat(values=[x,y],axis=1)

    g_1 = tf.nn.relu(tf.matmul(condition_x,g_w1)+g_b1)
    g_out = tf.matmul(g_1,g_w2)+g_b2

    return g_out

num_class = 10
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,num_class])
z = tf.placeholder(tf.float32,[None,100])

d_out = d_network(x,y)

g_out = g_network(z,y)
gan_out = d_network(g_out,y)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out,labels=tf.ones_like(d_out)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_out, labels=tf.zeros_like(gan_out)))
d_loss = d_loss_real + d_loss_fake

gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_out, labels=tf.ones_like(gan_out)))

d_train = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_param)
gan_train = tf.train.AdamOptimizer().minimize(gan_loss, var_list=g_param)

batch_size = 128
max_step = 100000
mnist = input_data.read_data_sets('../mnist', one_hot=True)
logger = get_logger("./log/info.log")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("training......")
    i = 0
    for step in range(max_step):
        batch_real,labels_real = mnist.train.next_batch(batch_size)
        batch_fake = random_data(batch_size,100)

        _,d_loss_train = sess.run([d_train,d_loss],feed_dict={x:batch_real, z:batch_fake, y:labels_real})
        _,gan_loss_train = sess.run([gan_train,gan_loss],feed_dict={z:batch_fake,y:labels_real})

        if step%1000 == 0 or step == max_step-1:

            logger.info("step %s: d_loss is %s, gan_loss is %s" % (step, d_loss_train, gan_loss_train))
            print("step %s: d_loss is %s, gan_loss is %s" % (step, d_loss_train, gan_loss_train))

    labels_test = np.zeros([16,num_class],dtype=np.float32)
    labels_test[:,5] = 1
    samples = sess.run(g_out, feed_dict={z: random_data(16, 100),y:labels_test})

    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    i += 1
    plt.close(fig)



