import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def sample_data(size, length=100):
    """
    生成值得均值和方差的数据
    :param size:
    :param length:
    :return:
    """
    data = []
    for _ in range(size):
        data.append(sorted(np.random.normal(4, 1.5, length)))
    return np.array(data)


def random_data(size, length=100):
    """
    随机生成数据
    :param size:
    :param length:
    :return:
    """
    data = []
    for _ in range(size):
        x = np.random.random(length)
        data.append(x)
    return np.array(data)


def preprocess_data(x):
    """
    计算每一组数据平均值和方差
    :param x:
    :return:
    """
    return [[np.mean(data), np.std(data)] for data in x]

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

def g_network(x):
    g1_w = weight_variable([1000,32])
    g1_b = bias_variable([32])
    g1 = tf.nn.relu(tf.matmul(x,g1_w)+g1_b)

    g2_w = weight_variable([32,32])
    g2_b = bias_variable([32])
    g2 = tf.nn.relu(tf.matmul(g1,g2_w)+g2_b)

    g3_w = weight_variable([32,1000])
    g3_b = bias_variable([1000])
    g_out = tf.matmul(g2,g3_w)+g3_b

    param_g = [g1_w,g2_w,g3_w,g1_b,g2_b,g3_b]

    return g_out

def d_network(x):
    d1_w = weight_variable([1000,32])
    d1_b = bias_variable([32])
    d1 = tf.nn.relu(tf.matmul(x,d1_w)+d1_b)

    d2_w = weight_variable([32,32])
    d2_b = bias_variable([32])
    d2 = tf.nn.relu(tf.matmul(d1,d2_w)+d2_b)

    d3_w = weight_variable([32,1])
    d3_b = bias_variable([1])
    d_out = tf.nn.softmax(tf.matmul(d2,d3_w)+d3_b)

    param_d = [d1_w,d2_w,d3_w,d1_b,d2_b,d3_b]

    return d_out,param_d

x = tf.placeholder(tf.float32,[None,1000])
y = tf.placeholder(tf.float32,[None])

#
g_out,param_g = g_network(x)
#
d_out,param_b = d_network(x)
#
gan_out = d_network(g_out)

d_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_out,y)



    


