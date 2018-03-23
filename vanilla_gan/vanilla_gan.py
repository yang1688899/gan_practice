import tensorflow as tf

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32,shape=[None,784])

d_w1 = weight_variable([784,128])
d_b1 = bias_variable([128])

d_w2 = weight_variable([128,1])
d_b2 = bias_variable([1])

param_d = [d_w1, d_w2, d_b1, d_b2]

g_w1 = weight_variable([100,128])
g_b1 = bias_variable([128])

g_w2 = weight_variable([128,784])
g_b2 = bias_variable([784])

param_g = [g_w1, g_w2, g_b1, g_b2]

