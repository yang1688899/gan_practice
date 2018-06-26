[//]: # (Image References)

[image1]: ./rm_img/graph.jpeg
[image2]: ./rm_img/train_d.png
[image3]: ./rm_img/train_g.png
[image4]: ./rm_img/fake.png
[image5]: ./rm_img/orgin.png

### *GANs简介*
GANs全称 Generative Adversarial Nets(生成对抗网络)，于2014年由 Ian Goodfellow 提出。生成对抗网络分成两个部分：生成网络(Generator Neural Network)和鉴别网络(Discriminator Neural Network)。直观的说生成网络(Generator Neural Network)和鉴别网络(Discriminator Neural Network)的关系可以理解为造假的和打假的关系，生成网络负责生成'假货'，而鉴别网络负责'打假'，分辨哪些是'真货'(真实数据)哪些是生成网络生成的'假货'。在训练过程中，生成网络尽量生成与'真货'(真实数据)一样的'假货'，用来骗过鉴别网络，鉴别网络则尽可能分辨哪些是'真货'(真实数据)哪些是生成网络生成的'假货'，生成网络与鉴别网络在这样的一轮轮类似于二人博弈的对抗中成长，最终两者达到一种动态均衡，生成网络生成的'假货'以假乱真，与真货再无区别，鉴别网络无法分辨。

详细请看[原论文](https://arxiv.org/abs/1406.2661)

### *GANs 是怎么训练的*
GANs的整个网络结构是这样的：

![alt text][image1]

(ps:图片网上找的，侵删)

上图中的生成网络其实就是一个简单的神经网络，输入一串随机数，输出一张图片。而鉴别网络则是一个二元分类神经网络，输入一张图片，输出图片来自真实数据集的概率。

训练过程:
* 随机生成N组随机数，输入N组数到生成网络，得到的N张图片，记为数据fake
* 随机从真实数据集中抽取N张图片，记为数据real
* 把数据fake全标注为0，数据real全标注为1，生成网络参数固定，训练鉴别网络，如下图：

![alt text][image2]


* 只用数据fake，全标注为1，鉴别网络参数固定，训练生成网络，如下图：

![alt text][image3]

(以上为一个GANs一个iteration的过程)

### *项目描述：*
利用mnist数据集，训练一个简单的无条件GANs，实现手写数字生成

### *实现步骤：*
* 设计生成网络，鉴别网络
* 设计损失函数，优化器
* 训练模型

#### *设计生成网络，鉴别网络*
生成网络，鉴别网络可以根据需要使用各种网络结构，如cnn，rnn等。这里因为数据相对简单，生成网络，鉴别网络都使用简单的三层神经网络即可。
如下：

```
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
```
其中鉴别网络的输入为786个值（对应28x28图片），输出为一个代表是否为真实数据概率的数值。生成网络输入为一百个随机数，输出为786个值（对应28x28图片）。

#### *设计损失函数，优化器*
先看代码：
```
x = tf.placeholder(tf.float32,shape=[None,784])
z = tf.placeholder(tf.float32,shape=[None,100])

d_out_real = d_network(x)

g_out = g_network(z)
d_out_fake = d_network(g_out)

d_loss = -tf.reduce_mean(tf.log(d_out_real) + tf.log(1. - d_out_fake))
g_loss = -tf.reduce_mean(tf.log(d_out_fake))

d_optimizer = tf.train.AdamOptimizer().minimize(d_loss,var_list=param_d)
g_optimizer = tf.train.AdamOptimizer().minimize(g_loss,var_list=param_g)
```

鉴别网络要使真实数据的输出d_out_real尽量为1，生成数据的输出d_out_fake尽量为0，因此需要最小化 -tf.reduce_mean(tf.log(d_out_real) + tf.log(1. - d_out_fake))。生成网络要使鉴别网络对生成数据的输出d_out_fake尽量为1，因此需要最小化tf.reduce_mean(tf.log(d_out_fake))。

优化器都选用Adam，这里要注意的是优化鉴别网络时只更新鉴别网络的参数，优化生成网络时只更新生成网络的参数。

#### *训练模型*
最后是训练模型，batch_size为256，一共进行了50000个itration，每个iteration先训练一次鉴别网络，然后一次生成网络（这里也可以尝试其他的训练策略，比如一个iteration训练一次鉴别网络，然后两次生成网络）代码如下：
```
batch_size = 256
max_step = 50000
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
```

  最后得到的生成模型生成的一些图片：
  
  ![alt text][image4]
  
  这是真实数据集中随机抽取的几张图片：
  
  ![alt text][image5]
 
