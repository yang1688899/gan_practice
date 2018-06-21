[//]: # (Image References)

[image1]: ./rm_img/graph.jpeg
[image2]: ./rm_img/train_d.png
[image3]: ./rm_img/train_g.png

### GANs简介:
GANs全称 Generative Adversarial Nets(生成对抗网络)，于2014年由 Ian Goodfellow 提出。生成对抗网络分成两个部分：生成器(generator)和鉴别器(discriminator)。直观的说生成器(generator)和鉴别器(discriminator)的关系可以理解为造假的和打假的关系，生成器负责生成'假货'，而鉴别器负责'打假'，分辨哪些是'真货'(真实数据)哪些是生成器生成的'假货'。在训练过程中，生成器尽量生成与'真货'(真实数据)一样的'假货'，用来骗过鉴别器，鉴别器则尽可能分辨哪些是'真货'(真实数据)哪些是生成器生成的'假货'，生成器与鉴别器在这样的一轮轮类似于二人博弈的对抗中成长，最终两者达到一种动态均衡，生成器生成的'假货'以假乱真，与真货再无区别，鉴别器无法分辨。

ps:通常生成器(generator)和鉴别器(discriminator)都是神经网络所以也可以称为生成网络(Generator Neural Network)和鉴别网络(Discriminator Neural Network)

详细请看[原论文](https://arxiv.org/abs/1406.2661)

### GANs 是怎么训练的：
GANs的整个网络结构是这样的：

![alt text][image1]

上图中的生成网络其实就是一个简单的神经网络，输入一串随机数，输出一张图片。而鉴别网络则是一个二元分类神经网络，输入一张图片，输出图片是来自真实数据集的概率。

训练过程(一个batch):
* 随机生成N组随机数，输入N组数到生成网络，得到的N张图片，记为数据集fake
* 随机从真实数据集中抽取N张图片，记为数据集real
* 把数据集fake全标注为0，数据集real全标注为1，生成网络参数固定，训练鉴别网络，如下图：

![alt text][image2]


* 只用数据集fake，全标注为1，鉴别网络参数固定，训练生成网络，如下图：

![alt text][image3]



### 项目描述：
利用mnist手写字母数据集，训练生成对抗网络(GANs)，实现手写字母生成

### 实现步骤：
* 
* 

#### 生成器，鉴别器
生成器，鉴别器可以根据需要使用各种网络结构，如cnn，rnn等。这里因为数据相对简单，生成器，鉴别器都使用简单的两层神经网络即可。
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
