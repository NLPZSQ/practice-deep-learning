import tensorflow as tf
import numpy as np
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

x = tf.placeholder(tf.float32,shape=(None,n_inputs))
y = tf.placeholder(tf.float32,shape = (None))

def neuron_layer(x,n_neurons,name,activation=None):
    n_inputs = int(x.get_shape()[1])
    stddev = 2/np.sqrt((n_inputs))
    #shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定
    init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
    w = tf.Variable(init,name="weight")
    b = tf.Variable(tf.zeros([n_neurons]),name = "biases")
    z = tf.matmul(x,w) + b
    if activation == 'relu':
        return tf.nn.relu(z)
    else:
        return z

