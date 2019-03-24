#batch norm is not suitable for RNN
import tensorflow as tf


def weight_variable(shape,reuse ,name='weights'):
    if reuse==False:
        initializer = tf.random_normal_initializer(mean=0., stddev=0.1, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    else:
        return tf.get_variable(name=name)

def bias_variable(shape,reuse, name='biases'):
    if reuse == False:
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    else:
        return tf.get_variable(name=name)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class batch_norm2(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):

    return tf.layers.batch_normalization(x,
                      #momentum=self.momentum,
                      #gamma_initializer=tf.constant_initializer(0.1),
                      epsilon=self.epsilon,
                      scale=True,
                      training=train,
                       name=self.name
                      )