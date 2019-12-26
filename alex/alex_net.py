#coding=utf-8
#Version:python3.5.2



import tensorflow as tf
import numpy as np
import alex.config as cfg

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.keep_prob = tf.constant(cfg.KEEP_PROB if is_training else 1,dtype=tf.float32)
        self.alpha = cfg.ALPHA
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        if is_training:
            self.logits = self.create()
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1)), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        else:
            self.logits = tf.nn.softmax(self.create())



    # Call the create function to build the computational graph of AlexNet

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 5 * 5 * 256])
        fc6 = fc(flattened, 5 * 5 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, self.num_class, relu=False, name='fc8')
        return fc8
    def loss_layer(self,predicts, labels, scope='loss_layer'):
        #labels one hot 编码
        with tf.variable_scope(scope):
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicts,labels=labels))
            tf.summary.scalar('cross_entropy', ce_loss)
            #TODO add some l2 loss
            #TODO label smooth
            tf.losses.add_loss(ce_loss)
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:#隔开不同层的权重
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])#为什么不设置初始化方法
#tf变量维度的创建与np.random的不同
    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
    #返回列表为组合连接做准备，可不可以把分组的组数设置为随机数
        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    # relu = tf.nn.relu(bias, name=scope.name)
    relu = tf.nn.leaky_relu(bias, alpha=cfg.ALPHA, name='leaky_relu')

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        # relu = tf.nn.relu(act)
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op

