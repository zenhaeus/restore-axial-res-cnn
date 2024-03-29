import tensorflow as tf
import numpy as np

"""
U-Net implementation according to https://arxiv.org/pdf/1505.04597.pdf

Generic implementation of U-Net with parametrised number of layers and filters.
Used for binary image segmentation.
"""


def forward(X, num_layers, root_size, dropout_keep=None, dilation_size=3, conv_size=3):
    """Build the U-Net static computation graph

    :param X: input of network, use `input_size_needed` to find which image size it expect
    :param num_layers: number of layers down
    :param root_size: number of filters of the first layer, will be doubled every down move
    :param dropout_keep: tf variable keep_proba to enable dropout or not
    :param dilation_size: dilated convolution filter size
    :param conv_size: convolution filter size
    :return: the network
    """
    net = X - 0.5

    num_filters = root_size
    conv = []

    # Contracting path
    conv_filter_size = (conv_size, conv_size)
    conv_filter_size_2 = (dilation_size, dilation_size)
    for layer_i in range(num_layers):
        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        with tf.variable_scope("conv_{}".format(layer_i)):
            net = tf.layers.conv2d(net, num_filters, conv_filter_size, padding='same', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            if dilation_size > 0:
                net = tf.layers.conv2d(net, num_filters, conv_filter_size_2, dilation_rate=(2, 2), padding='same', name="conv2")
                net = tf.nn.relu(net, name="relu2")

        conv.append(net)

        # Skip last max pooling layer as the last layer is followed by an upsampling
        if layer_i < num_layers - 1:
            net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool")

        num_filters *= 2

    num_filters = int(num_filters / 2)
    net = conv.pop()

    # Expansive path
    for layer_i in range(num_layers - 1):
        num_filters = int(num_filters / 2)

        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        net = tf.layers.conv2d_transpose(net, num_filters, strides=(2, 2), kernel_size=(2, 2),
                                         name="up_conv_{}".format(layer_i))

        traverse = conv.pop()
        with tf.variable_scope("crop_{}".format(layer_i)):
            traverse_crop = tf.image.resize_image_with_crop_or_pad(traverse, int(net.shape[1]), int(net.shape[2]))

        net = tf.concat([traverse_crop, net], axis=3, name="concat")

        with tf.variable_scope("conv_{}".format(num_layers + layer_i)):
            net = tf.layers.conv2d(net, num_filters, conv_filter_size, padding='same', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            if dilation_size > 0:
                net = tf.layers.conv2d(net, num_filters, conv_filter_size_2, dilation_rate=(2, 2), padding='same', name="conv2")
                net = tf.nn.relu(net, name="relu2")

    assert len(conv) == 0

    # Output layer
    net = tf.layers.conv2d(net, 1, (1, 1), padding='same', name="weight_output")

    return net
