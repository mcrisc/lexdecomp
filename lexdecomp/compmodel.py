"""Describes the sentence composition model.

The model is implemented as a CNN, as described in Wang et al. (2016)
[https://arxiv.org/abs/1602.07019].

This module implements the inference/loss/training pattern, adopted in
TensorFlow tutorials (see tutorial "TensorFlow Mechanics 101", section
Build the Graph).
"""
import tensorflow as tf


# hyper parameters from Wang et al. (2016):
FILTER_SIZES = [1, 2, 3]
NUM_FEATURE_MAPS = 500


def conv_filter(filter_size, embedding_size, in_channels):
    """Creates a convolutional filter."""
    filter_shape = [embedding_size, filter_size, in_channels, NUM_FEATURE_MAPS]
    weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[NUM_FEATURE_MAPS]),
                         dtype=tf.float32)
    return weights, biases


def conv_layer(in_data):
    """Apply a set of convolutional filters over input data.

    :param in_data: a tensor of shape [batch, EMBEDDING_SIZE,
    `sequence_length`, IN_CHANNELS]
    :returns: a tensor of shape [batch, total_feature_maps]
    """
    if hasattr(in_data, 'shape'):  # a Numpy array
        embedding_size = in_data.shape[1]
        sequence_length = in_data.shape[2]
        in_channels = in_data.shape[3]
    if hasattr(in_data, 'get_shape'):  # a TensorFlow placeholder
        embedding_size = in_data.get_shape()[1].value
        sequence_length = in_data.get_shape()[2].value
        in_channels = in_data.get_shape()[3].value
    else:
        raise TypeError('in_data must be either a Numpy array or '
                        'a TensorFlow placeholder')

    # convolutional layer
    pooled_outputs = []
    for filter_size in FILTER_SIZES:
        weights, biases = conv_filter(filter_size, embedding_size, in_channels)
        conv = tf.nn.conv2d(in_data, weights,
                            strides=[1, 1, 1, 1], padding='VALID')
        feature_map = tf.tanh(tf.nn.bias_add(conv, biases))
        # feature_map = tf.nn.relu(tf.nn.bias_add(conv, biases))
        width_conv = sequence_length - filter_size + 1
        pooled = tf.nn.max_pool(feature_map, ksize=[1, 1, width_conv, 1],
                                strides=[1, 1, 1, 1], padding='VALID')
        # pooled.shape = [batch, 1, 1, NUM_FEATURE_MAPS]
        pooled_outputs.append(pooled)

    # concatenating feature maps
    features = tf.concat(pooled_outputs,axis = 3)
#    features = [pooled_outputs[0], pooled_outputs[1], pooled_outputs[2]]
    total_feature_maps = len(FILTER_SIZES) * NUM_FEATURE_MAPS
    features_flat = tf.reshape(features, [-1, total_feature_maps])
    # features_flat.shape = [batch, total_feature_maps]
    return features_flat


def hidden_layer(features, keep_prob):
    """Build the fully-connected layer.
    """
    NUM_CLASSES = 1
    input_neurons = 2 * len(FILTER_SIZES) * NUM_FEATURE_MAPS

    weights = tf.Variable(tf.truncated_normal(
        [input_neurons, NUM_CLASSES], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]),
                         dtype=tf.float32)
    features_drop = tf.nn.dropout(features, keep_prob=keep_prob)
    scores = tf.nn.bias_add(
        tf.matmul(features_drop, weights), biases)
    scores_flat = tf.reshape(scores, [-1])
    prob_scores = tf.nn.softmax(scores_flat)
    return prob_scores


def inference(questions, sentences, keep_prob):
    """Build the composition model (CNN).
    """
    question_features = conv_layer(questions)
    sentence_features = conv_layer(sentences)
    features = tf.concat([question_features, sentence_features],axis=1)
#    features = [1, question_features, sentence_features]
    scores = hidden_layer(features, keep_prob)
    return scores


def loss(logits, labels):
    """Computes the loss from the logits and the labels.

    :param logits: scores predicted by the model - [batch,]
    :param labels: expected labels - [batch,]
    """
    # cross entropy
    expected = tf.nn.softmax(labels)
    loss = -tf.reduce_sum(expected * tf.log(logits))
    return loss


def training(loss):
    tf.summary.scalar('loss', loss)
    # optimizer
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
