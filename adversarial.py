from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

from model import deepnn

tf.flags.DEFINE_string("data_dir", "tmp/data", "The data directory.")
tf.flags.DEFINE_string("meta", "model.meta", "The saved meta graph")
tf.flags.DEFINE_string("saved_model", "saved_model",
                       "The folder containing saved model")
FLAGS = tf.app.flags.FLAGS


# ------------------------------------------------------#
# from https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks/fgsm.py
def fgsm(model, x, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
  """
    Fast gradient sign method.
    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533 for
    details.  This implements the revised version, since the original FGSM has
    label leaking problem (https://arxiv.org/abs/1611.01236).
    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
  x_adv = tf.identity(x)

  ybar = model(x_adv)
  yshape = tf.shape(ybar)
  ydim = yshape[1]

  indices = tf.argmax(ybar, axis=1)
  target = tf.cond(
      tf.equal(ydim, 1), lambda: tf.nn.relu(tf.sign(0.5 - ybar)),
      lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

  eps = tf.abs(eps)

  def _cond(x_adv, i):
    return tf.less(i, epochs)

  def _body(x_adv, i):
    ybar, logits = model(x_adv)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
    dy_dx, = tf.gradients(loss, x_adv)
    x_adv = tf.stop_gradient(x_adv + eps * tf.sign(dy_dx))
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
    return x_adv, i + 1

  x_adv, _ = tf.while_loop(
      _cond, _body, (x_adv, 0), back_prop=False, name='fgsm')
  return x_adv


# ------------------------------------------------------#


def select_digit_samples(data, digit=2, sample_size=10):
  """Samples mnist data for specified digit

  Args:
    data: A tf.contrib.learn.base.Datasets object, contains processed mnist data
    digit: An integer, the target digit to sample.
    sample_size: An integer, number of samples to return.
  """

  sample_images, sample_labels = [], []

  for i in range(0, len(data.test.images)):
    image = data.test.images[i]
    label = data.test.labels[i]

    if label[digit] == 1:
      sample_images.append(image)
      sample_labels.append(label)

    if len(sample_images) > sample_size:
      break

  return sample_images, sample_labels


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  with tf.Session() as sess:

    # restore model
    saver = tf.train.import_meta_graph(FLAGS.saved_model + "/" + FLAGS.meta)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model))

    # populate graph and get variables
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("model/x:0")
    y_ = graph.get_tensor_by_name("model/y_:0")
    training = graph.get_tensor_by_name("model/mode:0")
    y = graph.get_tensor_by_name("model/y:0")
    accuracy = graph.get_tensor_by_name("model/acc:0")

    batch = select_digit_samples(mnist)

    # sanity check the accuracy
    print('pre perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: batch[0],
                   y_: batch[1],
                   training: False}))

    with tf.variable_scope('model', reuse=True):
      x_adv = fgsm(deepnn, x, 0.01)

    adv_classifications = sess.run(y, {x: x_adv, y_: batch[1], training: 1.0})


if __name__ == "__main__":
  tf.app.run()