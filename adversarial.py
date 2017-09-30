import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

from model import deepnn

tf.flags.DEFINE_float("eps", 0.25, "Epsilon for FGSM.")
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
  target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)
  eps = tf.abs(eps)

  def _cond(x_adv, i):
    return tf.less(i, epochs)

  def _body(x_adv, i):
    ybar, logits = model(x_adv, logits=True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits),
        name="cse")
    dy_dx, = tf.gradients(loss, x_adv)
    x_adv = tf.stop_gradient(x_adv + eps * tf.sign(dy_dx))
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

    return x_adv, i + 1

  x_adv, _ = tf.while_loop(
      _cond, _body, (x_adv, 0), back_prop=True, name='fgsm')

  return x_adv


# ------------------------------------------------------#


def get_gradient(model, x, clip_min=0., clip_max=1.):
  x_grd = tf.identity(x)
  ybar = model(x_grd)
  yshape = tf.shape(ybar)
  ydim = yshape[1]

  indices = tf.argmax(ybar, axis=1)
  target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)

  ybar, logits = model(x_grd, logits=True)
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits),
      name="cse")

  dy_dx, = tf.gradients(loss, x_grd)

  return dy_dx


def select_digit_samples(data, digits=2, sample_size=10):
  """Samples mnist data for specified digit

  Args:
    data: A tf.contrib.learn.base.Datasets object, contains processed mnist data
    digit: An integer, the target digit to sample.
    sample_size: An integer, number of samples to return.
  """

  if not isinstance(digits, list):
    digits = [digits]

  sample_images, sample_labels = [], []

  for i in range(0, len(data.test.images)):
    image = data.test.images[i]
    label = data.test.labels[i]

    for d in digits:
      if label[d] == 1:
        sample_images.append(image)
        sample_labels.append(label)

    if len(sample_images) >= sample_size:
      break

  return sample_images, sample_labels


def write_jpeg(data, filepath):
  g = tf.Graph()
  with g.as_default():
    data_t = tf.placeholder(tf.uint8)
    op = tf.image.encode_jpeg(data_t, format='grayscale', quality=100)

  with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    data_np = sess.run(op, feed_dict={data_t: data})

  with open(filepath, 'w') as fd:
    fd.write(data_np)


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  with tf.Session() as sess:

    # restore model
    saver = tf.train.import_meta_graph(FLAGS.saved_model + "/" + FLAGS.meta)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model))

    # populate graph and get variables
    graph = tf.get_default_graph()

    # workaround for get_variable bug
    # https://github.com/tensorflow/tensorflow/issues/1325    
    for t in tf.global_variables():
      graph.get_tensor_by_name(t.name)

    x = graph.get_tensor_by_name("model/x:0")
    y_ = graph.get_tensor_by_name("model/y_:0")
    training = graph.get_tensor_by_name("model/mode:0")
    y = graph.get_tensor_by_name("model/y:0")
    accuracy = graph.get_tensor_by_name("model/acc:0")

    batch = select_digit_samples(mnist, digits=[2])

    # sanity check the accuracy
    prediction = tf.argmax(y, 1)
    classification = sess.run([prediction],
                              {x: batch[0],
                               y_: batch[1],
                               training: False})

    print(classification)
    print('pre perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: batch[0],
                   y_: batch[1],
                   training: False}))

    with tf.variable_scope('model', reuse=True):
      x_adv = fgsm(deepnn, x, FLAGS.eps, epochs=1)

    X_adv = sess.run(
        x_adv, feed_dict={x: batch[0],
                          y_: batch[1],
                          training: False})

    classification = sess.run([prediction],
                              {x: X_adv,
                               y_: batch[1],
                               training: False})

    print(classification)
    print('post perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: X_adv,
                   y_: batch[1],
                   training: False}))

    # apply target gradient transform
    target_batch = select_digit_samples(mnist, digits=6)
    with tf.variable_scope('model', reuse=True):
      grd = get_gradient(deepnn, x)

    gradients = sess.run(
        grd, {x: target_batch[0],
              y_: target_batch[1],
              training: False})

    X_adv = X_adv + np.sign(gradients)
    X_adv = np.clip(X_adv, 0.0, 1.0)

    classification = sess.run([prediction],
                              {x: X_adv,
                               y_: batch[1],
                               training: False})

    print(classification)
    print('post target perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: X_adv,
                   y_: batch[1],
                   training: False}))

    # output images
    output = []
    for i in range(len(X_adv)):

      # reshape to be valid image
      original = np.array(batch[0][i]).reshape(28, 28, 1)
      delta = np.divide(np.subtract(X_adv[i], batch[0][i]), FLAGS.eps).reshape(
          28, 28, 1)
      original_adv = np.array(X_adv[i]).reshape(28, 28, 1)

      # concatenate to rows
      out = np.concatenate([original, delta, original_adv], axis=1)
      out = np.array(out).reshape(28, 84, 1)
      out = np.multiply(out, 255)
      output.append(out)

    output = np.array(output).reshape(280, 84, 1)
    write_jpeg(output, "output.jpg".format(i))


if __name__ == "__main__":
  tf.app.run()