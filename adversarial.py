import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

from model import deepnn

tf.flags.DEFINE_integer(
    "origin", 2, "The origin MNIST class to generate adversarial examples for.")
tf.flags.DEFINE_integer(
    "target", 6, "The target MNIST class to pertubate origin samples into.")
tf.flags.DEFINE_integer(
    "wiggle_steps", 10000,
    "Upper bound on the number of wiggle operations in case epsilon is too big.")
tf.flags.DEFINE_integer(
    "sample_size", 10,
    "The number of samples to generate. (origin.sample_size = target.sample_size)"
)
tf.flags.DEFINE_float(
    "eps", 0.01,
    "The epsilon amount to wiggle towards the network gradient of target class.")
tf.flags.DEFINE_string("data_dir", "tmp/data",
                       "The data directory to save/load MNIST data.")
tf.flags.DEFINE_boolean("verbose", False, "Print info if true")
tf.flags.DEFINE_string("model_file", "model.meta",
                       "The filename of the saved meta graph.")
tf.flags.DEFINE_string(
    "model_dir", "tmp/run",
    "The model directory to load the trained MNIST classifier.")
tf.flags.DEFINE_string("output", "output.jpg",
                       "The desired filename of the output table image.")
FLAGS = tf.app.flags.FLAGS


def printOut(s):
  if FLAGS.verbose:
    print(s)


def get_gradient(model, x):
  """Returns gradient of network w.r.t. input


  Args:
    model: A function, model function which satisfies the signature
           def(x, logits) such that the final layer and a
           logits term is returned when logits parameter
           is true.
    x: A Tensor, input placeholder
  """
  x_grd = tf.identity(x)
  y_, logits = model(x_grd, logits=True)
  predict = tf.argmax(y_, axis=1)
  target = tf.one_hot(predict, tf.shape(y_)[1], on_value=1.0, off_value=0.0)
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


# from https://stackoverflow.com/questions/40320271/how-do-we-use-tf-image-encode-jpeg-to-write-out-an-image-in-tensorflow
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


def logOps(x, y_, training, x_adv, origin_batch, classification, i, count,
           limit, y):
  percent = y.eval({
      x: [x_adv[i]],
      y_: [origin_batch[1][i]],
      training: False
  })[0]
  printOut("image {} - target {}: {}, current {}: {}".format(
      i + 1, FLAGS.target, percent[FLAGS.target], classification, percent[
          classification]))
  count += 1
  if classification == FLAGS.target:
    print("found delta for image {}".format(i + 1))

  if count == limit:
    printOut("exit due to limit")


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  with tf.Session() as sess:

    # restore model
    saver = tf.train.import_meta_graph(FLAGS.model_dir + "/" + FLAGS.model_file)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))

    # populate graph and get variables
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("model/x:0")
    y_ = graph.get_tensor_by_name("model/y_:0")
    training = graph.get_tensor_by_name("model/mode:0")
    y = graph.get_tensor_by_name("model/y:0")
    accuracy = graph.get_tensor_by_name("model/acc:0")

    # get origin class examples
    origin_batch = select_digit_samples(
        mnist, digits=FLAGS.origin, sample_size=FLAGS.sample_size)

    prediction = tf.argmax(y, axis=1)

    # sanity check the origin accuracy
    classification = prediction.eval({
        x: origin_batch[0],
        y_: origin_batch[1],
        training: False
    })

    print("pre perturbations predictions: {}".format(classification))
    print('pre perturbations accuracy on origin class: %g' % accuracy.eval(
        feed_dict={x: origin_batch[0],
                   y_: origin_batch[1],
                   training: False}))

    # gradient operation
    with tf.variable_scope('model', reuse=True):
      grd = get_gradient(deepnn, x)

    # get target class examples
    target_batch = select_digit_samples(
        mnist, digits=FLAGS.target, sample_size=FLAGS.sample_size)

    # wiggle pixels towards target class
    x_adv = origin_batch[0][:]

    print("wiggling {} images...".format(FLAGS.sample_size))
    for i in range(len(x_adv)):
      count = 0

      # need this in case wiggling doesn't converge (i.e. too large of eps)
      limit = FLAGS.wiggle_steps

      # initial pre-wiggling classification
      classification = prediction.eval({
          x: [x_adv[i]],
          y_: [origin_batch[1][i]],
          training: False
      })[0]

      while classification != FLAGS.target and count < limit:
        # get gradients from classifying target class
        gradients_adv = grd.eval({
            x: [x_adv[i]],
            y_: [target_batch[1][i]],
            training: False
        })[0]

        # modified version of fast gradient sign method
        x_adv[i] = x_adv[i] + np.sign(gradients_adv) * FLAGS.eps
        x_adv[i] = np.clip(x_adv[i], 0.0, 1.0)

        # re-evaluate class
        classification = prediction.eval({
            x: [x_adv[i]],
            y_: [origin_batch[1][i]],
            training: False
        })[0]

        logOps(x, y_, training, x_adv, origin_batch, classification, i, count,
               limit, y)

    classification = prediction.eval({
        x: x_adv,
        y_: target_batch[1],
        training: False
    })
    print("post perturbations predictions: {}".format(classification))
    print('post target perturbations accuracy on target class: %g' % accuracy.
          eval(feed_dict={x: x_adv,
                          y_: target_batch[1],
                          training: False}))

    # write images to disk
    output = []
    for i in range(len(x_adv)):
      # reshape to be valid image
      original = np.array(origin_batch[0][i]).reshape(28, 28, 1)
      delta = np.subtract(x_adv[i], origin_batch[0][i]).reshape(28, 28, 1)
      original_adv = np.array(x_adv[i]).reshape(28, 28, 1)

      # concatenate to rows
      out = np.concatenate([original, delta, original_adv], axis=1)
      out = np.array(out).reshape(28, 84, 1)
      out = np.multiply(out, 255)
      output.append(out)

    output = np.array(output).reshape(28 * FLAGS.sample_size, 84, 1)
    write_jpeg(output, FLAGS.output)


if __name__ == "__main__":
  tf.app.run()