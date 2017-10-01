import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

from model import deepnn

tf.flags.DEFINE_integer("initial", 2,
                        "Initial class selected for perturbations.")
tf.flags.DEFINE_integer("target", 6, "Target class to pertubate to.")
tf.flags.DEFINE_integer("sample_size", 10, "Number of samples to generate.")
tf.flags.DEFINE_float("eps", 0.25, "Epsilon for FGSM.")
tf.flags.DEFINE_string("data_dir", "tmp/data", "The data directory.")
tf.flags.DEFINE_string("meta", "model.meta", "The saved meta graph")
tf.flags.DEFINE_string("saved_model", "saved_model",
                       "The folder containing saved model")
FLAGS = tf.app.flags.FLAGS


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

    x = graph.get_tensor_by_name("model/x:0")
    y_ = graph.get_tensor_by_name("model/y_:0")
    training = graph.get_tensor_by_name("model/mode:0")
    y = graph.get_tensor_by_name("model/y:0")
    accuracy = graph.get_tensor_by_name("model/acc:0")

    batch = select_digit_samples(
        mnist, digits=FLAGS.initial, sample_size=FLAGS.sample_size)

    # sanity check the accuracy
    prediction = tf.argmax(y, axis=1)
    classification = sess.run([prediction],
                              {x: batch[0],
                               y_: batch[1],
                               training: False})

    print(classification)
    print('pre perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: batch[0],
                   y_: batch[1],
                   training: False}))

    # apply target gradient transform
    target_batch = select_digit_samples(
        mnist, digits=FLAGS.target, sample_size=FLAGS.sample_size)
    with tf.variable_scope('model', reuse=True):
      grd = get_gradient(deepnn, x)

    # get initial class gradients 
    gradients_og = grd.eval({x: batch[0], y_: batch[1], training: False})

    # get gradients from classifying target
    gradients_adv = grd.eval({
        x: target_batch[0],
        y_: target_batch[1],
        training: False
    })

    # wiggle pixels for each image

    # for some reason this makes the output image better
    X_adv = np.array(batch[0][:])
    X_adv = X_adv + np.sign(gradients_og) * FLAGS.eps
    X_adv = X_adv - np.sign(gradients_adv) * FLAGS.eps
    X_adv = np.clip(X_adv, 0.0, 1.0)
    classification = sess.run([prediction],
                              {x: X_adv,
                               y_: batch[1],
                               training: False})

    percent = y.eval({x: X_adv, y_: batch[1], training: False})
    # wiggling
    print("wiggling...")
    for i in range(len(X_adv)):
      count = 0

      # need this until proof that wiggling algorithm will converge
      limit = 10000

      while classification[0][i] != FLAGS.target and count < limit:
        X_adv[i] = X_adv[i] - np.sign(gradients_og[i]) * FLAGS.eps * 0.001
        X_adv[i] = X_adv[i] + np.sign(gradients_adv[i]) * FLAGS.eps * 0.01
        X_adv[i] = np.clip(X_adv[i], 0.0, 1.0)
        classification[0][i] = sess.run(
            [prediction], {x: [X_adv[i]],
                           y_: [batch[1][i]],
                           training: False})[0][0]
        gradients_adv = grd.eval({
            x: X_adv,
            y_: target_batch[1],
            training: False
        })

        percent = y.eval({x: [X_adv[i]], y_: [batch[1][i]], training: False})[0]
        print("image {} - target {}: {}, current {}: {}".format(
            i + 1, FLAGS.target, percent[FLAGS.target], classification[0][i],
            percent[classification[0][i]]))

        count += 1
        if classification[0][i] == FLAGS.target:
          print("found delta for image {}".format(i + 1))

        if count == limit:
          print("exit due to limit")

    classification = sess.run([prediction],
                              {x: X_adv,
                               y_: batch[1],
                               training: False})
    print(classification)
    print('post target perturbations accuracy: %g' % accuracy.eval(
        feed_dict={x: X_adv,
                   y_: target_batch[1],
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

    output = np.array(output).reshape(28 * FLAGS.sample_size, 84, 1)
    write_jpeg(output, "output.jpg".format(i))


if __name__ == "__main__":
  tf.app.run()