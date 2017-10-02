from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from model import deepnn

tf.flags.DEFINE_string("data_dir", "tmp/data",
                       "The data directory to save/load MNIST data.")
tf.flags.DEFINE_string("output_dir", "tmp/run",
                       "The output directory for the trained model.")
tf.flags.DEFINE_integer("train_steps", 1000, "The amount of steps to train.")
FLAGS = tf.app.flags.FLAGS


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.variable_scope('model'):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    training = tf.placeholder(bool, (), name='mode')

    ybar, logits = deepnn(x, logits=True, training=training)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits),
        name="cse")

    correct_prediction = tf.equal(tf.argmax(ybar, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32), name="acc")

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.train_steps):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0],
                       y_: batch[1],
                       training: False})
        print('step %d, training accuracy %g' % (i, train_accuracy))

      train_step.run(feed_dict={x: batch[0], y_: batch[1], training: True})

    saver = tf.train.Saver()
    saver.save(sess, FLAGS.output_dir + "/model")

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        training: False
    }))


if __name__ == "__main__":
  tf.app.run()