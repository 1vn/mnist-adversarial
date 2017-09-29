from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from model import deepnn

tf.flags.DEFINE_string("data_dir", "tmp/data", "The data directory.")
tf.flags.DEFINE_string("output_dir", "tmp/run", "The output directory.")
tf.flags.DEFINE_integer("train_steps", 1000, "The amount of steps to train.")
FLAGS = tf.app.flags.FLAGS


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784], name="x")
  y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

  training = tf.placeholder(bool, (), name='mode')

  y_conv = deepnn(x, training=training)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv),
      name="cse")
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="acc")

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
    saver.save(sess, FLAGS.output_dir + "/model", global_step=i)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        training: False
    }))


if __name__ == "__main__":
  tf.app.run()