from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string("data_dir", "tmp/data", "The data directory.")
tf.flags.DEFINE_string("meta", "model-999.meta", "The saved meta graph")
tf.flags.DEFINE_string("saved_model", "saved_model",
                       "The folder containing saved model")
FLAGS = tf.app.flags.FLAGS


def select_digit_samples(data, digit=2, sample_size=10):
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
    accuracy = graph.get_tensor_by_name("acc:0")

    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
    y = graph.get_tensor_by_name("y:0")

    batch = select_digit_samples(mnist)

    feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}

    classifications = sess.run(y, feed_dict)

    # sanity check the accuracy
    print('pre perturbations accuracy: %g' % accuracy.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  tf.app.run()