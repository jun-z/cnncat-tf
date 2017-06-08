from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
from train import FLAGS, create_model

tf.app.flags.DEFINE_string('action', 'valid', 'Valid or test.')


def _eval():
    if FLAGS.action not in ['valid', 'test']:
        raise ValueError('Unrecognized action: %s' % FLAGS.action)

    eval_files = []
    for f in os.listdir(FLAGS.data_dir):
        if f.endswith('.%s.tfr' % FLAGS.action):
            eval_files.append(os.path.join(FLAGS.data_dir, f))

    fn_queue = tf.train.string_input_producer(
        string_tensor=eval_files, num_epochs=1)

    with tf.Session() as sess:
        _, _, model = create_model(sess, fn_queue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        corrects = 0
        total = 0
        try:
            step = 0
            while not coord.should_stop():
                loss, probs, labels, weights = sess.run(
                    [model.loss, model.probs, model.label, model.weight])
                corrects += ((labels == probs.argmax(axis=1)) * weights).sum()
                total += weights.sum()
                step += 1
        except tf.errors.OutOfRangeError:
            print('Evaluation done, weighted accuracy %.4f.'
                  % (corrects / total))
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    _eval()
