from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import tensorflow as tf
from cnn import CNN

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_string('filter_sizes', '2,3', 'Filter sizes.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs.')
tf.app.flags.DEFINE_integer('num_steps', 500, 'Max number of time steps')
tf.app.flags.DEFINE_integer('num_labels', 46, 'Number of labels.')
tf.app.flags.DEFINE_integer('num_filters', 64, 'Number of filters per size.')
tf.app.flags.DEFINE_integer('emb_size', 100, 'Size of embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 1000, 'Size of vocabulary.')
tf.app.flags.DEFINE_float('l2_cost', .0, 'L2 Cost.')
tf.app.flags.DEFINE_float('keep_prob', .5, 'Keep prob for dropout')
tf.app.flags.DEFINE_float('learning_rate', .001, 'Learning rate.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')

FLAGS = tf.app.flags.FLAGS


def create_model(session, fn_queue):
    filter_sizes = [int(fs) for fs in FLAGS.filter_sizes.split(',')]

    model = CNN(
        fn_queue,
        FLAGS.batch_size,
        FLAGS.num_steps,
        FLAGS.num_labels,
        FLAGS.num_filters,
        FLAGS.emb_size,
        FLAGS.vocab_size,
        filter_sizes,
        FLAGS.l2_cost,
        FLAGS.keep_prob,
        FLAGS.learning_rate,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.local_variables_initializer())
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('Created model with fresh parameters.')
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
        step = 0
    return step, model


def train():
    train_files = []
    for f in os.listdir(FLAGS.data_dir):
        if f.endswith('.train.tfr'):
            train_files.append(os.path.join(FLAGS.data_dir, f))

    fn_queue = tf.train.string_input_producer(
        string_tensor=train_files, num_epochs=FLAGS.num_epochs)

    with tf.Session() as sess:
        step, model = create_model(sess, fn_queue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                start = time.time()
                _, loss = sess.run([model.train, model.loss])
                duration = time.time() - start

                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' %
                          (step, loss, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' %
                  (FLAGS.num_epochs, step))
        finally:
            model.saver.save(
                sess,
                os.path.join(FLAGS.train_dir, 'doccat.ckpt'),
                global_step=step)

            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    train()