from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import glob
import json
import numpy as np
import tensorflow as tf
from cnn import CNN

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_string('filter_sizes', '2,3', 'Filter sizes.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs.')
tf.app.flags.DEFINE_integer('num_filters', 64, 'Number of filters per size.')
tf.app.flags.DEFINE_integer('emb_size', 100, 'Size of embedding.')
tf.app.flags.DEFINE_integer('max_to_keep', 0, 'Max number of models to keep.')
tf.app.flags.DEFINE_float('beta', .0, 'Beta for penalization.')
tf.app.flags.DEFINE_float('l2_cost', .0, 'L2 Cost.')
tf.app.flags.DEFINE_float('keep_prob', .5, 'Keep prob for dropout')
tf.app.flags.DEFINE_float('learning_rate', .001, 'Learning rate.')
tf.app.flags.DEFINE_bool('trainable_embs', True, 'Trainable embeddings.')
tf.app.flags.DEFINE_bool('pretrained_embs', False, 'Pretrained embeddings.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')

FLAGS = tf.app.flags.FLAGS


def create_model(session, fn_queue, meta):
    if meta['train_size'] % FLAGS.batch_size > 0:
        steps = meta['train_size'] // FLAGS.batch_size + 1
    else:
        steps = meta['train_size'] // FLAGS.batch_size

    filter_sizes = [int(fs) for fs in FLAGS.filter_sizes.split(',')]

    model = CNN(
        fn_queue,
        FLAGS.batch_size,
        meta['num_steps'],
        meta['num_labels'],
        FLAGS.num_filters,
        FLAGS.emb_size,
        meta['vocab_size'],
        filter_sizes,
        FLAGS.beta,
        FLAGS.l2_cost,
        FLAGS.keep_prob,
        meta['flattened'],
        get_embs(),
        FLAGS.trainable_embs,
        FLAGS.learning_rate,
        FLAGS.max_to_keep,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.local_variables_initializer())
        epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        epoch += 1
    else:
        print('Created model with fresh parameters.')
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
        epoch = 0
    return epoch, steps, model


def get_meta():
    with open(os.path.join(FLAGS.data_dir, 'meta.json')) as f:
        meta = json.load(f)
    return meta


def get_embs():
    if FLAGS.pretrained_embs:
        emb_files = glob.glob(os.path.join(FLAGS.data_dir, '*.vec'))

        if len(emb_files) == 0:
            raise Exception('Did not find file with .vec ext.')
        elif len(emb_files) > 1:
            raise Exception('Found more than one file with .vec ext.')

        print('Loading embeddings from %s.' % emb_files[0])

        embs = []
        with open(emb_files[0]) as f:
            for i, line in enumerate(f):
                if i == 0:
                    vocab_size, emb_size = [int(s) for s in line.split()]
                else:
                    embs.append(np.asarray(line.split()[1:], dtype='float32'))

        return np.vstack(embs)
    else:
        return None


def train():
    train_files = []
    for f in os.listdir(FLAGS.data_dir):
        if f.endswith('.train.tfr'):
            train_files.append(os.path.join(FLAGS.data_dir, f))

    fn_queue = tf.train.string_input_producer(
        string_tensor=train_files, num_epochs=FLAGS.num_epochs)

    if not os.path.exists(FLAGS.train_dir):
        print('Directory %s does not exist, creating...' % FLAGS.train_dir)
        os.makedirs(FLAGS.train_dir)

    with tf.Session() as sess:
        meta = get_meta()
        epoch, steps, model = create_model(sess, fn_queue, meta)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_epochs = epoch + FLAGS.num_epochs

        try:
            step = 0
            print('Epoch %d' % epoch)
            while not coord.should_stop():
                start = time.time()
                _, loss = sess.run([model.train, model.loss])
                duration = time.time() - start
                if step % 100 == 0:
                    print('  Step %s - loss = %.2f (%.3f sec)' %
                          (str(step).ljust(len(str(steps))), loss, duration))
                step += 1
                if step == steps:
                    model.saver.save(
                        sess,
                        os.path.join(FLAGS.train_dir, 'cnncat.ckpt'),
                        global_step=epoch)
                    step = 0
                    epoch += 1
                    if epoch < num_epochs:
                        print('Epoch %d' % epoch)
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs.' % num_epochs)
            if step != 0:
                model.saver.save(
                    sess,
                    os.path.join(FLAGS.train_dir, 'cnncat.ckpt'),
                    global_step=epoch)
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    train()
