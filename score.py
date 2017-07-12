from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import pandas as pd
import tensorflow as tf
from csv import QUOTE_NONE as QN

tf.app.flags.DEFINE_string('input', '', 'Input file.')
tf.app.flags.DEFINE_string('output', '', 'Output file.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_string('filter_sizes', '2,3', 'Filter sizes.')
tf.app.flags.DEFINE_integer('batch_size', 0, 'Batch size.')
tf.app.flags.DEFINE_integer('num_probs', 4, 'Number of probabilities.')
tf.app.flags.DEFINE_integer('num_filters', 64, 'Number of filters per size.')
tf.app.flags.DEFINE_integer('emb_size', 100, 'Size of embedding.')
tf.app.flags.DEFINE_float('learning_rate', .001, 'Learning rate.')
tf.app.flags.DEFINE_bool('labels', False, 'Whether the data has labels.')
tf.app.flags.DEFINE_bool('weights', False, 'Whether the data is weighted.')
tf.app.flags.DEFINE_bool('aggregate', False, 'Whether to aggregate data.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')

FLAGS = tf.app.flags.FLAGS

with open(os.path.join(FLAGS.data_dir, 'meta.json')) as f:
    meta = json.load(f)


class CNN(object):
    def __init__(self,
                 num_steps,
                 num_labels,
                 num_filters,
                 emb_size,
                 vocab_size,
                 filter_sizes,
                 keep_prob,
                 dtype=tf.float32):

        self.tokens = tf.placeholder(tf.int32, [None, num_steps])

        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'embedding', [vocab_size, emb_size], dtype=dtype)

            inp_emb = tf.expand_dims(
                tf.nn.embedding_lookup(embedding, self.tokens), -1)

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('ConvPool-%i' % filter_size):
                W = tf.get_variable(
                    'W', [filter_size, emb_size, 1, num_filters], dtype,
                    initializer=tf.truncated_normal_initializer(stddev=.1))
                b = tf.get_variable(
                    'b', [num_filters], dtype,
                    initializer=tf.constant_initializer(.1))

                conved = tf.nn.conv2d(
                    inp_emb, W,
                    strides=[1, 1, 1, 1], padding='VALID')

                pooled = tf.nn.max_pool(
                    tf.nn.relu(tf.nn.bias_add(conved, b)),
                    [1, num_steps - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
                outputs.append(pooled)

        output = tf.squeeze(tf.concat(3, outputs), axis=[1, 2])
        dropped = tf.nn.dropout(output, keep_prob)

        with tf.variable_scope('Projection'):
            W = tf.get_variable(
                'W', [len(filter_sizes) * num_filters, num_labels], dtype,
                initializer=tf.truncated_normal_initializer(stddev=.01))
            b = tf.get_variable(
                'b', [num_labels], dtype,
                initializer=tf.constant_initializer(.1))

        logits = tf.matmul(dropped, W) + b

        self.probs = tf.nn.softmax(logits)
        self.saver = tf.train.Saver(tf.global_variables())


class Model():
    def __init__(self, train_dir, num_steps):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model = load_model(self.sess, num_steps)

    def predict(self, tokens):
        return self.sess.run(
            self.model.probs, feed_dict={self.model.tokens: tokens})


def get_names():
    names = ['text']
    if FLAGS.labels:
        names.insert(0, 'label')
    if FLAGS.weights:
        names.append('weight')
    return names


def load_data(data_dir, ext):
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith(ext):
            with open(os.path.join(data_dir, f)) as f:
                lines = f.readlines()
    data = [line[:-1] if line.endswith('\n') else line for line in lines]
    return data


def load_model(session, num_steps):
    filter_sizes = [int(fs) for fs in FLAGS.filter_sizes.split(',')]

    model = CNN(
        num_steps,
        meta['num_labels'],
        FLAGS.num_filters,
        FLAGS.emb_size,
        meta['vocab_size'],
        filter_sizes,
        1.0,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.local_variables_initializer())
        return model
    else:
        raise Exception('Could not find model checkpoint.')


def get_tokens(chunk, vocab, num_steps):
    tokens = []
    for i, r in chunk.iterrows():
        row = [vocab.index(t)
               if t in vocab else 1
               for t in r['text'].split()]
        if len(row) > num_steps:
            tokens.append(row[:num_steps])
        else:
            tokens.append(row + [0] * (num_steps - len(row)))
    return tokens


def get_chunks():
    if FLAGS.aggregate:
        data = pd.read_csv(
            FLAGS.input, names=get_names(), sep='\t', quoting=QN)

        if not FLAGS.weights:
            data['weight'] = 1
        if FLAGS.labels:
            weights = data.groupby(['label', 'text']).weight.sum()
            weights.name = 'weight'
            data = weights.to_frame().reset_index()
            data.label = data.label.astype(str)
        else:
            weights = data.groupby('text').weight.sum()
            weights.name = 'weight'
            data = weights.to_frame().reset_index()

        chunks = []
        if len(data) < FLAGS.batch_size:
            chunks.append(data)
        else:
            nc = len(data) // FLAGS.batch_size
            for i in range(nc):
                chunks.append(data[i::nc])
    else:
        data = pd.read_csv(
            FLAGS.input, names=get_names(),
            sep='\t', quoting=QN, chunksize=FLAGS.batch_size)
    return chunks


def get_num_steps():
    chunks = get_chunks()
    num_steps = 0
    for chunk in chunks:
        _steps = chunk.text.apply(lambda x: len(x.split())).max()
        num_steps = max(num_steps, _steps)
    return num_steps


def main(_):
    if len(FLAGS.input) == 0:
        raise ValueError('Input cannot be empty.')

    if len(FLAGS.output) == 0:
        raise ValueError('Output cannot be empty.')

    vocab = load_data(FLAGS.data_dir, '.vocab')
    labels = load_data(FLAGS.data_dir, '.labels')
    chunks = get_chunks()
    num_steps = get_num_steps()

    model = Model(FLAGS.train_dir, num_steps)
    if FLAGS.labels:
        total = 0
        correct = 0
    for chunk in chunks:
        probs = model.predict(get_tokens(chunk, vocab, num_steps))
        idx = probs.argsort()
        for i in range(1, FLAGS.num_probs + 1):
            chunk['prob%i' % i] = [probs[j, l]
                                   for j, l in enumerate(idx[:, -i])]
            chunk['pred%i' % i] = [labels[l]
                                   for l in idx[:, -i]]

        if FLAGS.labels:
            total += chunk.weight.sum()
            correct += chunk[chunk.label == chunk.pred1].weight.sum()

        chunk.to_csv(
            FLAGS.output, index=False, header=False,
            sep='\t', quoting=QN,  mode='a')

    if FLAGS.labels:
        print('Accuracy: %.4f.' % (correct / total))


if __name__ == '__main__':
    tf.app.run()
