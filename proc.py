from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from csv import QUOTE_NONE as QN

tf.app.flags.DEFINE_string('input', '', 'Input file.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('aggregation', 'none', 'Aggregation scheme.')
tf.app.flags.DEFINE_integer('vocab_size', 100000, 'Max vocabulary size.')
tf.app.flags.DEFINE_integer('random_seed', 12358, 'Random seed.')
tf.app.flags.DEFINE_float('test_split', 0., 'Split for testing data.')
tf.app.flags.DEFINE_float('valid_split', .2, 'Split for validation data.')
tf.app.flags.DEFINE_bool('weights', False, 'Weighted training samples')
tf.app.flags.DEFINE_bool('pretrained_embs', False, 'Use word vectors.')

FLAGS = tf.app.flags.FLAGS


def get_path(fn):
    return os.path.join(FLAGS.data_dir, fn)


def get_names():
    names = ['label', 'text']
    if FLAGS.weights:
        names.append('weight')
    return names


def get_split(data):
    split = {}
    for t in data.text.unique():
        rn = np.random.random()
        if rn < FLAGS.test_split:
            split[t] = 'test'
        elif rn < FLAGS.test_split + FLAGS.valid_split:
            split[t] = 'valid'
        else:
            split[t] = 'train'
    return split


def get_vocab(tokens=None):
    specials = ['<pad>', '<unk>']
    if tokens:
        vocab = specials + sorted(tokens, key=tokens.get, reverse=True)
        if len(vocab) > FLAGS.vocab_size:
            vocab = vocab[:FLAGS.vocab_size]
        return vocab
    else:
        embs = glob.glob(get_path('*.vec'))
        tokens = []

        if len(embs) == 0:
            raise Exception('Did not find file with .vec ext.')
        elif len(embs) > 1:
            raise Exception('Found more than one file with .vec ext.')

        with open(embs[0]) as f:
            for i, line in enumerate(f):
                if i == 0:
                    emb_size = int(line.split()[1])
                else:
                    tokens.append(line.split()[0])
        return specials + tokens, emb_size


def read_data():
    data = pd.read_csv(
        FLAGS.input, names=get_names(), sep='\t', quoting=QN)
    if not FLAGS.weights:
        data['weight'] = 1
    return data


def init(data):
    if not FLAGS.weights:
        weights = data.groupby(['label', 'text']).weight.sum()
        weights.name = 'weight'
        data = weights.to_frame().reset_index()

    tokens = {}
    labels = set()
    max_length = data.text.apply(lambda x: len(x.split())).max()
    for i, r in data.iterrows():
        labels.add(str(r['label']))
        for t in r['text'].split():
            if t in tokens:
                tokens[t] += r['weight']
            else:
                tokens[t] = r['weight']

    if FLAGS.pretrained_embs:
        vocab, emb_size = get_vocab()
    else:
        vocab = get_vocab(tokens)
        emb_size = None

    labels = sorted(labels)

    write_list(vocab, 'vocab')
    write_list(labels, 'labels')
    return vocab, labels, max_length, emb_size


def aggregate(data):
    if FLAGS.aggregation not in ['none', 'asis', 'best', 'norm', 'flat']:
        raise ValueError('Unknown aggregation scheme %s.' % FLAGS.aggregation)

    if FLAGS.aggregation != 'none':
        weights = data.groupby(['label', 'text']).weight.sum()
        weights.name = 'weight'
        data = weights.to_frame().reset_index()

        if FLAGS.aggregation == 'asis':
            pass
        elif FLAGS.aggregation == 'best':
            data.sort_values('weight', ascending=False, inplace=True)
            data.drop_duplicates('text', inplace=True)
            data.weight = 1
        else:
            all_weights = data.groupby('text').weight.sum()
            all_weights.name = 'all_weights'
            all_weights = all_weights.to_frame().reset_index()
            data = pd.merge(data, all_weights)
            data.weight = data.weight / data.all_weights
            data.drop('all_weights', axis=1, inplace=True)

    return data.sample(frac=1.0)


def serialize(label, tokens, length, weight):
    seq = tf.train.SequenceExample()

    seq.context.feature['label'].int64_list.value.append(label)
    seq.context.feature['length'].int64_list.value.append(length)
    seq.context.feature['weight'].float_list.value.append(weight)

    _tokens = seq.feature_lists.feature_list['tokens']
    for t in tokens:
        _tokens.feature.add().int64_list.value.append(t)
    return seq.SerializeToString()


def write_list(l, kind):
    with open(get_path('cnncat.%s' % kind), 'w') as f:
        f.write('\n'.join(l))


def write_records(data, _vocab, _labels, _length):
    for k in ['train', 'valid', 'test']:
        writer = tf.python_io.TFRecordWriter(get_path('nopres.%s.tfr' % k))
        for i, r in data[data.split == k].iterrows():
            label = _labels.index(str(r['label']))
            tokens = []
            for t in r['text'].split():
                if t in _vocab:
                    tokens.append(_vocab.index(t))
                else:
                    tokens.append(1)

            length = len(tokens)
            tokens += [0] * (_length - length)
            writer.write(
                serialize(label, tokens, length, r['weight']))
        writer.close()


if __name__ == '__main__':
    np.random.seed(FLAGS.random_seed)

    print('Reading data...')
    data = read_data()

    print('Initializing...')
    _vocab, _labels, _length, emb_size = init(data)

    meta = {
        'num_steps': _length,
        'num_labels': len(_labels),
        'vocab_size': len(_vocab)
    }

    if FLAGS.pretrained_embs:
        meta['emb_size'] = emb_size

    print('Splitting data...')
    split = get_split(data)

    print('Aggregating...')
    data = aggregate(data)

    data['split'] = data.text.apply(lambda x: split[x])
    meta['train_size'] = data[data.split == 'train'].shape[0]

    print('Writing meta data...')
    with open(get_path('meta.json'), 'w') as f:
        json.dump(meta, f)

    print('Writing to TFRecords...')
    write_records(data, _vocab, _labels, _length)

    print('Done.')
