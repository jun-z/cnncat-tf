from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from csv import QUOTE_NONE as QN

tf.app.flags.DEFINE_string('input', '', 'Input file.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('embs_path', '', 'Path to pretrained embs.')
tf.app.flags.DEFINE_integer('random_seed', 12358, 'Random seed.')
tf.app.flags.DEFINE_integer('label_size', 0, 'Max size of labels.')
tf.app.flags.DEFINE_integer('vocab_size', 0, 'Max vocabulary size.')
tf.app.flags.DEFINE_integer('vocab_cutoff', 5, 'Cutoff for vocabulary.')
tf.app.flags.DEFINE_float('test_split', 0., 'Split for testing data.')
tf.app.flags.DEFINE_float('valid_split', .2, 'Split for validation data.')
tf.app.flags.DEFINE_bool('weights', False, 'Weighted training samples')
tf.app.flags.DEFINE_bool('flatten', False, 'Flatten to vectors.')
tf.app.flags.DEFINE_bool('oov_embs', False, 'Out of vocabulary embs.')
tf.app.flags.DEFINE_bool('aggregate', False, 'Aggregate weights.')
tf.app.flags.DEFINE_bool('best_only', False, 'Pick the best category.')
tf.app.flags.DEFINE_bool('normalize', False, 'Normalize the weights.')


FLAGS = tf.app.flags.FLAGS

if FLAGS.flatten:
    FLAGS.normalize = True

if FLAGS.best_only or FLAGS.normalize:
    FLAGS.aggregate = True


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
    if tokens:
        vocab = ['</s>', '<unk>']
        for k, v in tokens.iteritems():
            if v >= FLAGS.vocab_cutoff:
                vocab.append(k)
        if FLAGS.vocab_size > 0 and len(vocab) > FLAGS.vocab_size:
            vocab = vocab[:FLAGS.vocab_size]
    else:
        vocab = []
        with open(FLAGS.embs_path) as f:
            for i, line in enumerate(f):
                if i > 0:
                    vocab.append(line.split()[0])
        vocab.insert(1, '<unk>')
    return vocab


def read_data():
    data = pd.read_csv(
        FLAGS.input, names=get_names(),
        sep='\t', quoting=QN, dtype={'label': str})
    if not FLAGS.weights:
        data['weight'] = 1
    return data


def proc_embs(vocab):
    embs = {}
    with open(FLAGS.embs_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                emb_size = int(line.split()[1])
                vocab_size = len(vocab)
            else:
                embs[line.split()[0]] = line

    scale = np.sqrt(3) / np.sqrt(vocab_size + 2)
    with open(get_path('cnncat.embs.vec'), 'w') as f:
        f.write(' '.join([str(vocab_size), str(emb_size)]) + '\n')
        f.write(embs['</s>'])
        f.write(' '.join(['<unk>'] + ['0'] * emb_size) + '\n')
        for w in vocab[2:]:
            if w in embs:
                f.write(embs[w])
            else:
                ra = np.random.uniform(-scale, scale, emb_size)
                f.write(' '.join([w] + map(str, ra)) + '\n')

    return emb_size


def init(data):
    if not FLAGS.weights:
        weights = data.groupby(['label', 'text']).weight.sum()
        weights.name = 'weight'
        data = weights.to_frame().reset_index()

    labels = sorted(set(data.label))
    max_length = data.text.apply(lambda x: len(x.split())).max()

    if FLAGS.embs_path and FLAGS.oov_embs:
        vocab = get_vocab()
        emb_size = proc_embs(vocab)
    else:
        tokens = {}
        for i, r in data.iterrows():
            for t in r['text'].split():
                if t in tokens:
                    tokens[t] += r['weight']
                else:
                    tokens[t] = r['weight']
        vocab = get_vocab(tokens)

        if FLAGS.embs_path:
            emb_size = proc_embs(vocab)
        else:
            emb_size = None

    write_list(vocab, 'vocab')
    write_list(labels, 'labels')
    return vocab, labels, max_length, emb_size


def aggregate(data):
    weights = data.groupby(['label', 'text']).weight.sum()
    weights.name = 'weight'
    data = weights.to_frame().reset_index()

    if FLAGS.best_only:
        data.sort_values('weight', ascending=False, inplace=True)
        data.drop_duplicates('text', inplace=True)
        data.weight = 1

    return data


def normalize(data):
    train = data[data.split == 'train']
    other = data[data.split != 'train']

    if FLAGS.label_size:
        train = train.sort_values('weight', ascending=False)
        train = train.groupby('text').head(FLAGS.label_size)

    all_weights = train.groupby('text').weight.sum()
    all_weights.name = 'all_weights'
    all_weights = all_weights.to_frame().reset_index()
    train = pd.merge(train, all_weights)
    train.weight = train.weight / train.all_weights
    train.drop('all_weights', axis=1, inplace=True)

    train['split'] == 'train'

    return pd.concat([train, other], ignore_index=True)


def flatten(data):
    train = data[data.split == 'train']
    other = data[data.split != 'train']

    texts = []
    labels = []
    for t, d in train.groupby('text'):
        texts.append(t)
        label = {}
        for i, r in d.iterrows():
            label[r['label']] = r['weight']
        labels.append(label)

    train = pd.DataFrame({'text': texts, 'label': labels})
    train['weight'] = 1
    train['split'] = 'train'

    return pd.concat([train, other], ignore_index=True)


def serialize(label, tokens, length, weight):
    seq = tf.train.SequenceExample()

    if isinstance(label, dict):
        _labels = seq.feature_lists.feature_list['labels']
        _values = seq.feature_lists.feature_list['values']
        for l, v in label.iteritems():
            _labels.feature.add().int64_list.value.append(l)
            _values.feature.add().float_list.value.append(v)
    else:
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
    _vocab = {w: i for i, w in enumerate(_vocab)}
    _labels = {l: i for i, l in enumerate(_labels)}

    for k in ['train', 'valid', 'test']:
        writer = tf.python_io.TFRecordWriter(get_path('cnncat.%s.tfr' % k))
        for i, r in data[data.split == k].iterrows():
            if FLAGS.flatten and k == 'train':
                label = {_labels[k]: v for k, v in r['label'].iteritems()}
            else:
                label = _labels[r['label']]
            tokens = []
            for t in r['text'].split():
                if t in _vocab:
                    tokens.append(_vocab[t])
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
        'flattened': FLAGS.flatten,
        'num_steps': _length,
        'num_labels': len(_labels),
        'vocab_size': len(_vocab)
    }

    if FLAGS.embs_path:
        meta['emb_size'] = emb_size

    if FLAGS.aggregate:
        print('Aggregating...')
        data = aggregate(data)

    print('Splitting data...')
    split = get_split(data)
    data['split'] = data.text.apply(lambda x: split[x])

    if FLAGS.normalize:
        print('Normalizing...')
        data = normalize(data)

    if FLAGS.flatten:
        print('Flattening...')
        data = flatten(data)

    meta['train_size'] = data[data.split == 'train'].shape[0]

    print('Shuffling...')
    data = data.sample(frac=1.0)

    print('Writing meta data...')
    with open(get_path('meta.json'), 'w') as f:
        json.dump(meta, f)

    print('Writing to TFRecords...')
    write_records(data, _vocab, _labels, _length)

    print('Done.')
