import tensorflow as tf


def get_batch(fn_queue, num_steps, num_labels, batch_size, flattened):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(fn_queue)

    context_features = {
        'length': tf.FixedLenFeature([], dtype=tf.int64),
        'weight': tf.FixedLenFeature([], dtype=tf.float32),
    }
    sequence_features = {
        'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    if flattened:
        sequence_features.update({
            'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'values': tf.FixedLenSequenceFeature([], dtype=tf.float32)
        })
    else:
        context_features.update({
            'label': tf.FixedLenFeature([], dtype=tf.int64)
        })

    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features)

    example = {}
    example.update(context)
    example.update(sequence)

    if flattened:
        batch = tf.train.batch(
            example, batch_size,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
    else:
        batch = tf.train.batch(
            example, batch_size,
            allow_smaller_final_batch=True,
            shapes=[(), (), (num_steps), ()])
    return batch


class CNN(object):
    def __init__(self,
                 fn_queue,
                 batch_size,
                 num_steps,
                 num_labels,
                 num_filters,
                 emb_size,
                 vocab_size,
                 filter_sizes,
                 beta,
                 l2_cost,
                 keep_prob,
                 flattened,
                 initial_embs,
                 trainable_embs,
                 learning_rate,
                 max_to_keep,
                 dtype=tf.float32):

        batch = get_batch(
            fn_queue, num_steps, num_labels, batch_size, flattened)

        if flattened:
            self.labels = batch['labels']
            self.values = batch['values']
        else:
            self.label = batch['label']
        self.tokens = batch['tokens']
        self.weight = tf.cast(batch['weight'], dtype)

        with tf.device('/cpu:0'):
            if initial_embs is not None:
                embedding = tf.get_variable(
                    'embedding', dtype=dtype,
                    trainable=trainable_embs,
                    initializer=tf.constant(initial_embs))
            else:
                embedding = tf.get_variable(
                    'embedding', [vocab_size, emb_size],
                    dtype=dtype, trainable=trainable_embs)

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

        if flattened:
            targets = tf.one_hot(self.labels, num_labels, dtype=dtype)
            targets = tf.reduce_sum(
                targets * tf.expand_dims(self.values, -1), axis=1)
        else:
            targets = tf.one_hot(self.label, num_labels, dtype=dtype)

        self.probs = tf.nn.softmax(logits)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)

        if beta > 0:
            _entropy = -tf.reduce_sum(
                self.probs * tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)),
                reduction_indices=[1])
            xentropy = xentropy - beta * _entropy

        weighted = tf.mul(xentropy, self.weight)

        self.loss = tf.reduce_sum(weighted) / tf.reduce_sum(self.weight)

        if l2_cost > 0:
            self.loss += l2_cost * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)

        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=max_to_keep)
