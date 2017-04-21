import tensorflow as tf


def get_batch(fn_queue, num_steps, batch_size):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(fn_queue)

    context_features = {
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'length': tf.FixedLenFeature([], dtype=tf.int64),
        'weight': tf.FixedLenFeature([], dtype=tf.float32),
    }
    sequence_features = {
        'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features)

    example = {}
    example.update(context)
    example.update(sequence)

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
                 l2_cost,
                 keep_prob,
                 learning_rate,
                 dtype=tf.float32):

        batch = get_batch(fn_queue, num_steps, batch_size)

        self.label = batch['label']
        self.tokens = batch['tokens']
        self.weight = tf.cast(batch['weight'], dtype)

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
        targets = tf.one_hot(self.label, num_labels, dtype=dtype)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
        weighted = tf.mul(xentropy, self.weight)

        self.loss = tf.reduce_sum(weighted) / tf.reduce_sum(self.weight)

        if l2_cost > 0:
            self.loss += l2_cost * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)

        self.probs = tf.nn.softmax(logits)
        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(tf.global_variables())
