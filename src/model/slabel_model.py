import tensorflow as tf

from .basic_model import BasicModel

class SlabelModel(BasicModel):

    def __init__ (self, config):
        BasicModel.__init__(self, config)


    def _add_placeholders(self):
        with tf.variable_scope('placeholders'):
            """Inputs to be fed to the graph."""
            #shape = (batch_size, max length of sentence, max lenght of word)
            self.chars_in = tf.placeholder(tf.int32, [None, None], 'char-in')
            self.chars_len = tf.placeholder(tf.int32, [None], 'char-seq-len')
            #shape = (batch_size, max length of sentence)
            self.words_in = tf.placeholder(tf.int32, [None, None], 'words-input')
            #shape = (batch_size)
            self.words_len = tf.placeholder(tf.int32, [None], 'word-seq-len')
            #shape = (batch_size, max length of sentence)
            self.tags_in = tf.placeholder(tf.int32, [None, None], 'tag-input')
            #shape = (batch_size * max length of sentence, max length of label)
            self.labels_in = tf.placeholder(tf.int32, [None, None], 'label-input')
            #shape = (batch_size * max length of sentence)
            self.labels_len = tf.placeholder(tf.int32, [None], 'label-seq-len')
            #shape = (batch_size * length of sentences)
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            #dropout rate
            self.is_train = tf.placeholder(tf.bool, shape=(), name='is-train')

    def _add_char_lstm(self):
        with tf.variable_scope('char-LSTM-Layer', initializer=self.initializer):

            char_embed = self.embedding(
                                self.chars_in,
                                [self.nchars, self.char_dim],
                                self.dropouts[1],
                                self.is_train,
                                names=['char-embed','char-embed-dropout'])

            char_cell = self._single_cell(
                                    self.h_char,
                                    self.dropouts[1],
                                    self.is_train)

            _, self.ch_state = tf.nn.dynamic_rnn(char_cell,
                                            char_embed,
                                            sequence_length=self.chars_len,
                                            dtype=self.dtype,
                                            scope='char-lstm')



    def _add_word_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.variable_scope('word-LSTM-Layer'):
            # Forward and Backward direction cell
            word_embed = self.embedding(
                                    self.words_in,
                                    [self.nwords, self.word_dim],
                                    self.dropouts[1],
                                    self.is_train,
                                    names=['word-embed','word-embed-dropout'])

            tag_embed = self.embedding(
                                    self.tags_in,
                                    [self.ntags, self.tag_dim],
                                    self.dropouts[1],
                                    self.is_train,
                                    names=['tag-embed','tag-embed-dropout'])

            word_cell_fw = self._multi_cell(self.h_word,
                                            tf.constant(self.dropouts[0]),
                                            self.is_train,
                                            self.n_layers)

            word_cell_bw = self._multi_cell(self.h_word,
                                            tf.constant(self.dropouts[0]),
                                            self.is_train,
                                            self.n_layers)


            char_out_shape = [tf.shape(tag_embed)[0], -1, self.h_char]
            char_out = tf.reshape(self.ch_state[1], char_out_shape)
            w_bidi_in = tf.concat([word_embed, tag_embed, char_out], -1,
                                        name='word-bidi-in')

            # Get lstm cell output
            w_bidi_out , _ = tf.nn.bidirectional_dynamic_rnn(
                                tf.contrib.rnn.MultiRNNCell(word_cell_fw),
                                tf.contrib.rnn.MultiRNNCell(word_cell_bw),
                                w_bidi_in,
                                sequence_length=self.words_len,
                                dtype=self.dtype)
            w_bidi_out_c = tf.concat(w_bidi_out , -1, name='word-bidi-out')

            encode_state = tf.concat([w_bidi_in, w_bidi_out_c], -1)

            self.encode_state = tf.layers.dense(encode_state, self.h_label)

    def _add_label_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.variable_scope('tag-LSTM-Layer'):
            label_embed = self.embedding(
                                self.labels_in,
                                [self.nlabels, self.label_dim],
                                self.dropouts[1],
                                self.is_train,
                                names=['label-embed','label-embed-dropout'])

            dec_init_state = tf.reshape(self.encode_state, [-1, self.h_label])

            self.label_init = tf.contrib.rnn.LSTMStateTuple(
                                        dec_init_state,
                                        tf.zeros_like(dec_init_state))

            label_cell = self._single_cell(
                                    self.h_label,
                                    self.dropouts[1],
                                    self.is_train)

            self.decode_out, self.decode_state = tf.nn.dynamic_rnn(
                                                label_cell,
                                                label_embed,
                                                initial_state=self.label_init,
                                                sequence_length=self.labels_len,
                                                dtype=self.dtype)

    def _add_attention(self):
        with tf.variable_scope('Attention'):
            es_shape = tf.shape(self.encode_state)[0]
            k = tf.layers.dense(self.decode_out, self.attention_dim)
            atten_k = tf.reshape(k, [es_shape, -1, self.attention_dim])

            atten_q = tf.layers.dense(
                                self.encode_state,
                                self.attention_dim,
                                activation=self.activation_fn)

            alpha = tf.nn.softmax(tf.einsum('aij,akj->aik', atten_k, atten_q))
            context = tf.einsum('aij,ajk->aik', alpha, self.encode_state)
            context = tf.reshape(context, tf.shape(self.decode_out))

            self.attention = tf.concat([self.decode_out, context], -1)

    def _add_projection(self):
        with tf.variable_scope('probabilities'):

            logits = tf.layers.dense(
                            self.attention,
                            self.projection_dim,
                            activation=self.activation_fn)

            mask_t = tf.sequence_mask(self.labels_len, dtype=tf.int32)
            v = tf.dynamic_partition(logits, mask_t, 2)[1]

            self.logits = tf.layers.dense(v, self.nlabels)
            # compute softmax
            self.probs = tf.nn.softmax(self.logits, name='probs')

    def _add_loss(self):

        with tf.variable_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.nlabels)

            self.loss = tf.losses.softmax_cross_entropy(
                                logits=self.logits,
                                onehot_labels=targets_1hot,
                                reduction=tf.losses.Reduction.MEAN)

    def _add_train_op(self):
        self.global_step = tf.Variable(
                                initial_value=0,
                                trainable=False,
                                dtype=tf.int32,
                                name='g_step')

        self.epoch = tf.Variable(
                            initial_value=0,
                            trainable=False,
                            dtype=tf.int32,
                            name='epoch')

        self.optimizer = self.optimizer_fn().minimize(
                                                self.loss,
                                                global_step=self.global_step)

    def build_graph(self):
        with tf.Graph().as_default() as g:
            # with tf.device('/gpu:{}'.format(self.gpu_id)):
            with tf.variable_scope('slabel', initializer=self.initializer, dtype=self.dtype):
                self._add_placeholders()
                self._add_char_lstm()
                self._add_word_bidi_lstm()
                self._add_label_lstm_layer()
                self._add_attention()
                self._add_projection()
                self._add_loss()
                self._add_train_op()
        return g


        """"TRAIN Part """
    def step(self, batch, output_feed, is_train=False):
        """ Training step, returns the loss"""
        input_feed = {
            self.words_in: batch.words.input,
            self.words_len: batch.words.length,
            self.chars_in : batch.chars.input,
            self.chars_len : batch.chars.length,
            self.tags_in : batch.tags.input,
            self.labels_in: batch.labels.input,
            self.labels_len: batch.labels.length,
            self.targets: batch.targets,
            self.is_train : is_train}
        return self.sess.run(output_feed, input_feed)

    """"Decode Part """
    def encode_top_state(self, enc_bv):
        """Return the top states from encoder for decoder."""
        input_feed = {self.words_in: enc_bv.words.input,
                        self.words_len: enc_bv.words.length,
                        self.chars_in : enc_bv.chars.input,
                        self.chars_len : enc_bv.chars.length,
                        self.tags_in: enc_bv.tags.input,
                        self.is_train : False}
        return self.sess.run(self.encode_state, input_feed)

    def decode_topk(self, latest_tokens, dec_init_states, enc_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.label_init : dec_init_states,
            self.labels_in: np.array(latest_tokens),
            self.encode_state : enc_state,
            self.labels_len: np.ones(len(latest_tokens), np.int32),
            self.is_train : False}
        output_feed = [self.decode_state, self.probs]
        states, probs = self.sess.run(output_feed, input_feed)
        topk_ids = np.array([np.argsort(np.squeeze(p))[-k:] for p in probs])
        topk_probs = np.array([p[k_id] for p,k_id in zip(probs,topk_ids)])
        return topk_ids, topk_probs, states
