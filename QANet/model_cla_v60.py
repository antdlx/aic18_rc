import tensorflow as tf
from QANet.layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention

class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True,wrong=False, opt=False, demo = False, graph = None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c = tf.placeholder(tf.int64, [None, config.test_para_limit],"context")
                self.q = tf.placeholder(tf.int64, [None, config.test_ques_limit],"question")
                self.ch = tf.placeholder(tf.int64, [None, config.test_para_limit, config.char_limit],"context_char")
                self.qh = tf.placeholder(tf.int64, [None, config.test_ques_limit, config.char_limit],"question_char")
                self.ans = tf.placeholder(tf.int64, [None, config.test_para_limit],"answer_index")
            else:
                if trainable or wrong:
                    self.c, self.ch, self.q, self.qh, self.ans,self.qa_id = batch.get_next()
                else:
                    self.c, self.ch, self.q, self.qh,self.id= batch.get_next()


            # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32),trainable=False)
            # self.word_mat = tf.get_variable("word_mat",[len(word_mat),300],initializer=tf.constant_initializer(
            #     word_mat, dtype=tf.float32), trainable=False)
            # self.char_mat = tf.get_variable(
            #     "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
            self.char_mat = self.word_mat
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)

                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                # self.ans = tf.slice(self.ans, [0, 0], [N, self.c_maxlen])

            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward(trainable)
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                # self.lr = tf.train.cosine_decay_restarts(config.learning_rate, self.global_step, config.num_steps)
                # self.lr = tf.maximum(self.lr, config.end_learning_rate)
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self,trainable):
        config = self.config
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
        if not trainable:
            N = 1

        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

			# Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 2, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 2, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, d, name = "input_projection")]
            for i in range(2):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )

        with tf.variable_scope("Output_Layer"):
            concat = tf.concat([self.enc[1], self.enc[2]],axis = -1)
            # concat = conv(concat, 1, bias=False, name="start_pointer")
            # concat = tf.squeeze(concat)
            after_conv = tf.layers.conv1d(
                inputs=concat,
                filters=32,
                kernel_size=5,
                padding="same",
                activation=tf.nn.relu
            )
            after_pool = tf.layers.max_pooling1d(inputs=after_conv,pool_size=4,strides=1)
            pool_flat = tf.reshape(after_pool, [-1, 97 * 32])

            dense = tf.layers.dense(inputs=pool_flat,units=3,use_bias=False)
            # after_dropout = tf.layers.dropout(inputs=dense, rate=1.0 - self.dropout)
            # dense2 = tf.layers.dense(inputs=after_dropout,units=3,activation=tf.nn.relu)
            self.logits = dense

            # ans_logist = conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "ans")
            # ans_logist = tf.squeeze(ans_logist, -1)
            # ans_logist =  tf.nn.softmax(ans_logist)
            # ans_logist3 = tf.layers.dense(ans_logist,3,activation=tf.nn.relu)
            # self.logits = ans_logist3

            outer = tf.nn.softmax(self.logits)
            self.ansp = tf.argmax(outer, axis=1)

            if trainable:
                self.outer = outer

                # losses = tf.nn.softmax_cross_entropy_with_logits(
                #     logits=dense2, labels=self.ans)
                # self.loss = tf.reduce_mean(losses)
                losses = tf.losses.softmax_cross_entropy(self.ans, self.logits)
                self.loss = losses

                self.acc = tf.reduce_mean(tf.cast(tf.equal(self.ansp, tf.argmax(self.ans, 1)), tf.float32))

                if config.l2_norm is not None:
                    variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                    self.loss += l2_loss

                if config.decay is not None:
                    self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
                    ema_op = self.var_ema.apply(tf.trainable_variables())
                    with tf.control_dependencies([ema_op]):
                        self.loss = tf.identity(self.loss)

                        self.assign_vars = []
                        for var in tf.global_variables():
                            v = self.var_ema.average(var)
                            if v:
                                self.assign_vars.append(tf.assign(var,v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
