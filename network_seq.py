import tensorflow as tf
import config
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



class Input_manager:
    def __init__(self, devices, IO_tool):

        with tf.name_scope('Inputs'):

            with tf.name_scope("Input"):
                self.input_batch = tf.placeholder(tf.uint8, shape=(None, config.Batch_size, config.seq_len, config.frames_per_step, config.out_H, config.out_W, config.input_channels), name="Input")
                self.h_input = tf.placeholder(tf.float32, shape=(None, config.Batch_size, config.lstm_units), name="Previous_hidden_state")
                self.c_input = tf.placeholder(tf.float32, shape=(None, config.Batch_size, config.lstm_units), name="Previous_hidden_state")

            with tf.name_scope("Now_target"):
                self.labels = tf.placeholder(tf.int32, shape=(None, config.Batch_size, config.seq_len), name="Target")
                self.next_labels = tf.placeholder(tf.int32, shape=(None, config.Batch_size), name="next_labels")
                self.dec_embeddings = tf.Variable(tf.random_uniform([len(IO_tool.dataset.label_to_id), config.decoder_embedding_size]))

class activity_network:
    def __init__(self, number_of_classes, number_of_activities, Input_manager, device_j, IO_tool):
        self.number_of_classes = number_of_classes
        self.number_of_activities = number_of_activities
        self.out_vocab_size = len(IO_tool.dataset.label_to_id)

        with tf.name_scope('Activity_Recognition_Network'):

            with tf.variable_scope('var_name'):
                wc = {
                    'wc1': self._variable_with_weight_decay('wc1', [3, 3, 3, config.input_channels, 64], 0.04, 0.00),
                    'wc2': self._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                    'wc3a': self._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                    'wc3b': self._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                    'wc4a': self._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                    'wc4b': self._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                    'wc5a': self._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                    'wc5b': self._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00)}
                bc = {
                    'bc1': self._variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                    'bc2': self._variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                    'bc3a': self._variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                    'bc3b': self._variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                    'bc4a': self._variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                    'bc4b': self._variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                    'bc5a': self._variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                    'bc5b': self._variable_with_weight_decay('bc5b', [512], 0.04, 0.0)}

            with tf.name_scope("Input"):
                self.input_batch = tf.squeeze(Input_manager.input_batch[device_j, :, :, :, :, :])
                self.input_batch = tf.cast(self.input_batch, tf.float32)
                self.h_input = tf.squeeze(Input_manager.h_input[device_j, :, :])
                self.c_input = tf.squeeze(Input_manager.c_input[device_j, :, :])

            with tf.name_scope("Now_Target"):
                self.labels = tf.squeeze(Input_manager.labels[device_j, :, :])
                now_dec_input = tf.concat([tf.fill([config.Batch_size, 1], IO_tool.dataset.label_to_id['go']), self.labels], 1)
                self.now_one_hot_label= tf.one_hot(self.labels, depth = self.out_vocab_size)
                now_dec_embed_input = tf.nn.embedding_lookup(Input_manager.dec_embeddings, now_dec_input)
                now_target_len = tf.ones(shape=(config.Batch_size), dtype=tf.int32)*config.seq_len

            
            with tf.name_scope("Next_Target"):
                self.next_labels = tf.squeeze(Input_manager.next_labels[device_j, :])
                self.next_one_hot_label= tf.one_hot(self.next_labels, depth = self.out_vocab_size)
                
            def C3d(Tensor):
                with tf.name_scope('C3d'):
                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv1 = self.conv3d('conv1', Tensor, wc['wc1'], bc['bc1'])
                        conv1 = tf.nn.leaky_relu(conv1, name='relu1')
                        pool1 = self.max_pool('pool1', conv1, k=1)

                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv2 = self.conv3d('conv2', pool1, wc['wc2'], bc['bc2'])
                        conv2 = tf.nn.leaky_relu(conv2, name='relu2')
                        pool2 = self.max_pool('pool2', conv2, k=2)

                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv3 = self.conv3d('conv3a', pool2, wc['wc3a'], bc['bc3a'])
                        conv3 = tf.nn.leaky_relu(conv3, name='relu3a')
                        conv3 = self.conv3d('conv3b', conv3, wc['wc3b'], bc['bc3b'])
                        conv3 = tf.nn.leaky_relu(conv3, name='relu3b')
                        pool3 = self.max_pool('pool3', conv3, k=2)

                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv4 = self.conv3d('conv4a', pool3, wc['wc4a'], bc['bc4a'])
                        conv4 = tf.nn.leaky_relu(conv4, name='relu4a')
                        conv4 = self.conv3d('conv4b', conv4, wc['wc4b'], bc['bc4b'])
                        conv4 = tf.nn.leaky_relu(conv4, name='relu4b')
                        pool4 = self.max_pool('pool4', conv4, k=2)

                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv5 = self.conv3d('conv5a', pool4, wc['wc5a'], bc['bc5a'])
                        conv5 = tf.nn.leaky_relu(conv5, name='relu5a')
                        conv5 = self.conv3d('conv5b', conv5, wc['wc5b'], bc['bc5b'])
                        conv5 = tf.nn.leaky_relu(conv5, name='relu5b')
                        pool5 = self.max_pool('pool5', conv5, k=2)

                    with tf.name_scope('reshape_c3d'):
                        reshape_1_cd = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
                        reshape_2_cd = tf.contrib.layers.flatten(reshape_1_cd)
                    
                    return reshape_2_cd

            with tf.name_scope('c3d_mapfn'):
                self.c3d_out = tf.map_fn(lambda x: C3d(x), self.input_batch)

            with tf.name_scope("Dimension_Encoder"):
                dense1_cd = tf.layers.dense(self.c3d_out, config.enc_fc_1)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_2)
                self.out_pL = tf.layers.dense(dense2_cd, config.lstm_units)
                exp_out_pL = tf.expand_dims(self.out_pL, 1)

            with tf.name_scope("Dimension_Decoder"):
                dense1_cd = tf.layers.dense(self.out_pL, config.enc_fc_2)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_1)
                self.autoenc_out = tf.layers.dense(dense2_cd, self.c3d_out.shape[-1])

            with tf.name_scope("Lstm_encoder"):
                encoder_cell = tf.contrib.rnn.LSTMCell(config.lstm_units, name='now_cell')
                state = tf.contrib.rnn.LSTMStateTuple(self.c_input, self.h_input)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.out_pL,
                                                                    initial_state=state,
                                                                    dtype=tf.float32)
                self.c_out = encoder_state.c
                self.h_out = encoder_state.h
   
            def decoder_lstm(dec_lstm_units):
                output_layer = tf.layers.Dense(self.out_vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
                decoder_cell = tf.contrib.rnn.LSTMCell(dec_lstm_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return decoder_cell, output_layer

            def train_lstm(encoder_state, decoder_cell, output_layer, dec_embed_input, target_len):
                train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=target_len)
                # Decoder
                train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_state, output_layer)
                # Dynamic decoding
                train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True)
                training_logit = tf.identity(train_output.rnn_output, 'logits')
                training_softmax = tf.nn.softmax(training_logit)
                training_predictions = tf.argmax(input=training_softmax, axis=2, name="classes")
                training_one_hot_prediction= tf.one_hot(training_predictions, depth = training_softmax.shape[-1])
                return training_logit, training_softmax, training_predictions, training_one_hot_prediction

            with tf.name_scope('Now_Decoder'):
                now_decoder, now_output_layer = decoder_lstm(config.lstm_units)
                self.now_logit, self.now_softmax, self.now_predictions, self.now_one_hot_prediction = train_lstm(encoder_state, now_decoder, now_output_layer, now_dec_embed_input, now_target_len)

            with tf.name_scope('Next_classifier'):
                self.now_softmax.set_shape([None, config.seq_len, self.out_vocab_size])
                flat_now = tf.contrib.layers.flatten(self.now_softmax)
                C_composedVec = tf.concat([encoder_state.c, flat_now], 1)
                H_composedVec = tf.concat([encoder_state.h, flat_now], 1)
                new_C = tf.layers.dense(C_composedVec, config.lstm_units)
                new_H = tf.layers.dense(H_composedVec, config.lstm_units)
                H_composedVec = tf.concat([new_C, new_H], 1)

                next_dense_1 = tf.layers.dense(H_composedVec, config.pre_class)
                self.next_logit = tf.layers.dense(next_dense_1, self.number_of_classes)
                self.next_softmax = tf.nn.softmax(self.next_logit)
                self.next_predictions = tf.argmax(input=self.next_softmax, axis=1, name="c3d_prediction")
                self.next_one_hot_prediction= tf.one_hot(self.next_predictions, depth = self.next_softmax.shape[-1])
            

            with tf.name_scope("c3d_classifier"):
                self.out_cd = tf.layers.dense(self.c3d_out, config.pre_class)
                self.out_cd = tf.layers.dense(self.out_cd, self.number_of_classes)
                self.softmax_c3d = tf.nn.softmax(self.out_cd)
                self.predictions_c3d = tf.argmax(input=self.softmax_c3d, axis=2, name="c3d_prediction")
                self.c3d_one_hot_prediction= tf.one_hot(self.predictions_c3d, depth = self.softmax_c3d.shape[-1])

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv3d(self, name, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
            b)

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

class Training:
    def __init__(self,Networks):
        with tf.name_scope('Training_and_Metrics'):
            with tf.name_scope('Loaders_and_Savers'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.variables = tf.contrib.slim.get_variables_to_restore()
                with tf.name_scope('Model_Saver'):
                    self.model_saver = tf.train.Saver()
                with tf.name_scope('C3D_Loaders'):
                    Load = ['wc2', 'wc3a', 'wc3b', 'wc4a', 'wc4b', 'wc5a', 'wc5b', 'bc1',
                            'bc2', 'bc3a', 'bc3b', 'bc4a', 'bc4b', 'bc5a', 'bc5b']
                    c3d_loader_variables = [v for v in self.variables if 'Network/var_name' in v.name.split(':')[0] and v.name.split(':')[0].split('/')[-1] in Load]
                    name_to_vars = {''.join(v.op.name.split('Network/')[0:]): v for v in c3d_loader_variables}
                    self.c3d_loader = tf.train.Saver(name_to_vars)

            with tf.name_scope('Metrics'):
                z = 0
                for Net in Networks:
                    if z == 0:
                        c3d_pred_conc = Networks[Net].c3d_one_hot_prediction
                        now_pred_conc = Networks[Net].now_one_hot_prediction
                        next_pred_conc = Networks[Net].next_one_hot_prediction
                        now_label_conc = Networks[Net].now_one_hot_label
                        next_label_conc = Networks[Net].next_one_hot_label
                        z +=1
                    else:
                        c3d_pred_conc = tf.concat([c3d_pred_conc, Networks[Net].c3d_one_hot_prediction], axis=0)
                        now_pred_conc = tf.concat([now_pred_conc, Networks[Net].now_one_hot_prediction], axis=0)
                        next_pred_conc = tf.concat([next_pred_conc, Networks[Net].next_one_hot_prediction], axis=0)
                        now_label_conc = tf.concat([now_label_conc, Networks[Net].now_one_hot_label], axis=0)
                        next_label_conc = tf.concat([next_label_conc, Networks[Net].next_one_hot_label], axis=0)

                with tf.name_scope('Metrics_calculation'):
                    c3d_precision, c3d_recall, c3d_f1, c3d_accuracy = self.accuracy_metrics(c3d_pred_conc, now_label_conc)
                    now_precision, now_recall, now_f1, now_accuracy = self.accuracy_metrics(now_pred_conc, now_label_conc)
                    next_precision, next_recall, next_f1, next_accuracy = self.accuracy_metrics(next_pred_conc, next_label_conc)

            with tf.name_scope('Loss'):
                for Net in Networks:
                    z = 0
                    with tf.name_scope(Net):
                        with tf.name_scope("C3d_Loss"):
                            cross_entropy_c3d_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].now_one_hot_label, logits=Networks[Net].out_cd)
                            c3d_loss = tf.reduce_sum(cross_entropy_c3d_vec)

                        with tf.name_scope("Now_Loss"):
                            cross_entropy_Now_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].now_one_hot_label, logits=Networks[Net].now_logit)
                            now_loss = tf.reduce_sum(cross_entropy_Now_vec)

                        with tf.name_scope("Next_Loss"):
                            cross_entropy_Next_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].next_one_hot_label, logits=Networks[Net].next_logit)
                            next_loss = tf.reduce_sum(cross_entropy_Next_vec)

                        with tf.name_scope("Autoencoder_Loss"):
                            auto_enc_loss=tf.reduce_mean(tf.square(Networks[Net].autoenc_out-Networks[Net].c3d_out))

                        if z == 0:
                            c3d_loss_sum = c3d_loss
                            now_loss_sum = now_loss
                            next_loss_sum = next_loss
                            auto_enc_loss_sum = auto_enc_loss
                            z += 1
                        else:
                            c3d_loss_sum += c3d_loss
                            now_loss_sum += now_loss
                            next_loss_sum += next_loss
                            auto_enc_loss_sum += auto_enc_loss

                with tf.name_scope("Global_Loss"):
                    c3d_loss_sum = tf.cast(c3d_loss_sum, tf.float64)
                    now_loss_sum = tf.cast(now_loss_sum, tf.float64)
                    next_loss_sum = tf.cast(next_loss_sum, tf.float64)
                    auto_enc_loss_sum = tf.cast(auto_enc_loss_sum, tf.float64)
                    total_loss = (c3d_recall)*(now_recall*next_loss_sum + (1-now_recall)*now_loss_sum) + (1 - c3d_recall) * c3d_loss_sum + auto_enc_loss_sum

            with tf.name_scope("Optimizer"):
                Train_variable = [v for v in self.variables if 'Openpose' not in v.name.split('/')[0]]
                Train_variable = [v for v in Train_variable if 'MobilenetV1' not in v.name.split('/')[0]]

                starter_learning_rate = 0.0001
                learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                            1000, 0.9)
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=total_loss,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam',
                    clip_gradients=config.gradient_clipping_norm,
                    variables=Train_variable)

            with tf.name_scope('Summary'):
                j = 0
                for Net in Networks:
                    if j == 0:
                        conc_predictions_c3d = Networks[Net].predictions_c3d
                        conc_predictions_now = Networks[Net].now_predictions
                        conc_predictions_next = Networks[Net].next_predictions
                        conc_now_labels = Networks[Net].now_one_hot_label
                        conc_next_labels = Networks[Net].next_one_hot_label
                    else:
                        conc_predictions_c3d = tf.concat([conc_predictions_c3d,Networks[Net].predictions_c3d], axis=0)
                        conc_predictions_now = tf.concat([conc_predictions_now,Networks[Net].now_predictions], axis=0)
                        conc_predictions_next = tf.concat([conc_predictions_next,Networks[Net].next_predictions], axis=0)
                        conc_now_labels = tf.concat([conc_now_labels,Networks[Net].now_one_hot_label], axis=0)
                        conc_next_labels = tf.concat([conc_next_labels,Networks[Net].next_one_hot_label], axis=0)
                    j += 1


                # tf.summary.histogram("c_out", self.c_out)
                # tf.summary.histogram("h_out", self.h_out)
                # tf.summary.histogram("c_in", self.c_input)
                # tf.summary.histogram("h_in", self.h_input)
                # # tf.summary.histogram("labels_target", argmax_labels)
                tf.summary.histogram("Now_classification", conc_predictions_now)
                tf.summary.histogram("Next_classification", conc_predictions_next)
                tf.summary.histogram("c3d_classification", conc_predictions_c3d)
                tf.summary.histogram("Now_label", tf.argmax(input=now_label_conc, axis=-1))
                tf.summary.histogram("Next_label", tf.argmax(input=next_label_conc, axis=-1))
                tf.summary.scalar('Lstm_loss', now_loss_sum)
                tf.summary.scalar('auto_enc_loss_sum', auto_enc_loss_sum)
                tf.summary.scalar('C3d_Loss', c3d_loss_sum)
                tf.summary.scalar('Now_Loss', now_loss_sum)
                tf.summary.scalar('Next_Loss', next_loss_sum)
                tf.summary.scalar('now_recall', now_recall)
                tf.summary.scalar('next_recall', next_recall)
                tf.summary.scalar('c3d_recall', c3d_recall)
                self.merged = tf.summary.merge_all()

            with tf.name_scope('Outputs'):
                self.predictions_now = []
                self.predictions_next = []
                self.predictions_c3d = []
                self.c_out_list = []
                self.h_out_list = []
                for Net in Networks:
                    self.predictions_now.append(Networks[Net].now_predictions)
                    self.predictions_next.append(Networks[Net].next_predictions)
                    self.predictions_c3d.append(Networks[Net].predictions_c3d)
                    self.c_out_list.append(Networks[Net].c_out)
                    self.h_out_list.append(Networks[Net].h_out)

            with tf.name_scope("Initializer"):
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                self.init = tf.group(init_local, init_global)
            debug_writer = tf.summary.FileWriter("debug", tf.get_default_graph())


    def accuracy_metrics(self, predicted, actual):
        with tf.name_scope('accuracy_metrics'):
            predicted = tf.cast(predicted, tf.int64)
            actual = tf.cast(actual, tf.int64)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # accuracy = (TP) / (TP + FP + TN + FN)
            # accuracy = (TP) / (TP + FP + TN + FN)
            accuracy = (TP + TN) / (TP + FP + TN + FN)
        return precision, recall, f1, accuracy
