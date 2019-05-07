import tensorflow as tf
from tensorflow.python.layers.core import Dense
import config

class next_number_network:
    def __init__(self, sequence_length, input_len, max_pi, max_out, IO_tool, char_number, learning_rate, lstm_units):
        if input_len == -1:
            lstm_len = max_pi
        else:
            lstm_len = sequence_length

        if max_pi >= 5000 and sequence_length== -1:
            self.Batch_size = 25
        else:
            self.Batch_size = config.Batch_size

        with tf.name_scope('Pi_Network'):

            with tf.name_scope("Input"):
                self.input_batch = tf.placeholder(tf.float32, shape=(self.Batch_size, lstm_len), name="Input")
                casted_input_batch = tf.cast(self.input_batch, tf.int32)
                input_batch= tf.one_hot(casted_input_batch, depth = char_number + 2)
                # self.input_batch = tf.placeholder(tf.float32, shape=(self.Batch_size, lstm_len, char_number + 2), name="Input")
                self.h_input = tf.placeholder(tf.float32, shape=(self.Batch_size, lstm_units), name="Previous_hidden_state")
                self.c_input = tf.placeholder(tf.float32, shape=(self.Batch_size, lstm_units), name="Previous_hidden_state")

            with tf.name_scope("Target"):
                self.next_label = tf.placeholder(tf.float32, shape=(self.Batch_size, max_out+1), name="Target")
                casted_target_embedded = tf.cast(self.next_label, tf.int32)
                dec_input = tf.concat([tf.fill([self.Batch_size, 1], IO_tool.dataset.vocab_to_id['<GO>']), casted_target_embedded], 1)
                one_hot_label= tf.one_hot(casted_target_embedded, depth = char_number + 2)
                dec_embeddings = tf.Variable(tf.random_uniform([char_number + 2, config.decoder_embedding_size]))
                dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

            with tf.name_scope('Input_Helper'):
                self.input_len = tf.placeholder(tf.int32, shape=(self.Batch_size), name="input_len")
                self.target_len = tf.placeholder(tf.int32, shape=(self.Batch_size), name="target_len")

            with tf.name_scope("Lstm_encoder"):
                encoder_cell = tf.contrib.rnn.LSTMCell(lstm_units, name='now_cell')
                # state = tf.contrib.rnn.LSTMStateTuple(self.c_input, self.h_input)
                if config.sequence_length == -1 :
                    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, input_batch,
                                                                      # initial_state=state,
                                                                      sequence_length = self.input_len,
                                                                      dtype=tf.float32)
                else:
                    state = tf.contrib.rnn.LSTMStateTuple(self.c_input, self.h_input)
                    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, input_batch,
                                                                      initial_state=state,
                                                                      sequence_length = self.input_len,
                                                                      dtype=tf.float32)


                # last = self.last_relevant(decoder_output, self.length(self.input_batch))
                # decoder_output = tf.expand_dims(decoder_output, axis = 1)
                self.c_out = encoder_state.c
                self.h_out = encoder_state.h

            with tf.name_scope('Lstm_cell'):
                # decoder_input = tf.tile(tf.expand_dims(last, 1), [1, max_out, 1])
                # decoder_cell = tf.contrib.rnn.LSTMCell(lstm_units, name='history_cell')
                # output, state = tf.nn.dynamic_rnn(decoder_cell, decoder_input,
                                                                  # dtype=tf.float32)
                                                                    # Build RNN cell
                output_layer = Dense(char_number + 2,
                                     kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

                decoder_cell = tf.contrib.rnn.LSTMCell(lstm_units,
                                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

            with tf.name_scope('Train_Lstm'):
                # Helper
                # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                #     self.next_label, self.start_token, char_number + 2-1)
                train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                           sequence_length=self.target_len)
                # Decoder
                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, train_helper, encoder_state, output_layer)

                # Dynamic decoding
                train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True)
                training_logit = tf.identity(train_output.rnn_output, 'logits')
                training_softmax = tf.nn.softmax(training_logit)
                training_predictions = tf.argmax(input=training_softmax, axis=2, name="classes")
                training_one_hot_prediction= tf.one_hot(training_predictions, depth = training_softmax.shape[-1])

            with tf.name_scope('Inference_Lstm'):
                # Helper
                # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                #     self.next_label, self.start_token, char_number + 2-1)
                # test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                #                                       tf.fill([self.Batch_size], IO_tool.dataset.vocab_to_id['<GO>']),
                #                                       IO_tool.dataset.vocab_to_id['<EOS>'])
                test_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=dec_embed_input,
                                                                                  sequence_length=self.target_len,
                                                                                  embedding=dec_embeddings,
                                                                                  sampling_probability=1.0)
                # Decoder
                test_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, test_helper, encoder_state, output_layer)

                # Dynamic decoding
                # print(max_out)
                test_output, _, _ = tf.contrib.seq2seq.dynamic_decode(test_decoder, impute_finished=True,
                                                                      maximum_iterations=max_out+1)
                testing_logit = tf.identity(test_output.rnn_output, 'logits')
                testing_softmax = tf.nn.softmax(testing_logit)
                testing_predictions = tf.argmax(input=testing_softmax, axis=2, name="classes")
                self.testing_id = testing_predictions
                testing_one_hot_prediction= tf.one_hot(testing_predictions, depth = testing_softmax.shape[-1])
            #
            # with tf.name_scope("Next_Number"):
            #     next_number_logit = tf.layers.dense(output.rnn_output, config.pre_class)
            #     next_number_logit = tf.layers.dense(next_number_logit, config.pre_class)
            #     self.next_number_logit = tf.layers.dense(next_number_logit, char_number + 2)
            #     self.next_number_softmax = tf.nn.softmax(self.next_number_logit)
            #     self.next_number_predictions = tf.argmax(input=self.next_number_softmax, axis=2, name="classes")
            #     # self.next_number_predictions = tf.Print(self.next_number_predictions, [tf.shape(self.next_number_softmax)])
            #     # self.next_number_predictions = tf.Print(self.next_number_predictions, [self.next_number_predictions], first_n=1000)
            #     self.next_number_one_hot_prediction= tf.one_hot(self.next_number_predictions, depth = self.next_number_softmax.shape[-1])

            with tf.name_scope('Loaders_and_Savers'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.variables = tf.contrib.slim.get_variables_to_restore()
                with tf.name_scope('Model_Saver'):
                    self.model_saver = tf.train.Saver()

            with tf.name_scope('Metrics_calculation'):
                train_next_number_precision, self.train_next_number_recall, train_next_number_f1, train_next_number_accuracy = self.accuracy_metrics(training_one_hot_prediction[:,1:-1], one_hot_label[:,1:-1])
                test_next_number_precision, self.test_next_number_recall, test_next_number_f1, test_next_number_accuracy = self.accuracy_metrics(testing_one_hot_prediction[:,1:-1], one_hot_label[:,1:-1])

            with tf.name_scope("next_number_Loss"):
                masks = tf.sequence_mask(self.target_len, max_out+1, dtype=tf.float32, name='masks')
                total_loss = tf.contrib.seq2seq.sequence_loss(training_logit,casted_target_embedded,masks)
                self.total_loss = total_loss
                # print(training_logit)
                # print(testing_logit)
                # print(casted_target_embedded)
                # print(masks)
                test_total_loss = tf.contrib.seq2seq.sequence_loss(testing_logit,casted_target_embedded,masks)
                # test_total_loss = total_loss
                self.test_total_loss = test_total_loss

            with tf.name_scope("Optimizer"):
                Train_variable = [v for v in self.variables if 'Openpose' not in v.name.split('/')[0]]
                Train_variable = [v for v in Train_variable if 'MobilenetV1' not in v.name.split('/')[0]]

                learning_rate = learning_rate
                # learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.9)
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=total_loss,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam',
                    clip_gradients=config.gradient_clipping_norm,
                    summaries = ["learning_rate"],
                    variables=Train_variable)

            with tf.name_scope('summary'):
                tf.summary.histogram("train_next_number_classification", training_predictions)
                tf.summary.histogram("next_number_classification",testing_predictions)
                tf.summary.histogram("next_number_target", self.next_label)
                tf.summary.scalar("Loss", total_loss)
                tf.summary.scalar("Inference_Loss", test_total_loss)
                # tf.summary.scalar("learning_rate", learning_rate)
                # tf.summary.scalar("Precision", next_number_precision)
                tf.summary.scalar("train_recall", self.train_next_number_recall)
                tf.summary.scalar("inference_recall", self.test_next_number_recall)
                # tf.summary.scalar("f1", next_number_f1)
                # tf.summary.scalar("accuracy", next_number_accuracy)
                self.merged = tf.summary.merge_all()

            with tf.name_scope("Initializer"):
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                self.init = tf.group(init_local, init_global)

    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


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
            accuracy = (TP + TN) / (TP + FP + TN + FN)
        return precision, recall, f1, accuracy
