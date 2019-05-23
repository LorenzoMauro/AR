# deploy
out_H = 112
out_W = out_H
op_input_width = 368
op_input_height = op_input_width
frames_per_step = 6
input_channels = 7
seq_len = 4
enc_fc_1 = 4000
enc_fc_2 = int(enc_fc_1 / 2)
lstm_units = int(enc_fc_2 / 2)
pre_class = int(lstm_units / 2)
encoder_lstm_layers = 3*[lstm_units]
hidden_states_dim = lstm_units
