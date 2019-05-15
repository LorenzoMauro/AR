#[TENSORFLOW]
TF_CPP_MIN_LOG_LEVEL = '3'
inter_op_parallelism_threads=7
allow_growth = True

#[Train]
c3d_ucf_weights = "sports1m_finetuning_ucf101.model"
Batch_size = 2
frames_per_step = 6
window_size = 1
load_previous_weigth = False
load_pretrained_weigth = False
model_filename = './checkpoint/Net_weigths.model'
tot_steps = 1000000
processes = 12
tasks = 10
reuse_HSM = True
reuse_output_collection = False
Action = True
tree_or_graph = "graph"
balance_key = 'all'
is_ordered = False

#[Network]
learning_rate = 0.01
gradient_clipping_norm = 1.0
c3d_dropout = 0.6
preLstm_dropout = 0.6
Lstm_dropout = 0.6
input_channels = 7
enc_fc_1 = 4000
enc_fc_2 = int(enc_fc_1 / 2)
lstm_units = int(enc_fc_2 / 2)
pre_class = int(lstm_units / 2)
encoder_lstm_layers = 3*[lstm_units]
matrix_attention = False
decoder_embedding_size = 20

#[Batch]
out_H = 112
out_W = 112
hidden_states_dim = lstm_units
current_accuracy = 0
snow_ball = False
snow_ball_step_count = 0
snow_ball_per_class = 2000
snow_ball_classes = 3
op_input_width = 368
op_input_height = 368
show_pic = False
seq_len = 4

#[Dataset]
validation_fraction = 0.2
split_seconds = True

#[Annotation]
rebuild = False
limit_classes = True
classes_to_use = ['milk', 'coffee'] # ['friedegg', 'cereals', 'milk']
ocado_annotation = 'dataset/ocado.json'
kit_annotation = 'dataset/kit.json'
ocado_path = 'dataset/Video/Ocado'
kit_path = 'dataset/Video/kit_dataset'
breakfast_annotation = 'dataset/breakfast.json'
acitivityNet_annotation = 'dataset/activity_net.v1-3.min.json'
breakfast_path = 'dataset/Video/BreakfastII_15fps_qvga_sync'
iccv_path = 'dataset/Video/activitynet_selected'
iccv_json = ['Long jump.json', 'Triple jump.json']
iccv__annotation = breakfast_annotation = 'dataset/iccv.json'
dataset = 'Ocado'
breakfast_fps = 15
