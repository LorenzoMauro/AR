#[TENSORFLOW]
TF_CPP_MIN_LOG_LEVEL = '3'
inter_op_parallelism_threads=7
allow_growth = True

#[Train]
c3d_ucf_weights = "sports1m_finetuning_ucf101.model"
Batch_size = 10
frames_per_step = 6
window_size = 1
load_previous_weigth = True
load_pretrained_weigth = True
load_c3d = False
model_filename = './checkpoint/Net_weigths.model'
deploy_folder = './Help-System/model/activity_network_model'
deploy_folder = './Help-System/model/activity_network_model.ckpt'
tot_steps = 1000000
processes = 20
tasks = 40
val_task = 40
reuse_HSM = True
reuse_output_collection = False
Action = True
tree_or_graph = "graph"
balance_key = 'all'
is_ordered = False

#[Network]
learning_rate_start = 0.00001
gradient_clipping_norm = 1.0
plh_dropout = 0.5
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
validation_fraction = 0.15
split_seconds = True

#[Annotation]
rebuild = False
limit_classes = True
use_prep = True
classes_to_use = ['milk', 'coffee'] # ['friedegg', 'cereals', 'milk']
ocado_annotation = 'dataset/ocado.json'
kit_activity_annotation = 'dataset/kit_activity.json'
kit_help_annotation_temp = 'dataset/kit_help_temp.json'
kit_help_annotation = 'dataset/kit_help.json'
ocado_path = 'dataset/Video1'
kit_path = 'dataset/Video/kit_dataset'
breakfast_annotation = 'dataset/breakfast.json'
acitivityNet_annotation = 'dataset/activity_net.v1-3.min.json'
breakfast_path = 'dataset/Video/BreakfastII_15fps_qvga_sync'
iccv_path = 'dataset/Video/activitynet_selected'
iccv_json = ['Long jump.json', 'Triple jump.json']
iccv__annotation = breakfast_annotation = 'dataset/iccv.json'
dataset = 'Ocado'
breakfast_fps = 15


#confusion
base_mult = Batch_size*4
reset_confusion_step = base_mult*int(22000/base_mult)
update_confusion = base_mult*int(10000/base_mult)
val_step = base_mult*int(6000/base_mult)
no_sil_step = 100000