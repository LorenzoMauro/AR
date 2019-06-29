import tensorflow as tf
import os
import multiprocessing.dummy as mp
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pprint
from batch_generator_test import IO_manager
from confusion_tool import confusion_tool as confusion_tool_class
from network_seq import activity_network
from network_seq import Training
from network_seq import Input_manager
import config
from tensorflow.python.client import device_lib

pp = pprint.PrettyPrinter(indent=4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '0'
tf_config = tf.ConfigProto(inter_op_parallelism_threads=config.inter_op_parallelism_threads, allow_soft_placement = True)
tf_config.gpu_options.allow_growth = config.allow_growth

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']
    # return local_device_protos

def train():
    with tf.Session(config=tf_config) as sess:
        IO_tool = IO_manager(sess)
        number_of_classes = IO_tool.num_classes
        available_gpus = get_available_gpus()
        j=0
        Net_collection = {}
        Input_net = Input_manager(len(available_gpus), IO_tool)
        for device in available_gpus:
            with tf.device(device.name):
                print(device.name)
                with tf.variable_scope('Network') as scope:
                    if j>0:
                        scope.reuse_variables()
                    Net_collection['Network_' + str(j)] = activity_network(number_of_classes, Input_net, j, IO_tool)
                    j = j+1
        # with tf.device(available_gpus[-1].name):
        Train_Net = Training(Net_collection, IO_tool)
        IO_tool.start_openPose()
        train_writer = tf.summary.FileWriter("logdir/train", sess.graph)
        val_writer = tf.summary.FileWriter("logdir/val", sess.graph)
        
        IO_tool.openpose.load_openpose_weights()
        sess.run(Train_Net.init)

        # Loading initial C3d or presaved network
        if os.path.isfile('./checkpoint/checkpoint') and config.load_pretrained_weigth:
            print('new model loaded')
            ckpts = tf.train.latest_checkpoint('./checkpoint')
            vars_in_checkpoint = tf.train.list_variables(ckpts)
            variables = tf.contrib.slim.get_variables_to_restore()
            ckpt_var_name = []
            ckpt_var_shape = {}
            for el in vars_in_checkpoint:
                ckpt_var_name.append(el[0])
                ckpt_var_shape[el[0]] = el[1]
            var_list = [v for v in variables if v.name.split(':')[0] in ckpt_var_name]
            var_list = [v for v in var_list if list(v.shape) == ckpt_var_shape[v.name.split(':')[0]]]
            loader = tf.train.Saver(var_list=var_list)
            loader.restore(sess, ckpts)
        elif config.load_c3d:
            print('original c3d loaded')
            Train_Net.c3d_loader.restore(sess, config.c3d_ucf_weights)
        step = 0
        training = True
        with tf.name_scope('whole_saver'):
            whole_saver = tf.train.Saver()
        whole_saver.save(sess, config.model_filename, global_step=0)
        pbar_whole = tqdm(total=(config.tot_steps), desc='Step')
        confusion_tool = confusion_tool_class(number_of_classes, IO_tool, sess, Train_Net)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        while step < config.tot_steps:
            ready_batch = 0
            pbar = tqdm(total=(config.tasks * config.Batch_size * config.seq_len * len(available_gpus) * config.frames_per_step + len(available_gpus)*config.tasks - 1), leave=False, desc='Batch Generation')
            ready_batch = IO_tool.compute_batch(pbar, Devices=len(available_gpus), Train=training)
            for batch in ready_batch:
                summary, t_op, now_pred, next_pred, c3d_pred, help_pred, c_state, h_state = sess.run([Train_Net.merged, Train_Net.train_op,
                                                                                            Train_Net.predictions_now_conc, Train_Net.predictions_next_conc, Train_Net.predictions_c3d_conc, Train_Net.predictions_help_conc,
                                                                                            Train_Net.c_out_list, Train_Net.h_out_list],
                                                                                            # options=run_options,
                                                                                            # run_metadata=run_metadata,
                                                                                            feed_dict={Input_net.input_batch: batch['X'],
                                                                                                        Input_net.drop_out_prob: config.plh_dropout,
                                                                                                        Input_net.in_lstm_drop_out_prob: config.plh_dropout,
                                                                                                        Input_net.out_lstm_drop_out_prob: config.plh_dropout,
                                                                                                        Input_net.state_lstm_drop_out_prob: config.plh_dropout,
                                                                                                        Input_net.labels: batch['Y'],
                                                                                                        Input_net.help_labels: batch['help_Y'],
                                                                                                        Input_net.c_input: batch['c'],
                                                                                                        Input_net.h_input: batch['h'],
                                                                                                        Input_net.next_labels: batch['next_Y'],
                                                                                                        Train_Net.now_weight: batch['now_weight'],
                                                                                                        Train_Net.next_weight: batch['next_weight'],
                                                                                                        Train_Net.help_weight: batch['help_weight'],
                                                                                                        Input_net.obj_input: batch['obj_input']})

                # print(batch['Y'][0,0,...], now_pred[0,...])
                # print(batch['Y'][0,0,...], c3d_pred[0,...])
                # print(batch['next_Y'][0,0,...], next_pred[0,...])
                # print(batch['help_Y'][0,0,...], help_pred[0,...])
                for j in range(len(batch['video_name_collection'])):
                    for y in range(c_state[0].shape[1]):
                        video_name = batch['video_name_collection'][j][y]
                        segment = batch['segment_collection'][j][y][1]
                        h = h_state[j][:, y, :]
                        c = c_state[j][:, y, :]
                        IO_tool.add_hidden_state(video_name,
                                                segment,
                                                h,
                                                c)
                        # IO_tool.add_output_collection(batch['video_name_collection'][j][y],
                        #                         batch['segment_collection'][j][y][1],
                        #                         multi[j][y],
                        #                         batch['multi_next_Y'][j][y],
                        #                         forecast[j][y],
                        #                         batch['next_Y'][j][y])

                # print(np.reshape(batch['Y'][...,:4], (-1, 1)).shape)
                # print(np.reshape(now_pred[...,:4], (-1, 1)).shape)

                # print(np.reshape(batch['Y'][...,:4], (-1, 1)).shape)
                # print(np.reshape(c3d_pred, (-1, 1)).shape)

                # print(np.reshape(batch['next_Y'], (-1,1)).shape)
                # print(np.reshape(next_pred, (-1, 1)).shape)

                # print(np.reshape(batch['help_Y'][...,:3], (-1, 1)).shape)
                # print(np.reshape(help_pred[...,:3], (-1, 1)).shape)

                step = step + config.Batch_size*len(available_gpus)
                # train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                train_writer.add_summary(summary, step)
                pbar_whole.update(config.Batch_size*len(available_gpus))

                target_now = np.reshape(batch['Y'][...,:config.seq_len], (-1, 1))
                pred_now = np.reshape(now_pred[...,:config.seq_len], (-1, 1))

                target_c3d =np.reshape(batch['Y'][...,:config.seq_len], (-1, 1))
                pred_c3d = np.reshape(c3d_pred, (-1, 1))

                target_next = np.reshape(batch['next_Y'], (-1,1))
                pred_next = np.reshape(next_pred, (-1,1))

                target_help = np.reshape(batch['help_Y'][...,:3], (-1, 1))
                pred_help = np.reshape(help_pred[...,:3], (-1, 1))

                data_collection = {}
                data_collection['now_train'] = {}
                data_collection['now_train']['taget'] = target_now
                data_collection['now_train']['y_pred'] = pred_now
                data_collection['c3d_train'] = {}
                data_collection['c3d_train']['taget'] = target_c3d
                data_collection['c3d_train']['y_pred'] = pred_c3d
                data_collection['next_train'] = {}
                data_collection['next_train']['taget'] = target_next
                data_collection['next_train']['y_pred'] = pred_next
                data_collection['help_train'] = {}
                data_collection['help_train']['taget'] = target_help
                data_collection['help_train']['y_pred'] = pred_help
                confusion_tool.update_confusion(data_collection, train_writer, step)

                

                if step % config.val_step == 0 or (step + 1) == config.tot_steps:
                    validation = True
                    if validation:
                        val_step = step
                        pbar_val = tqdm(total=(config.tasks * config.Batch_size * config.seq_len * len(available_gpus) * config.frames_per_step + len(available_gpus)*config.tasks - 1), leave=False, desc='Validation Generation')

                        val_batch = IO_tool.compute_batch(pbar_val, Devices=len(available_gpus), Train=False, augment=False)
                        pbar_val_batch = tqdm(total=len(val_batch), leave=False, desc='Validation run')

                        for batch in val_batch:

                            summary, now_pred, next_pred, c3d_pred, help_pred, softmax_now, softmax_next, softmax_help, c_state, h_state = sess.run([Train_Net.merged,
                                                                                                                                                    Train_Net.predictions_now_conc, Train_Net.predictions_next_conc, Train_Net.predictions_c3d_conc, Train_Net.predictions_help_conc,
                                                                                                                                                    Train_Net.softmax_now, Train_Net.softmax_next, Train_Net.softmax_help,
                                                                                                                                                    Train_Net.c_out_list, Train_Net.h_out_list],
                                                                                                                                                    feed_dict={Input_net.input_batch: batch['X'],
                                                                                                                                                                Input_net.labels: batch['Y'],
                                                                                                                                                                Input_net.help_labels: batch['help_Y'],
                                                                                                                                                                Input_net.c_input: batch['c'],
                                                                                                                                                                Input_net.h_input: batch['h'],
                                                                                                                                                                Input_net.next_labels: batch['next_Y'],
                                                                                                                                                                Train_Net.now_weight: batch['now_weight'],          
                                                                                                                                                                Train_Net.next_weight: batch['next_weight'],           
                                                                                                                                                                Train_Net.help_weight: batch['help_weight'],
                                                                                                                                                                Input_net.obj_input: batch['obj_input']})           

                            for j in range(len(batch['video_name_collection'])):
                                for y in range(c_state[0].shape[1]):
                                    video_name = batch['video_name_collection'][j][y]
                                    segment = batch['segment_collection'][j][y][1]
                                    now_label = batch['Y'][j][y]
                                    next_label = batch['next_Y'][j][y]
                                    help_label = batch['help_Y'][j][y]
                                    now_softmax = softmax_now[j][y, ...]
                                    next_softmax = softmax_next[j][y, ...]
                                    help_softmax = softmax_help[j][y, ...]
                                    h = h_state[j][:, y, :]
                                    c = c_state[j][:, y, :]
                                    IO_tool.add_hidden_state(video_name,
                                                segment,
                                                h,
                                                c)
                                    IO_tool.add_output_collection(video_name,
                                                                  segment,
                                                                  now_label,
                                                                  now_softmax,
                                                                  next_label,
                                                                  next_softmax,
                                                                  help_label,
                                                                  help_softmax)

                            target_now = np.reshape(batch['Y'][...,:4], (-1, 1))
                            pred_now = np.reshape(now_pred[...,:4], (-1, 1))

                            target_c3d =np.reshape(batch['Y'][...,:4], (-1, 1))
                            pred_c3d = np.reshape(c3d_pred, (-1, 1))

                            target_next = np.reshape(batch['next_Y'], (-1,1))
                            pred_next = np.reshape(next_pred, (-1,1))

                            target_help = np.reshape(batch['help_Y'][...,:3], (-1, 1))
                            pred_help = np.reshape(help_pred[...,:3], (-1, 1))

                            data_collection = {}
                            data_collection['now_val'] = {}
                            data_collection['now_val']['taget'] = target_now
                            data_collection['now_val']['y_pred'] = pred_now
                            data_collection['c3d_val'] = {}
                            data_collection['c3d_val']['taget'] = target_c3d
                            data_collection['c3d_val']['y_pred'] = pred_c3d
                            data_collection['next_val'] = {}
                            data_collection['next_val']['taget'] = target_next
                            data_collection['next_val']['y_pred'] = pred_next
                            data_collection['help_val'] = {}
                            data_collection['help_val']['taget'] = target_help
                            data_collection['help_val']['y_pred'] = pred_help
                            pbar_val_batch.update(1)
                            confusion_tool.update_confusion(data_collection, val_writer, val_step)
                            val_writer.add_summary(summary, val_step + config.Batch_size*len(available_gpus))
                            val_step += 1
                        pbar_val_batch.close()

                    IO_tool.save_hidden_state_collection()
                    IO_tool.save_output_collection()
                    IO_tool.hidden_states_statistics()
                    whole_saver.save(sess, config.model_filename, global_step=step)


train()
