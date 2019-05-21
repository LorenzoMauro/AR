import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
import multiprocessing.dummy as mt


class activity_network:
    def __init__(self, sess=None):
        # creating a Session
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # load architecture in graph and weights in session and initialize
        self.architecture = tf.train.import_meta_graph('architecture/Net_weigths.model-10250.meta')
        self.latest_ckp = tf.train.latest_checkpoint('./architecture')
        self.graph = tf.get_default_graph()
        self.architecture.restore(self.sess, self.latest_ckp)
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        sess.run(self.init)

        # Network parameters
        self.batch_size = 1
        self.number_of_clip = 4
        self.frames_per_clip = 10
        self.num_classes = 16
        self.input_H_res = 368
        self.input_W_res = 368
        self.max_num_frames = 2 * 4 * 10
        self.fps = 30

        # Show progress bar to visualize datasets creation
        self.use_pbar = True

        # Retrieving Pose input and outputs
        self.pose_input = self.graph.get_tensor_by_name('image:0')
        pose_out_name_1 = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm' in n.name][0]
        self.pose_out_1 = self.graph.get_tensor_by_name(pose_out_name_1 + ":0")
        pose_out_name_2 = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm' in n.name][0]
        self.pose_out_2 = self.graph.get_tensor_by_name(pose_out_name_2 + ":0")

        # Retrieving activity recognition network inputs and outputs
        self.input = self.graph.get_tensor_by_name("Inputs/Input/Input:0")
        self.h_input = self.graph.get_tensor_by_name("Inputs/Input/h_input:0")
        self.c_input = self.graph.get_tensor_by_name("Inputs/Input/c_input:0")
        self.now_label = self.graph.get_tensor_by_name("Inputs/Now_target/now_label:0")
        self.help_label = self.graph.get_tensor_by_name("Inputs/Now_target/help_label:0")
        self.next_label = self.graph.get_tensor_by_name("Inputs/Now_target/next_label:0")
        self.now_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Now_Decoder_inference/softmax_out:0")
        self.now_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Help_Decoder_inference/softmax_out:0")
        self.now_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Next_classifier/Softmax_out:0")

    def create_graph_log(self):
        # This function create a tensorboard log which shows the network as_graph_def
        file_writer = tf.summary.FileWriter("tensorboardLogs", tf.get_default_graph())
        file_writer.close()

    def extract_frames_from_video(self, video_path, segment):
        pass

    def compute_optical_flow(self, frames_matrix):
        pass

    def compute_pose(self, X):
        pass

    def multithread_whole_preprocessing(self, video_path, segment):
        pass

    def multithread_matrix_preprocessing(self, Data):
        pass

    def create_input_tensor(self, ready_batch):
        pass

    def compute_activity_from_video(self, video_path, t_end):
        pass

    def compute_activity_from_tensor(self, Data):
        pass