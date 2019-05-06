import os
import cv2
import numpy as np
import random
import pprint
import time
from poseEstimation import OpenPose
from tqdm import tqdm
import pickle
from dataset_manager import Dataset
import config
import multiprocessing.dummy as mp
from PIL import Image
import datetime
import tensorflow as tf
import json
import h5py
import numpy as np
import time


class preprocess:
    def __init__(self):
        with tf.Session() as sess:
            with h5py.File("Data/processed/dataset.h5", "w") as hf:
                self.sess = sess
                self.openpose = OpenPose(self.sess)
                self.openpose.load_openpose_weights()
                json_data = open(config.ocado_annotation).read()
                dataset = json.loads(json_data)
                pbar_file = tqdm(total=len(dataset), leave=False, desc='Files')
                for root, dirs, files in os.walk(config.ocado_path):
                    for fl in files:
                        if fl in dataset:

                            dict_video = {}
                            path = root + '/' + fl
                            video = cv2.VideoCapture(path)
                            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                            #Zero Frame
                            video.set(1, 0)
                            ret, prev = video.read()
                            dict_video[0] = {}
                            dict_video[0]['rgb'] = prev
                            dict_video[0]['gray'] = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                            pbar_frame = tqdm(total=length, leave=False, desc='Frame')

                            for frame in range(1, length):
                                try:
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                                    gray_prev = dict_video[frame-1]['gray']
                                    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, flow=None,
                                                                    pyr_scale=0.5, levels=1,
                                                                    winsize=15, iterations=3,
                                                                    poly_n=5, poly_sigma=1.1, flags=0)
                                    norm_flow = flow
                                    norm_flow = cv2.normalize(flow, norm_flow, 0, 255, cv2.NORM_MINMAX)
                                    res_im = cv2.resize(im, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                                    pafMat, heatMat = self.openpose.compute_pose_frame(res_im)
                                    base = path + '_' + str(frame)
                                    dset = hf.create_dataset(base + "_rgb", data=im)
                                    dset = hf.create_dataset(base + "_gray", data=gray)
                                    dset = hf.create_dataset(base + "_of", data=norm_flow)
                                    dset = hf.create_dataset(base + "_pafmat", data=pafMat)
                                    dset = hf.create_dataset(base + "_heatMat", data=heatMat)

                                    # dict_video[frame] = {}
                                    # dict_video[frame]['rgb'] = im
                                    # dict_video[frame]['gray'] = gray
                                    # dict_video[frame]['of'] = norm_flow
                                    # dict_video[frame]['pafmat'] = pafMat
                                    # dict_video[frame]['heatMat'] = heatMat
                                except Exception as e:
                                    print(path + '    frame:' + str(frame))
                                    pass
                                pbar_frame.update(1)
                            pbar_frame.refresh()
                            # pbar_frame.clear()
                            pbar_frame.close()
                            # with open(root + '/' + fl.split('.')[0] + '.pkl', 'wb') as f:
                            #     pickle.dump(dict_video, f, pickle.HIGHEST_PROTOCOL)

                        pbar_file.update(1)
                pbar_file.refresh()
                pbar_file.clear()
                pbar_file.close()
prep = preprocess()
