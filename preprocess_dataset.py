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
import datetime
import tensorflow as tf
import json
import h5py
import numpy as np
import time


class preprocess:
    def __init__(self):
        with tf.Session() as sess:
            with h5py.File("Dataset/preprocessed/dataset.h5", "w") as hf:
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
                            print(ret)
                            print(type(prev))
                            gray =  cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                            dset = hf.create_dataset(path + '+' + str(0) + "+gray", data=gray)
                            dset = hf.create_dataset(path + '+' + str(0) + "+rgb", data=prev)

                            pbar_frame = tqdm(total=length, leave=False, desc='Frame')

                            for frame in range(1, length):
                                
                                try:
                                    base = path + '+' + str(frame)
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                                    gray_prev = hf[path + '+' + str(frame-1) + '+gray']
                                    gray_prev = np.array(gray_prev)
                                    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, flow=None,
                                                                    pyr_scale=0.5, levels=1,
                                                                    winsize=15, iterations=3,
                                                                    poly_n=5, poly_sigma=1.1, flags=0)
                                    norm_flow = flow
                                    norm_flow = cv2.normalize(flow, norm_flow, 0, 255, cv2.NORM_MINMAX)
                                    res_im = cv2.resize(im, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                                    pafMat, heatMat = self.openpose.compute_pose_frame(res_im)
                                    dset = hf.create_dataset(base + "+rgb", data=im)
                                    dset = hf.create_dataset(base + "+gray", data=gray)
                                    dset = hf.create_dataset(base + "+of", data=norm_flow)
                                    dset = hf.create_dataset(base + "+pafmat", data=pafMat)
                                    dset = hf.create_dataset(base + "+heatMat", data=heatMat)
                                except Exception as e:
                                    print(e)
                                    print(path + '    frame:' + str(frame))
                                    pass
                                pbar_frame.update(1)
                            pbar_frame.refresh()
                            pbar_frame.close()

                        pbar_file.update(1)
                pbar_file.refresh()
                pbar_file.clear()
                pbar_file.close()
prep = preprocess()
