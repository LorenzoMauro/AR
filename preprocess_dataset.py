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
            with h5py.File("dataset/preprocessed/dataset.h5", "a") as hf:
                print(hf.keys())
                self.sess = sess
                self.openpose = OpenPose(self.sess)
                self.openpose.load_openpose_weights()
                json_data = open(config.ocado_annotation).read()
                dataset = json.loads(json_data)
                pbar_file = tqdm(total=len(dataset), leave=False, desc='Files')
                for root, dirs, files in os.walk(config.ocado_path):
                    for fl in files:
                        path = root + '/' + fl
                        if fl in dataset:
                            if path not in hf.keys():
                                video = cv2.VideoCapture(path)
                            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            print(path)
                            print(length)
                            video.set(1, 0)
                            ret, prev = video.read()
                            gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                            pbar_frame = tqdm(
                                total=length, leave=False, desc='Frame')
                            video_matrix = np.zeros(
                                shape=(length, 368, 368, 8), dtype=np.uint8)
                            frame_matrix = np.zeros(
                                shape=(1, 368, 368, 8), dtype=float)
                            res_gray = cv2.resize(gray, dsize=(
                                368, 368), interpolation=cv2.INTER_CUBIC)
                            frame_matrix[0, :, :, 7] = res_gray
                            frame_matrix = 255 * frame_matrix  # Now scale by 255
                            frame_matrix = frame_matrix.astype(np.uint8)
                            video_matrix[0, :, :, :] = frame_matrix

                            for frame in range(1, length):
                                try:
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    video.set(1, frame)
                                    ret, im = video.read()
                                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                                    res_gray = cv2.resize(gray, dsize=(
                                        368, 368), interpolation=cv2.INTER_CUBIC)
                                    gray_prev = video_matrix[frame-1, :, :, 7]
                                    gray_prev_float = gray_prev.astype(
                                        np.uint8)
                                    flow = cv2.calcOpticalFlowFarneback(gray_prev_float, res_gray, flow=None,
                                                                        pyr_scale=0.5, levels=1,
                                                                        winsize=15, iterations=3,
                                                                        poly_n=5, poly_sigma=1.1, flags=0)
                                    norm_flow = flow
                                    norm_flow = cv2.normalize(
                                        flow, norm_flow, 0, 255, cv2.NORM_MINMAX)
                                    res_im = cv2.resize(im, dsize=(
                                        368, 368), interpolation=cv2.INTER_CUBIC)
                                    res_norm_flow = cv2.resize(norm_flow, dsize=(
                                        368, 368), interpolation=cv2.INTER_CUBIC)
                                    pafMat, heatMat = self.openpose.compute_pose_frame(
                                        res_im)
                                    res_pafMat = cv2.resize(pafMat, dsize=(
                                        368, 368), interpolation=cv2.INTER_CUBIC)
                                    res_heatMat = cv2.resize(heatMat, dsize=(
                                        368, 368), interpolation=cv2.INTER_CUBIC)
                                    frame_matrix = np.zeros(
                                        shape=(1, 368, 368, 8), dtype=float)

                                    frame_matrix[0, :, :, :3] = res_im
                                    frame_matrix[0, :, :, 7] = res_gray
                                    frame_matrix[0, :, :, 5:7] = res_norm_flow
                                    frame_matrix[0, :, :, 3] = res_pafMat
                                    frame_matrix[0, :, :, 4] = res_heatMat
                                    frame_matrix = 255 * frame_matrix  # Now scale by 255
                                    frame_matrix = frame_matrix.astype(
                                        np.uint8)

                                    video_matrix[frame, :, :, :] = frame_matrix
                                except Exception as e:
                                    print(e)
                                    print(path + '    frame:' + str(frame))
                                    pass
                                pbar_frame.update(1)
                            dset = hf.create_dataset(path, data=video_matrix)
                            pbar_frame.refresh()
                            pbar_frame.close()
                            pbar_file.update(1)
                pbar_file.refresh()
                pbar_file.clear()
                pbar_file.close()
prep = preprocess()
