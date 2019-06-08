import os
import cv2
import numpy as np
import random
import pprint
import time
from poseEstimation import OpenPose
from tqdm import tqdm
import pickle
import config
import multiprocessing.dummy as mp
from PIL import Image
import datetime

with open('dataset/ordered_collection.pkl', 'rb') as f:
    collection = pickle.load(f)

with open('dataset/id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)

comb_count = {}
for video in collection:
    step_history = []
    path = collection[video][0]['path']
    if 'cam_0' in path or 'cam_6' in path:
        continue
    for sec in collection[video]:
        entry = collection[video][sec]

        current_label =entry['now_label']
        next_label =entry['next_label']
        couple =entry['all_next_label']
        path =entry['path']
        segment =entry['segment'] 
        label_history =entry['history'] 
        step =entry['time_step']
        help_label =entry['help'] 
        step_history =entry['step_history'] 
        obj_label =entry['obj_label'] 
        
        step_history.append(current_label)
        comb = ''
        if len(step_history) >= 4:
            for step_label in step_history[-4:]:
                comb += id_to_label[step_label] + '#'
            help_word = id_to_label[help_label]
            if help_word == 'sil':
                help_word = 'sil sil sil'
            comb +=  id_to_label[next_label] + '#' +  help_word
            comb = comb.replace(' ', '#')
            video_name = video.split('/')[-1]
            if comb not in comb_count:
                comb_count[comb] = {}
                comb_count[comb]['count'] = 1
                comb_count[comb]['path'] = video_name
                comb_count[comb]['second'] = sec
            else:
                comb_count[comb]['count'] += 1
                comb_count[comb]['path'] = video_name
                comb_count[comb]['second'] = sec

with open('dataset/comb_count.csv', 'w') as f:
    for key in comb_count.keys():
        f.write("%s,%s,%s,%s\n"%(key, comb_count[key]['count'], comb_count[key]['path'], comb_count[key]['second']))