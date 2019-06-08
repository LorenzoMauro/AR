import os
import cv2
import numpy as np
import random
import pprint
import time
from tqdm import tqdm
import pickle
import config
import datetime

with open('dataset/ordered_collection.pkl', 'rb') as f:
    collection = pickle.load(f)

with open('dataset/id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)

def create_comb_string(step_history, next_label, help_label):
    comb = ''
    if len(step_history) >= 4:
            for step_label in step_history[-4:]:
                comb += id_to_label[step_label] + '#'
            help_word = id_to_label[help_label]
            if help_word == 'sil':
                help_word = 'sil sil sil'
            comb +=  id_to_label[next_label] + '#' +  help_word
    return comb

def add_csv_correct(correct_comb_count, comb, video, sec):
    if comb != '':
            comb = comb.replace(' ', '#')
            video_name = video.split('/')[-1]
            if comb not in correct_comb_count:
                correct_comb_count[comb] = {}
                correct_comb_count[comb]['count'] = 1
            else:
                correct_comb_count[comb]['count'] += 1
    return correct_comb_count

def add_csv_wrong(wrong_comb_count, comb, video, sec):
    if comb != '':
            video_name = video.split('/')[-1]
            comb = comb + '#' +  video_name + '#' + str(sec)
            comb = comb.replace(' ', '#')
            wrong_comb_count.append(comb)
    return wrong_comb_count

def check_correct_comb(current_label, next_label, help_label):
    Correct = True
    action_help = help_label.split(' ')[0]
    if current_label == 'giveobj' and  action_help != 'get_from_technician_and_put_on_the_table':
        Correct = False
    if next_label == 'giveobj' and  action_help != 'get_from_technician_and_put_on_the_table':
        Correct = False
    if current_label == 'requestobj' and  action_help != 'give_to_technician':
        Correct = False
    if next_label == 'requestobj' and  action_help != 'give_to_technician':
        Correct = False
    
    return Correct

correct_comb_count = {}
wrong_comb_count = []
for video in collection:
    step_history = []
    if 'cam0' in video or 'cam6' in video:
        continue
    for sec in collection[video]:
        entry = collection[video][sec]

        current_label =entry['now_label']
        next_label =entry['next_label']
        help_label =entry['help'] 
        
        step_history.append(current_label)
        comb = create_comb_string(step_history, next_label, help_label)
        Correct = check_correct_comb(current_label, next_label, help_label)
        if Correct:
            correct_comb_count = add_csv_correct(correct_comb_count, comb, video, sec)
        else:
            wrong_comb_count = add_csv_wrong(wrong_comb_count, comb, video, sec)

with open('dataset/comb_count.csv', 'w') as f:
    for key in correct_comb_count.keys():
        if correct_comb_count[key]['count'] == 1:
            continue
        f.write("%s,%s\n"%(key, correct_comb_count[key]['count']))


with open('dataset/wrong_comb_count.csv', 'w') as f:
    for key in wrong_comb_count:
        f.write("%s\n"%(key))