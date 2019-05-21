import json
import config
import os
import pprint
from tqdm import tqdm
import cv2
import pickle
import random
pp = pprint.PrettyPrinter(indent=4)

def load(name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

json_data = open(config.kit_activity_annotation).read()
activity_dataset = json.loads(json_data)
json_data = open(config.kit_help_annotation).read()
help_dataset = json.loads(json_data)
json_data = open(config.kit_help_annotation_temp).read()
help_dataset_temp = json.loads(json_data)


frame_now = load('frame_label')
label_to_id = load('label_to_id')
id_to_label = load('id_to_label')
word_to_id = load('word_to_id')
id_to_word = load('id_to_word')

situation_count_now_next_help = {}
for video in frame_now:
    for frame in frame_now[video]:
            now_label = frame_now[video][frame]['now']
            next_label = frame_now[video][frame]['next']
            help_label = frame_now[video][frame]['help']
            now_word = id_to_label[now_label]
            next_word = id_to_label[next_label]
            help_word = id_to_label[help_label]
            if now_word not in situation_count_now_next_help:
                situation_count_now_next_help[now_word] = {}
            if next_word not in situation_count_now_next_help[now_word]:
                situation_count_now_next_help[now_word][next_word] = {}
            if help_word not in situation_count_now_next_help[now_word][next_word]:
                situation_count_now_next_help[now_word][next_word][help_word] = 0
            situation_count_now_next_help[now_word][next_word][help_word] += 1
        
pp.pprint(situation_count_now_next_help)

situation_count_help_now_next = {}
for video in frame_now:
    for frame in frame_now[video]:
            now_label = frame_now[video][frame]['now']
            next_label = frame_now[video][frame]['next']
            help_label = frame_now[video][frame]['help']
            now_word = id_to_label[now_label]
            next_word = id_to_label[next_label]
            help_word = id_to_label[help_label]
            if help_word not in situation_count_help_now_next:
                situation_count_help_now_next[help_word] = {}
            if now_word not in situation_count_help_now_next[help_word]:
                situation_count_help_now_next[help_word][now_word] = {}
            if next_word not in situation_count_help_now_next[help_word][now_word]:
                situation_count_help_now_next[help_word][now_word][next_word] = 0
            situation_count_help_now_next[help_word][now_word][next_word] += 1
        
pp.pprint(situation_count_help_now_next)

r_video = random.choice(list(help_dataset.keys()))

pp.pprint(help_dataset[r_video])
pp.pprint(help_dataset_temp[r_video])

pp.pprint(id_to_word)
pp.pprint(id_to_label)

