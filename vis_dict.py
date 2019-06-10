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

ordered_collection = load('ordered_collection')

keys = list(ordered_collection.keys())
pp.pprint(ordered_collection[keys[0]][0])

for video in ordered_collection:
    for sec in ordered_collection[video]:
        print(video, sec, ordered_collection[video][sec]['object_label']

