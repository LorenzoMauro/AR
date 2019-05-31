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
import prep_dataset_manager as prep_dataset

with open('dataset/output_states_collection.pkl', 'rb') as f:
    output = pickle.load(f)

pp.pprint(output)