import json
import os
import cv2
from tqdm import tqdm
import random
import pprint
import time
import pickle
import config
from annotation_generator import Annotation
import numpy as np


pp = pprint.PrettyPrinter(indent=4)


class Dataset:
    def __init__(self):
        if (os.path.isfile('dataset/frame_label.pkl') and
                os.path.isfile('dataset/train_collection.pkl') and
                os.path.isfile('dataset/test_collection.pkl') and
                os.path.isfile('dataset/ordered_collection.pkl') and
                os.path.isfile('dataset/label_to_id.pkl') and
                os.path.isfile('dataset/id_to_label.pkl') and
                not config.rebuild):

            self.label_to_id = self.load('label_to_id')
            self.id_to_label = self.load('id_to_label')
            self.number_of_classes = len(self.id_to_label)
            self.activity_to_id = self.load('activity_to_id')
            self.id_to_activity = self.load('id_to_activity')
            self.number_of_activities = len(self.id_to_activity)
            self.frame_now = self.load('frame_label')
            self.train_collection = self.load('train_collection')
            self.test_collection = self.load('test_collection')
            self.ordered_collection = self.load('ordered_collection')
        else:
            self.generate_dataset()

        pp.pprint(self.id_to_label)

    def generate_dataset(self):
        self.whole_dataset = Annotation().Dataset
        self.label_to_id, self.id_to_label = self.create_labels_mappings()
        self.save(self.label_to_id, 'label_to_id')
        self.save(self.id_to_label, 'id_to_label')
        self.number_of_classes = len(self.id_to_label)
        self.activity_to_id, self.id_to_activity = self.create_activity_mappings()
        self.save(self.activity_to_id, 'activity_to_id')
        self.save(self.id_to_activity, 'id_to_activity')
        self.number_of_activities = len(self.id_to_label)
        self.validation_fraction = config.validation_fraction
        self.frame_label = self.compute_frame_label(self.whole_dataset, next=False)
        if not config.split_seconds:
            self.Train_dataset, self.Val_dataset = self.split_dataset()
            self.train_collection,self.ordered_collection, self.train_multi_list, self.train_couple_count, self.max_history= self.new_collection(self.Train_dataset)
            self.test_collection,self.ordered_collection, self.test_multi_list, self.test_couple_count, self.max_history_val = self.new_collection(self.Val_dataset)
            if self.max_history_val > self.max_history:
                self.max_history = self.max_history_val
        else:
            self.collection, self.ordered_collection, self.multi_list, self.couple_count, self.max_history= self.new_collection(self.whole_dataset)
            self.train_collection, self.test_collection = self.split_dataset_second(self.collection)

        self.save(self.frame_label, 'frame_label')
        self.save(self.train_collection, 'train_collection')
        self.save(self.test_collection, 'test_collection')
        self.save(self.ordered_collection, 'ordered_collection')

    def split_dataset(self):
        dataset_train = self.whole_dataset
        validation = {}
        random.seed(time.time())
        entry_val = int(len(self.whole_dataset) * self.validation_fraction)
        for i in range(entry_val):
            entry_name = random.choice(list(dataset_train.keys()))
            validation[entry_name] = dataset_train[entry_name]
            self.whole_dataset.pop(entry_name)
        return dataset_train, validation

    def split_dataset_second(self, dataset):
        validation = {}
        random.seed(time.time())
        entry_val = int(len(self.whole_dataset) * self.validation_fraction)
        for i in range(entry_val):
            if config.balance_key == 'all':
                r_activity = random.choice(list(dataset.keys()))
                r_now = random.choice(list(dataset[r_activity].keys()))
                r_next = random.choice(list(dataset[r_activity][r_now].keys()))
                r_index = random.randrange(len(dataset[r_activity][r_now][r_next]))
                entry = dataset[r_activity][r_now][r_next][r_index]
                if r_activity not in validation:
                    validation[r_activity] = {}
                if r_now not in validation[r_activity]:
                    validation[r_activity][r_now] = {}
                if r_next not in validation[r_activity][r_now]:
                    validation[r_activity][r_now][r_next] = []
                validation[r_activity][r_now][r_next].append(entry)
                del dataset[r_activity][r_now][r_next][r_index]
            else:
                random_couple = random.choice(list(dataset))
                r_index = random.randrange(len(dataset[random_couple]))
                entry = dataset[random_couple][r_index]
                if random_couple not in validation:
                    validation[random_couple] = []
                validation[random_couple].append(entry)
                del dataset[random_couple][r_index]
        return dataset, validation

    def create_labels_mappings(self):
        label_to_id = {}
        id_to_label = {}
        label_to_id['sil'] = 0
        id_to_label[0] = 'sil'
        label_to_id['go'] = 1
        id_to_label[1] = 'go'
        label_to_id['end'] = 2
        id_to_label[2] = 'end'
        i = 3
        for video_name in self.whole_dataset:
            for segment in self.whole_dataset[video_name]:
                label = segment['label'].lower()
                if label not in label_to_id:
                    label_to_id[label] = i
                    id_to_label[i] = label
                    i += 1
        return label_to_id, id_to_label

    def create_activity_mappings(self):
        activity_to_id = {}
        id_to_activity = {}
        activity_to_id['sil'] = 0
        id_to_activity[0] = 'sil'
        i = 1
        for video_name in self.whole_dataset:
            for segment in self.whole_dataset[video_name]:
                activity = segment['activity'].lower()
                if activity not in activity_to_id:
                    activity_to_id[activity] = i
                    id_to_activity[i] = activity
                    i += 1
        return activity_to_id, id_to_activity

    def compute_frame_label(self, dataset, next):
        collection = {}
        iter_count = 0
        files_path = {}
        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                if path in dataset.keys():
                    files_path[path] = path
                elif fl in dataset.keys():
                    files_path[fl] = path

        pbar = tqdm(total=(len(files_path)), leave=False, desc='Generating Frame')

        for entry in files_path:
            path = files_path[entry]
            video = cv2.VideoCapture(path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
            fps = video.get(cv2.CAP_PROP_FPS)
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if tot_frames is 0:
                continue
            frames_label = {}
            for frame in range(1, tot_frames):
                frame_in_msec = (frame / float(fps)) * 1000
                label = 'sil'
                labels = {'now': self.label_to_id[label], 'next': self.label_to_id[label], 'activity': self.activity_to_id[label]}
                for annotation in dataset[entry]:
                    segment = annotation['milliseconds']
                    if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                        if annotation['label'] in list(self.label_to_id.keys()):
                            labels['now'] = self.label_to_id[annotation['label']]
                        if annotation['next_label'] in self.label_to_id.keys():
                            labels['next'] = self.label_to_id[annotation['next_label']]
                        if annotation['activity'] in self.activity_to_id.keys():
                            labels['activity'] = self.activity_to_id[annotation['activity']]
                        break
                frames_label[frame] = labels
            for frame in range(1, tot_frames):
                if frames_label[frame]['now'] == 0:
                    for follow_frame in range(frame + 1, tot_frames):
                        if frames_label[follow_frame]['now'] == 0:
                            pass
                        if frames_label[follow_frame]['now'] != 0:
                            frames_label[frame]['next'] = frames_label[follow_frame]['now']
                            break
            collection[path] = {}
            collection[path] = frames_label
            pbar.update(1)
        pbar.close()
        return collection

    def new_collection(self, dataset):
        collection = {}
        ordered_collection = {}
        couple_count =  {}
        tree_list = {}
        graph_list = {}
        video_by_history = {}
        files_path = {}

        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                if path in dataset.keys():
                    files_path[path] = path
                elif fl in dataset.keys():
                    files_path[fl] = path

        pbar = tqdm(total=(len(files_path)), leave=False, desc='Creating Annotation')
        max_history = 0
        for entry in files_path:
            path = files_path[entry]
            video = cv2.VideoCapture(path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            fps = video.get(cv2.CAP_PROP_FPS)
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            tot_steps = int(tot_frames/(config.window_size*fps))
            label_history = []
            if tot_steps is 0:
                break
            step_label = {}
            not_zero = False
            for step in range(tot_steps):
                max_frame = int((step+1)*config.window_size*fps)+1
                if max_frame > tot_frames:
                    continue
                frame_list = [frame for frame in range(int(step*config.window_size*fps + 1),int((step+1)*config.window_size*fps)+1)]
                segment = [frame_list[0], frame_list[-1]]
                current_label = self.label_calculator(frame_list, path, 'now')
                next_label = self.label_calculator(frame_list, path, 'next')
                activity = self.label_calculator(frame_list, path, 'activity')
                # if current_label == 0:
                    # continue
                if len(label_history) == 0:
                    label_history.append(current_label)
                elif current_label != label_history[-1]:
                    label_history.append(current_label)

                if len(label_history) > max_history:
                    max_history = len(label_history)

                if tuple(label_history) not in tree_list:
                    tree_list[tuple(label_history)] = [next_label]
                else:
                    if next_label not in tree_list[tuple(label_history)]:
                        tree_list[tuple(label_history)].append(next_label)

                if current_label not in graph_list:
                    graph_list[current_label] = [next_label]
                else:
                    if next_label not in graph_list[current_label]:
                        graph_list[current_label].append(next_label)

                couple = str(current_label) + '-' + str(next_label)
                if couple not in couple_count:
                    couple_count[couple] = 1
                else:
                    couple_count[couple] += 1


                entry = {'now_label' : current_label, 'next_label' : next_label, 'all_next_label' : couple,
                         'path': path, 'segment':segment, 'history':label_history, 'activity': activity, 'time_step': step}
                if path not in ordered_collection:
                    ordered_collection[path] = {}
                ordered_collection[path][step] = entry
                 
                if config.balance_key != 'all':
                    if config.balance_key is 'now':
                        balance = current_label
                    elif config.balance_key is 'next':
                        balance = next_label
                    elif config.balance_key is 'couple':
                        balance = couple
                    if balance not in collection:
                        collection[balance] = [entry]
                    else:
                        collection[balance].append(entry)
                elif config.balance_key == 'all':
                    if activity not in collection:
                        collection[activity] = {}
                    if current_label not in collection[activity]:
                        collection[activity][current_label] = {}
                    if next_label not in collection[activity][current_label]:
                        collection[activity][current_label][next_label] = []
                    collection[activity][current_label][next_label].append(entry)

            pbar.update(1)
        for x in collection:
            if config.balance_key == 'all':
                for now_label in collection[x]:
                    for next_label in collection[x][now_label]:
                        for entry in collection[x][now_label][next_label]:
                            if config.tree_or_graph is 'tree':
                                all_next_label = tree_list[tuple(entry['history'])]
                            elif config.tree_or_graph is 'graph':
                                all_next_label = graph_list[entry['now_label']]
                            entry['all_next_label'] = all_next_label
            else:
                for entry in collection[x]:
                    if config.tree_or_graph is 'tree':
                        all_next_label = tree_list[entry['history']]
                    elif config.tree_or_graph is 'graph':
                        all_next_label = graph_list[entry['now_label']]
                    entry['all_next_label'] = all_next_label
        if config.tree_or_graph is 'tree':
            multi_list = tree_list
        elif config.tree_or_graph is 'graph':
            multi_list = graph_list
        transition = np.zeros(shape=(len(self.id_to_label), len(self.id_to_label)), dtype=float)
        total = 0
        for x in couple_count:
            now = int(x.split('-')[0])
            next = int(x.split('-')[1])
            transition[now,next] += couple_count[x]
        for i in range(transition.shape[0]):
            tot_row = 0
            for j in range(transition.shape[1]):
                tot_row +=  transition[i,j]
            for j in range(transition.shape[1]):
                transition[i,j] /=  tot_row

        pbar.close()
        return collection, ordered_collection, multi_list, couple_count, max_history

    def label_calculator(self, frame_list, path, next_current):
        label_clip = {}
        for frame in frame_list:
            if frame not in self.frame_label[path]:
                print(frame_list)
                print(frame_list)
            label = self.frame_label[path][frame][next_current]
            if label not in label_clip:
                label_clip[label] = 0
            label_clip[label] += 1
        try:
            final_label = max(label_clip, key=label_clip.get)
        except Exception as e:
            print(label_clip, frame_list)
            final_label = 0
            pass
        return final_label

    def save(self, obj, name):
        with open('dataset/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
