import json
import config
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

class Annotation:
    def __init__(self):
        if config.dataset is 'Ocado':
            json_data = open(config.ocado_annotation).read()
            Dataset = json.loads(json_data)
            Dataset = self.create_ocado_annotation(Dataset)
            self.Dataset = Dataset

    def create_ocado_annotation(self, activity_dataset, help_dataset):
        label_collection = []
        for root, dirs, files in os.walk(config.ocado_path):
            for fl in files:
                if fl in activity_dataset:
                    path = root + '/' + fl
                    video = cv2.VideoCapture(path)
                    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
                    fps = video.get(cv2.CAP_PROP_FPS)
                    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                    for index in range(len(activity_dataset[fl])):
                        label = activity_dataset[fl][index]['label'].split(':')[1].lower()
                        label = self.clean_label(label)
                        activity_dataset[fl][index]['label'] = label
                        label_collection.append(label)

                    for index in range(len(help_dataset[fl])):
                        label = help_dataset[fl][index]['label'].split(':')[0].lower()
                        label = self.clean_label(label)
                        help_dataset[fl][index]['label'] = label
                        label_collection.append(label)

        label_to_id, id_to_label = create_labels_mappings(label_collection)
        frames_label = self.compute_frame_label(activity_dataset, help_dataset, config.kit_path, label_to_id)

        for fl in activity_dataset[fl]:
            path = root + '/' + fl
            video = cv2.VideoCapture(path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
            fps = video.get(cv2.CAP_PROP_FPS)
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for index in range(len(activity_dataset[fl]) - 1):
                        segment = annotation['milliseconds']
                        frame = 
                        activity_dataset[fl][index]['next_label'] = activity_dataset[fl][index + 1]['label']
                    activity_dataset[fl][-1]['next_label'] = 'sil'
                else:
                    if fl in activity_dataset:
                        del activity_dataset[fl]
        return dataset

    def compute_frame_label(self, activity_dataset, help_dataset, dataset_path, label_to_id):
            collection = {}
            iter_count = 0
            files_path = {}
            for root, dirs, files in os.walk(dataset_path):
                for fl in files:
                    path = root + '/' + fl
                    if path in activity_dataset.keys():
                        files_path[path] = path
                    elif fl in activity_dataset.keys():
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
                    labels = {'now': label_to_id[label], 'next': label_to_id[label], 'help': label_to_id[label]}
                    for annotation in activity_dataset[entry]:
                        segment = annotation['milliseconds']
                        if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                            if annotation['label'] in list(label_to_id.keys()):
                                labels['now'] = label_to_id[annotation['label']]
                            break
                    for annotation in help_dataset[entry]:
                        segment = annotation['milliseconds']
                        if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                            if annotation['label'] in list(label_to_id.keys()):
                                labels['help'] = label_to_id[annotation['label']]
                            break
                    frames_label[frame] = labels
                for frame in range(1, tot_frames):
                    current_label = frames_label[frame]['now']
                    find_next = True
                    next_frame = frame + 1
                    while True:
                        next_action = frames_label[next_frame]['now']
                        if next_action != current action:
                            frames_label[frame]['next'] = next_action
                            find_next = False
                collection[path] = {}
                collection[path] = frames_label
                pbar.update(1)
            pbar.close()
            return collection

    def create_labels_mappings(self, label_collection):
        label_to_id = {}
        id_to_label = {}
        label_to_id['sil'] = 0
        id_to_label[0] = 'sil'
        label_to_id['go'] = 1
        id_to_label[1] = 'go'
        label_to_id['end'] = 2
        id_to_label[2] = 'end'
        i = 3
        for label in label_collection:
            if label not in label_to_id:
                label_to_id[label] = i
                id_to_label[i] = label
                i += 1
        return label_to_id, id_to_label

    def clean_label(self, label):
        if label[-1] == ' ':
            label = label[:-1]
        elif label == 'run':
            label = 'running'
        elif label == 'jump':
            label = 'jumping'
        elif label == 'stand up':
            label = 'standing up'
        elif label == 'stand up':
            label = 'standing up'
        elif label == 'be ready':
            label = 'getting ready'
        elif label == 'person:ask tool':
            label = 'person:give tool'
        elif label == 'person:ask tool':
            label = 'person:give tool'
        return label
