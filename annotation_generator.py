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

    def create_ocado_annotation(self, dataset):
        for root, dirs, files in os.walk(config.ocado_path):
            for fl in files:
                if fl in dataset:
                    path = root + '/' + fl
                    for index in range(len(dataset[fl])):
                        dataset[fl][index]['activity'] = 'O'
                        label = dataset[fl][index]['label'].split(':')[1].lower()
                        label = self.clean_label(label)
                        dataset[fl][index]['label'] = label
                    for index in range(len(dataset[fl]) - 1):
                        dataset[fl][index]['next_label'] = dataset[fl][index + 1]['label']
                    dataset[fl][-1]['next_label'] = 'sil'
                else:
                    if fl in dataset:
                        del dataset[fl]
        return dataset

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
