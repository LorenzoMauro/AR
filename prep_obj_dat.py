import json
import config
import os
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)

def load(name):
    with open('dataset/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

help_dataset = {}
for root, dirs, files in os.walk('dataset/object_label/'):
    for fl in files:
        print(fl)
        if fl.split('.')[1] == 'json' and 'trial' in fl.split('.')[0]:
            path = root +  '/' + fl
            print(path)
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            obj_dataset.update(Dataset)
            print(len(obj_dataset))

print(obj_dataset)

ordered_collection = load('ordered_collection')
new_collection = {}
for path in ordered_collection:
    Trial_name = path.split('/')[-2]
    if 'Trial' not in Trial_name:
        full_video_name = path.split('/')[-1].split('cam')[1]
        cut_video_name = full_video_name.split('cam')[1]
        for new_path in ordered_collection:
            new_video_name = new_path.split('/')[-1]
            if cut_video_name in new_video_name and full_video_name != new_video_name:
                Trial_name = new_path.split('/')[-2]
        print(full_video_name, new_path, Correct_Trial_name)
    new_collection[path] = obj_dataset[Trial_name]
            

with open(config.kit_obj_annotation, 'w') as outfile:
    json.dump(new_collection, outfile)

# pp.pprint(help_dataset)