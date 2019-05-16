import json
import config
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

help_dataset = {}
for root, dirs, files in os.walk('dataset'):
    for fl in files:
        if fl.split('.')[1] == 'json' and 'labels_user' in fl.split('.')[0]:
            path = root +  '/' + fl
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            help_dataset.update(Dataset)

with open(config.kit_activity_annotation, 'w') as outfile:
    json.dump(help_dataset, outfile)

pp.pprint(help_dataset)