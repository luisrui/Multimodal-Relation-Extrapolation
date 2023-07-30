import json
import pandas as pd
import os
import numpy as np
import random

dataset = 'FB15K-237'
data_path = '../../origin_data/' + dataset
spel = 40

with open(os.path.join(data_path, 'train.tsv'), 'r') as f:
    dataset = f.readlines()

wholedata = []
for line in dataset:
    line_split = line[:-1]
    line_split = line_split.split('\t')
    wholedata.append(line_split)

rel_list = np.unique([triple[1] for triple in wholedata])
random.shuffle(rel_list)

test_tasks, train_tasks = dict(), dict()
count_test, count_train = 0, 0

print('Start Data Split for seen&unseen relations')
for triple in wholedata:
    rel = triple[1]
    if rel in rel_list[:spel]:
        if rel not in test_tasks.keys():
            test_tasks.update({rel : list()})
        test_tasks[rel].append(triple)
        count_test += 1
    else:
        if rel not in train_tasks.keys():
            train_tasks.update({rel : list()})
        train_tasks[rel].append(triple)
        count_train += 1
print("test samples: ", count_test)
print("train samples: ", count_train)

with open(os.path.join(data_path, 'test_tasks.json'), 'w') as f:
    json.dump(test_tasks, f)
with open(os.path.join(data_path, 'train_tasks.json'), 'w') as f:
    json.dump(train_tasks, f)

print("Finish Data Spliting!")