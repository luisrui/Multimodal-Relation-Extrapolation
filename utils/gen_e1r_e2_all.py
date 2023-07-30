import json
import pandas as pd
import os

dataset = 'FB15K-237'
data_path = '../../origin_data/' + dataset

data = []
with open(os.path.join(data_path, 'train.tsv'), 'r') as f:
    for line in f.readlines():
        line_split = line[:-1].split('\t')
        data.append(line_split)

e1rel_e2 = dict()
for triple in data:
    concept = triple[0] + triple[1]
    if concept not in e1rel_e2.keys():
        e1rel_e2.update({concept : list()})
    e1rel_e2[concept].append(triple[2])

count_train = 0
for rel in e1rel_e2.keys():
    count_train += e1rel_e2[rel].__len__()
print(count_train)
assert count_train == data.__len__()

with open(os.path.join(data_path, 'e1rel_e2_all.json'), 'w') as f:
    json.dump(e1rel_e2, f)

print('Finish generate the e1r_e2_all.json file!')