import os
import json
import random
from collections import defaultdict

dataset = 'FB15K-237'
data_path = '../../origin_data/' + dataset

with open(os.path.join(data_path, 'train.tsv'), 'r') as f:
    data = []
    for line in f.readlines():
        data.append(line[:-1].split('\t'))

whole_data = defaultdict(list)
for t in data:
    rel = t[1]
    whole_data[rel].append(t)

e1rel_e2 = json.load(open(os.path.join(data_path, 'e1rel_e2_all.json')))
entities2ids = json.load(open(os.path.join(data_path, 'entity2ids.json')))
entities = list(entities2ids.keys())

rel2candidates_all = dict()
for rel in whole_data.keys():
    rel2candidates_all.update({rel : list()})
    candidates = random.sample(entities, 300)
    rel2candidates_all[rel] = candidates

with open(os.path.join(data_path, 'rel2candidates_all.json'), 'w') as f:
    json.dump(rel2candidates_all, f)

print("Finish Generating the rel2candidates file!")