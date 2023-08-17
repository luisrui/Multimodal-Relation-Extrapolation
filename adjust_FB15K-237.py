import os
import json
import numpy as np

data_path = './origin_data/FB15K-237/'
train_tasks = json.load(open('./origin_data/FB15K-237/train_tasks.json'))
test_tasks = json.load(open('./origin_data/FB15K-237/test_tasks.json'))
tasks = {**train_tasks, **test_tasks}
rels = list(tasks.keys())
rels.sort(key = lambda x:tasks[x].__len__(), reverse=True)
entities = []
train_rels = []
for i, r in enumerate(rels):
    for tri in tasks[r]:
        if tri[0] not in entities:
            entities.append(tri[0])
        if tri[2] not in entities:
            entities.append(tri[2])
    train_rels.append(r)
    if entities.__len__() >= 14541 - 50:
        print(i)
        break
print('test relations', 237 - (i + 1))

train_tasks, test_tasks = dict(), dict() 
for rel in train_rels:
    train_tasks[rel] = tasks[rel]
rels = list(tasks.keys())
rels.sort(key = lambda x:tasks[x].__len__(), reverse=True)
test_rels = rels[len(train_rels):]

entities_all = []
for rel in tasks.keys():
    for tri in tasks[rel]:
        entities_all.append(tri[0])
        entities_all.append(tri[2])
ent_uni = np.unique(entities_all)

delete_ent = [i for i in ent_uni if i not in entities]
for rel in test_rels:
    samples = []
    for tri in tasks[rel]:
        h, r, t = tri
        if h not in delete_ent and t not in delete_ent:
            samples.append(tri)
    if samples.__len__() > 0:
        test_tasks[rel] = samples

with open(os.path.join(data_path, 'test_tasks.json'), 'w') as f:
    json.dump(test_tasks, f)
with open(os.path.join(data_path, 'train_tasks.json'), 'w') as f:
    json.dump(train_tasks, f)

print("Finish Data Spliting!")