import os
import random
import json

dataset = 'FB15K-237'

if not os.path.exists('./data'):
    os.mkdir('./data/')

if not os.path.exists(f'./data/{dataset}'):
    os.mkdir(f'./data/{dataset}/')

src_path = f'../origin_data/{dataset}/'
tgt_path = f'./data/{dataset}/'

random.seed(123)

train_tasks = json.load(open(os.path.join(src_path, 'train_tasks.json'), 'r'))
train_samples = []
for rel in train_tasks.keys():
    for sample in train_tasks[rel]:
        h, r, t = sample
        train_samples.append(h + '\t' + r + '\t' + t + '\n')

random.shuffle(train_samples)

#split = len(train_data) - 1
split = len(train_samples) - len(train_samples) / 20.0
split = int(split)

new_train = train_samples[:split]
new_valid = train_samples[split:]

with open(os.path.join(tgt_path, 'train.tsv'), 'w') as fp:
    fp.writelines(new_train)

with open(os.path.join(tgt_path, 'valid.tsv'), 'w') as fp:
    fp.writelines(new_valid)

if os.path.exists(os.path.join(src_path, 'test.tsv')):
    with open(os.path.join(src_path, 'test.tsv'), 'r') as fp:
        test_data = fp.readlines()
        new_test = test_data

    with open(os.path.join(tgt_path, 'test.tsv'), 'w') as fp:
        fp.writelines(new_test)
