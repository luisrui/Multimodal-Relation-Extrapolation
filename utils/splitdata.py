import os
import random

dataset = 'FB15K-237'

if not os.path.exists('./data'):
    os.mkdir('./data/')

if not os.path.exists(f'./data/{dataset}'):
    os.mkdir(f'./data/{dataset}/')

src_path = f'../origin_data/{dataset}/'
tgt_path = f'./data/{dataset}/'

random.seed(123)

with open(os.path.join(src_path, 'train.tsv'), 'r') as fp:
    train_data = fp.readlines()

random.shuffle(train_data)

#split = len(train_data) - 1
split = len(train_data) - len(train_data) / 20.0
split = int(split)

new_train = train_data[:split]
new_valid = train_data[split:]

with open(os.path.join(tgt_path, 'train.tsv'), 'w') as fp:
    fp.writelines(new_train)

with open(os.path.join(tgt_path, 'valid.tsv'), 'w') as fp:
    fp.writelines(new_valid)

with open(os.path.join(src_path, 'test.tsv'), 'r') as fp:
    test_data = fp.readlines()
    new_test = test_data

with open(os.path.join(tgt_path, 'test.tsv'), 'w') as fp:
    fp.writelines(new_test)
