from sentence_transformers import SentenceTransformer
import numpy as np

dataset = 'FB15K-237'
data_path= f'../origin_data/{dataset}/'

with open(data_path + 'relation2textlong.txt', 'r') as fin:
    rels, texts = [], []
    for line in fin.readlines():
        rel, text = line[:-1].split('\t')
        rels.append(rel)
        texts.append(text)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

np.save(data_path + dataset + '-relations2text.npy', embeddings)
print('Finish pretraining the embedding of text descriptions of relations!')