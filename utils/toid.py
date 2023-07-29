import os, re, json

dataset = 'FB15K-237'

'''
    This file is for generating id for each entity and relation, and map the id to three aforehead
    generated files(train.tsv, test.tsv, valid.tsv) if you have your own mapped ids with entities& 
    relations, just map the three data file into ids
''' 

src_path = f'./data/{dataset}/'
path = lambda x: os.path.join(src_path, x)
file_list = ['train','valid','test']
out = f'./data/{dataset}/'
path2 = lambda x: os.path.join(out, x)

all = []
allset = set()
for file in file_list:
    with open(path(file+'.tsv'), 'r') as fp:
        data = fp.readlines()
        all.append(data)
        allset.update(data)


def toid(data):
    ent2id = {}
    id2ent = {}
    ent_num = 0
    rel2id = {}
    id2rel = {}
    t_ids = set()
    rel_num = 0
    def updateDict(idto, toid, num, item):
        if item in toid.keys():
            return toid[item], num
        toid[item] = num
        idto[num] = item
        return num, num+1
    update_ent = lambda num, item: updateDict(id2ent, ent2id, num, item)
    update_rel = lambda num, item: updateDict(id2rel, rel2id, num, item)
    
    LL = []
    for ii in data:
        L = []
        for aa in ii:
            try:
                h,r,t = re.findall('(.+?)\t(.+?)\t(.+?)\n',aa)[0]
            except:
                h,r,t = aa.split('\t')
            h_id, ent_num = update_ent(ent_num, h)
            t_id, ent_num = update_ent(ent_num, t)
            r_id, rel_num = update_rel(rel_num, r)
            t_ids.add(f'{t_id}\n')
            L.append(f'{h_id} {t_id} {r_id}\n')
        L = [f'{len(L)}\n'] + L
        LL.append(L)
    assert len(ent2id) == len(id2ent)
    assert len(rel2id) == len(id2rel)
    Ent = [f'{len(ent2id)}\n']
    for i in range(len(ent2id)):
        Ent.append(f'{id2ent[i]}\t{i}\n')
    Rel = [f'{len(rel2id)}\n']
    for i in range(len(rel2id)):
        Rel.append(f'{id2rel[i]}\t{i}\n')

    return LL, Ent, Rel, list(t_ids)


all, ent2id, rel2id, t_ids = toid(all)

with open(path2('entity2id.txt'), 'w') as fp:
    fp.writelines(ent2id)

with open(path2('relation2id.txt'), 'w') as fp:
    fp.writelines(rel2id)

with open(path2('t_ids.txt'), 'w') as fp:
    fp.writelines(t_ids)

# with open(f"../origin_data/{dataset}/entity2ids.json", 'r') as fout:
#     ent_id = json.load(fout)
# with open(f"../origin_data/{dataset}/relation2ids.json", 'r') as fout:
#     rel_id = json.load(fout)

# all_id = []
# for data in all:
#     triple_id = []
#     for triple in data:
#         try:
#             h,r,t = re.findall('(.+?)\t(.+?)\t(.+?)\n',triple)[0]
#         except:
#             h,r,t = triple.split('\t')
#         triple_id.append([ent_id[h], rel_id[r], ent_id[t]])
#     triple_id.insert(0, [len(triple_id)])
#     all_id.append(triple_id)

# for idx, file in enumerate(file_list):
#     with open(path2(file+'2id.txt'), 'w') as fp:
#         for triple in all_id[idx]:
#             try:
#                 fp.write(str(triple[0])+'\t'+str(triple[1])+'\t'+str(triple[2])+'\n')
#             except:
#                 fp.write(str(triple[0])+'\n')

for idx, file in enumerate(file_list):
    with open(path2(file+'2id.txt'), 'w') as fp:
        fp.writelines(all[idx])   
