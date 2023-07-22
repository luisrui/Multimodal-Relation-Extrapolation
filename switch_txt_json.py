import json

dataset = "WN18/"

with open('./origin_data/' + dataset + 'entity2id.txt', 'r') as f:
    entity2ids = dict()
    entity2id = f.readlines()
    for line in entity2id[1:]:
        entity, id = line.split()
        entity2ids.update({entity : int(id)})

with open('./origin_data/' + dataset + 'relation2id.txt', 'r') as f:
    relation2ids = dict()
    relation2id = f.readlines()
    for line in relation2id[1:]:
        relation, id = line.split()
        relation2ids.update({relation : int(id)})

with open('./origin_data/' + dataset + 'relation2ids.json', 'w') as f:
    json.dump(relation2ids, f)

with open('./origin_data/' + dataset + 'entity2ids.json', 'w') as f:
    json.dump(entity2ids, f)
