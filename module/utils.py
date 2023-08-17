import os
import torch
import numpy as np
import json
import einops
import pickle
from tqdm.auto import trange

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_m3ae_embed(src_path, args, model, dataset, unpaired_text_dataset):
    ### load the entity with text description file 
    with open(src_path + "/entity2ids.json", 'r') as fin:
        ent_id = json.load(fin)
    with open(src_path + "/entity2textlong.txt", 'r') as fin:
        ent_text = dict()
        for line in fin.readlines():
            if line[-1] == '\n':
                ent, text = line[:-1].split('\t')
            else:
                ent, text = line.split('\t')
            ent_text.update({ent : text})
        #### split the entity into image-pair category and text-only category
        paired_entity, unpaired_entity = [], []
        for filename in os.listdir(os.path.join(src_path, "images")):
            #entity = filename[1:]
            entity = '/' + filename.replace('.', '/')
            assert entity in ent_text.keys()
            if entity in ent_id.keys():
                paired_entity.append(entity)
        for ent in ent_id.keys():
            if ent not in paired_entity:
                unpaired_entity.append(ent)

    #Read image-text pair in each embedding
    embedding = [None] * len(ent_id)
    
    for idx in trange(len(paired_entity)):
        image, text_token, mask = dataset.__getitem__(idx)
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        image_patches = einops.rearrange(image, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=args.patch_size,
            p2=args.patch_size)
        text_token = torch.from_numpy(text_token).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        ent_name = paired_entity[idx]
        with torch.no_grad():
            representation = model.forward_representation(image=image_patches, text=text_token, text_padding_mask=mask, deterministic=True)
        embedding[ent_id[ent_name]] = representation.numpy()

    #Read unpaired text entity in each embedding
    for idx in trange(len(unpaired_entity)):
        unpaired_text, unpaired_text_padding_mask = unpaired_text_dataset.__getitem__(idx)
        unpaired_text = torch.from_numpy(unpaired_text).unsqueeze(0)
        unpaired_text_padding_mask = torch.from_numpy(unpaired_text_padding_mask).unsqueeze(0)
        ent_name = unpaired_entity[idx]
        with torch.no_grad():
            representation = model.forward_representation(image=None, text=unpaired_text, text_padding_mask=unpaired_text_padding_mask, deterministic=True)
        embedding[ent_id[ent_name]] = representation.numpy()

    #print(embedding)
    with open(os.path.join(src_path, 'M3AE_embed.pkl'), 'wb') as fout:
        pickle.dump(embedding, fout)

def negative_sampling():
    return