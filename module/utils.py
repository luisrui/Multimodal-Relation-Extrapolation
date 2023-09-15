import os
from os import path as osp
import torch
import numpy as np
import json
import einops
import pickle
from tqdm.auto import trange
from collections import defaultdict
from torch_geometric.loader import NeighborSampler
from .data import MMKGDataset

# @title Model size config
def get_transformer_by_config(model_type, config):
    if model_type == 'small':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'base':
        config.emb_dim = 768
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 12
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'large':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'huge':
        config.emb_dim = 1280
        config.dec_emb_dim = 512
        config.depth = 32
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'debug':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 2
        config.dec_depth = 2
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'tiny':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 2
        config.dec_depth = 2
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'tiny4':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 4
        config.dec_depth = 4
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    else:
        raise ValueError('Unsupported model type!')

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


def generate_subgraph(triples):
    heads, relations, tails = triples
    edge_index = torch.tensor([[h, t] for h, t in zip(heads, tails)], dtype=torch.long).t().contiguous()
    edge_type = relations.clone().detach()
    node_list = torch.unique(edge_index.flatten())
    return node_list, edge_index, edge_type

def generate_batchdata(triples, images, texts, text_padding_masks):
    node_list, sub_edge_index, sub_edge_type = generate_subgraph(triples)
    

    batch = dict()
    images = torch.cat(images, dim=0)
    texts = torch.cat(texts, dim=0)
    text_padding_masks = torch.cat(text_padding_masks, dim=0)
    batch['image'] = images
    batch['text'] = texts
    batch['text_padding_mask'] = text_padding_masks
    return node_list, sub_edge_index, sub_edge_type, batch

def generate_fix_samples(args, model, sample_size, batch_size, mode):
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    #Load test info
    print(f'Start load {mode} dataset!')
    
    graph_test_dataset = MMKGDataset(
        config=MMKGDataset.get_default_config(),
        train_file=f'{mode}_tasks.json',
        name=args.dataset,  
        root=data_path, 
        mode=mode,
        mm_info=None
    )
    print(f'Finish load {mode} dataset!')

    test_graph = graph_test_dataset.get_struc_dataset()
    
    test_dataloader = NeighborSampler(
        test_graph.edge_index,
        sizes=[sample_size],
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.dataloader_n_workers
    )

    model.eval()
    saved_info = dict()
    step_counter = trange(0, len(test_dataloader), ncols=0)
    for step, data in zip(step_counter, test_dataloader):
        batch_size, n_id, adjs = data
        edge_index_expand, edge_type_expand = model.generate_eval_list(
            local_global_id={k: v.item() for k, v in zip(range(len(n_id)), n_id)},
            edge_index=adjs.edge_index.to(device),
            edge_type=test_graph.edge_type[adjs.e_id], 
            )
        saved_info[step] = {
            'step':step,
            'batch_size':len(adjs.e_id),
            'edge_index_expand':edge_index_expand.numpy().tolist(),
            'edge_type_expand':edge_type_expand.numpy().tolist(),
            'n_id':n_id.numpy().tolist()
        }
    with open(f'./origin_data/{args.dataset}/{mode}/sub_{mode}_samples.json', 'w') as fout:
        json.dump(saved_info, fout)
    print('generate success!')