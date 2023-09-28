import os
from os import path as osp
import torch
import numpy as np
import json
import einops
import pickle
import wandb
import tempfile
import time
import uuid
import torch

from copy import copy
from socket import gethostname
from tqdm.auto import trange
from collections import defaultdict
from ml_collections import ConfigDict
from torch_geometric.loader import NeighborSampler
from .data import MMKGDataset
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
from tqdm import tqdm

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

def load_appendix_data(data_path, mode):
    # Instantiation of multimodal Graph Dataset
    e_id = json.load(open(os.path.join(data_path, 'entity2ids_zsl.json')))
    r_id = json.load(open(os.path.join(data_path, 'relation2ids.json')))
    h, r, t = list(), list(), list()
    with open(os.path.join(data_path, f'{mode}/{mode}_tasks_zsl.json'), 'r') as f:
        print(f'Loading triples from {f.name}')
        task = json.load(f)
    for rel in task.keys():
        for tri in task[rel]:
            head, rel, tail = tri
            h.append(e_id[head])
            r.append(r_id[rel])
            t.append(e_id[tail])
    triples = [h, r, t]
    with open(os.path.join(data_path, 'MultiModalInfo.pkl'), 'rb') as f:
        #print(f'Loading base dataset from {f.name}')
        mm_info = pickle.load(f)

    with open(os.path.join(data_path, 'rel_descriptions_zsl'), 'r') as fin:
        #print(f'Loading relation description from {fin.name}')
        rel_des_file = fin.readlines()
    return triples, mm_info, rel_des_file, e_id, r_id

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = "M3AE"
        config.project = "m3ae"
        config.output_dir = "/media/omnisky/sdb/grade2020/cairui/Dawnet/checkpoints"
        config.random_delay = 0.0
        #config.experiment_id = config_dict.placeholder(str)
        config.experiment_id = '7676b3b021144edeb881f54a5ee7cf17'
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)
        config.prefix_to_id = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            if self.config.prefix_to_id:
                self.config.experiment_id = "{}--{}".format(
                    self.config.prefix, self.config.experiment_id
                )
            else:
                self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        # if "hostname" not in self._variant:
        #     self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.output_dir,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
                resume=True,
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
                pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir

def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length ** 0.5)
    x = inputs.view(batch, height, width, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(batch, height * patch_size, width * patch_size, -1)
    return x

def mask_select(mask, this, other=None):
    if other is None:
        other = torch.tensor(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = torch.unsqueeze(mask, dim=-1)
    return torch.where(mask == 0.0, this, other)

def image_float2int(image):
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

def create_log_images(images, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), n=5):
    images = [x.cpu().numpy() for x in images]
    rows = np.concatenate(images, axis=2)
    result = np.array([rows[i] * std + mean for i in range(n)])
    result = np.concatenate(result, axis=0)
    return image_float2int(result)
  
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

def generate_fix_samples(args, model, sample_size, batch_size, mode):
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    #Load test info
    print(f'Start load {mode} dataset!')
    
    graph_test_dataset = MMKGDataset(
        config=MMKGDataset.get_default_config(),
        train_file=f'{mode}_tasks_zsl.json',
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

def transer_subgraph2candidates(data_path, mode, neg_length=300):
    sub_samples = json.load(open(os.path.join(data_path, mode, f'sub_{mode}_samples.json')))
    pos_neg_tri = {}
    for info in sub_samples.values():
        select_nodes = info['n_id']
        local2global = {idx : key for idx, key in enumerate(select_nodes)}
        batch_size = info['batch_size']
        edge_index_expand = info['edge_index_expand']
        edge_type_expand = info['edge_type_expand']
        samples = [[local2global[h], r, local2global[t]] for h, r, t in zip(edge_index_expand[0], edge_type_expand, edge_index_expand[1])]
        true_triples = samples[:batch_size]
        for idx, true in enumerate(true_triples):
            candidates = [samples[idx + i * batch_size] for i in range(neg_length)]
            head_cor, tail_cor = [], []
            for can in candidates[1:]:
                h, r, t = can
                if h == true[0]:
                    tail_cor.append(can[2])
                else:
                    head_cor.append(can[0])
            key = str(true[0]) + '\t' + str(true[1]) + '\t' + str(true[2])
            pos_neg_tri[key] = {'head' : head_cor, 'tail' : tail_cor}
    with open(os.path.join(data_path, mode, 'sample_candidates.json'), 'w') as f:
        json.dump(pos_neg_tri, f)
    print('generate success!')

def generate_ent_rel_embed(args, dataset, model, device):
    graph = dataset.get_struc_dataset()
    
    ### entity part
    ent_embs_unprocessed = torch.rand(graph.num_nodes, model.model.reduced_dim)
    # generate multimodal cls tokens for each entity
    num_nodes = graph.num_nodes
    batch_size = 512
    batch_num = graph.num_nodes // batch_size - 1 if graph.num_nodes % batch_size == 0 else graph.num_nodes // batch_size
    for i in tqdm(range(batch_num)):
        if batch_size * (i + 1) <= num_nodes:
            node_list = torch.arange(start=batch_size * i, end=batch_size * (i + 1))
        else:
            node_list = torch.arange(start=batch_size * i, end=num_nodes)
        batch_data = dataset.generate_batch(node_list, batch_rels=[])
        if torch.numel(batch_data['image']) == 0:
            batch_data['image'] = None
        else:
            batch_data['image'] = batch_data['image'].to(device)
        if torch.numel(batch_data['text']) == 0:
            batch_data['text'] = None
        else:
            batch_data['text'] = batch_data['text'].to(device)
    
        batch_data['text_padding_mask'] = batch_data['text_padding_mask'].to(device)
        if batch_data['image'] is not None:
            batch, height, width, channels = batch_data['image'].shape
            height, width = height // args.patch_size, width // args.patch_size
            image_patches = batch_data['image'].view(batch, height, args.patch_size, width, args.patch_size, channels)
            image_patches = image_patches.permute(0, 1, 3, 2, 4, 5).contiguous()
            image_patches = image_patches.view(batch, height * width, args.patch_size**2 * channels)
        else:
            image_patches = None
        with torch.no_grad():
            x_gcn, _ = model.model.M3AEmodel.forward_representation(
                image=image_patches,  text=batch_data['text'], text_padding_mask=batch_data['text_padding_mask'], deterministic=True
            )
            x_gcn = x_gcn.view(x_gcn.shape[0], -1)
            ent_embs_unprocessed[node_list[0]:node_list[-1]+1] = x_gcn.cpu()
    
    ent_embs_unprocessed = ent_embs_unprocessed.to(device)
    with torch.no_grad():
        ent_embs = model.model.gcn_forward_encoder(
            x = ent_embs_unprocessed,
            edge_index = graph.edge_index.to(device),
            edge_type = graph.edge_type.to(device)
        )
    ent_embs = ent_embs.cpu()
    ### relation part
    rel_list = torch.arange(0, model.model.num_relations)
    batch_data = dataset.generate_batch([], rel_list)
    with torch.no_grad():
        rel_embs = model.model.forward_relation_emb(
            description_tokens = batch_data['rel_des'].to(device),
            des_padding_mask = batch_data['rel_des_padding_mask'].to(device)
        )
    rel_embs = rel_embs.cpu()

    return ent_embs, rel_embs

        

        