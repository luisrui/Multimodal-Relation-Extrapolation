import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os import path as osp
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborSampler

from tqdm.auto import tqdm, trange
from ml_collections import ConfigDict
from collections import deque
import numpy as np
import itertools
import pickle

from args import read_options
from module.utils import (
    set_random_seed, generate_m3ae_embed, generate_batchdata
)
from module.model import (
    UnifiedModel
)
from module.data import (
    MMKGDataset, MultiModalKnowledgeGraphDataset
)
from module.NegativeSampling import NegativeSampling
from module.loss import MarginLoss

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)

    print('Start dataset preprocessing!')
    graph_dataset = MMKGDataset(
        config=MMKGDataset.get_default_config(),
        train_file='train_tasks_all.json',
        name=args.dataset,  
        root=data_path, 
    )
    print('Finish dataset preprocessing!')

    print('Start Model Instantiation!')
    part_model = UnifiedModel(
        args=args,
        mm_info=graph_dataset.mm_info,
        hidden_channels=200,
        dataset=graph_dataset,
        num_relations=237
    )
    
    model = NegativeSampling(
        args = args,
        model= part_model,
        loss_fn = MarginLoss(margin=10.0),
        neg_rel = 0,
        neg_ent = 1,
        sampling_mode = 'normal'
    ).to(device)
    print('Fisish Model Instantiation!')

    graph = graph_dataset.get_struc_dataset()
    dataloader = NeighborSampler(
        graph.edge_index,
        sizes=[args.sample_size],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_n_workers
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_warmup_epochs * steps_per_epoch // args.accumulate_grad_steps,
        T_mult=1,
        eta_min=args.lr_minimum
    )

    start_step = 0
    step_counter = trange(start_step, total_steps, ncols=0)

    # def infinite_iterator(iterator):
    #     inf_iterator = itertools.cycle(iterator)
    #     while True:
    #         yield next(inf_iterator)
    # dataloader = infinite_iterator(dataloader)

    losses = deque([], 100)
    print('Start Fusion Training!')
    for step, data in zip(step_counter, dataloader):
        batch_size, n_id, adjs = data
        batch_data, features = graph_dataset.generate_batch(n_id)
        batch_data['image'] = batch_data['image'].to(torch.float32).to(device)
        batch_data['text'] = batch_data['text'].to(device)
        batch_data['text_padding_mask'] = batch_data['text_padding_mask'].to(device)
        batch_data['unpaired_text'] = batch_data['unpaired_text'].to(device)
        batch_data['unpaired_text_padding_mask'] = batch_data['unpaired_text_padding_mask'].to(device)
        optimizer.zero_grad()
        loss, info = model(
            node_list=n_id.to(device),
            x=features, 
            edge_index=adjs.edge_index.to(device),
            edge_type=graph.edge_type[adjs.e_id], 
            batch=batch_data
            )
        #loss, info = first_fusion_train(model, batch, args)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        step_counter.set_description("Step %d | loss: %f | loss_res_gcn: %f | image_loss: %f | text_loss: %f " % (
            step, 
            np.mean(losses), 
            info['loss_res_gcn'], 
            info['image_loss'], 
            info['text_loss']
            )
        )

        if (step+1) % steps_per_epoch == 0:
            scheduler.step()
    
    torch.save(model, f"./saved_models/Unimodal_{args.dataset}.pth")

if __name__ == "__main__":
    args = read_options()
    main(args)