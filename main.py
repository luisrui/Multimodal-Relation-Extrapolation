import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from os import path as osp
import torch
import torch.nn as nn
import pickle

from tqdm.auto import tqdm, trange
from ml_collections import ConfigDict
from collections import deque
import numpy as np
import itertools

from args import read_options
from module.utils import (
    set_random_seed, generate_m3ae_embed
)
from module.model import (
    UnifiedModel
)
from module.data import (
    MMKGDataset
)
from module.NegativeSampling import NegativeSampling
from module.loss import MarginLoss

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    graph_dataset = MMKGDataset(config=MMKGDataset.get_default_config(), name=args.dataset, root=data_path, device=device)
    if not osp.exists(osp.join(data_path, 'multimodal_processed.pkl')):
        graph_dataset.multimodal_prepro()
    with open(osp.join(data_path, 'multimodal_processed.pkl'), 'rb') as fin:
        graph_dataset.mm_info = pickle.load(fin)
        graph_dataset.preprocessed = True

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
        loss_fn = MarginLoss(margin=5.0),
        neg_rel = 0,
        neg_ent = 1,
        sampling_mode = 'normal'
    ).to(device)
    print('Fisish Model Instantiation!')

    num_nodes = graph_dataset.num_nodes
    node_list_all = torch.arange(0, num_nodes)
    dataloader = torch.utils.data.DataLoader(
        node_list_all,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
    )
    steps_per_epoch = int(len(node_list_all) / args.batch_size)
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

    def infinite_iterator(iterator):
        inf_iterator = itertools.cycle(iterator)
        while True:
            yield next(inf_iterator)
    dataloader = infinite_iterator(dataloader)

    struc_dataset = graph_dataset.get_struc_dataset()
    losses = deque([], 100)
    print('Start Fusion Training!')
    for step, node_list in zip(step_counter, dataloader):
        sub_edge_index, sub_edge_type, batch_data = graph_dataset.generate_batch(node_list)
        loss, info = model(
            node_list=node_list,
            x=part_model.m3ae_tokens, 
            edge_index=sub_edge_index,
            edge_type=sub_edge_type, 
            batch=batch_data
            )
        #loss, info = first_fusion_train(model, batch, args)
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        step_counter.set_description("Step %d | loss: %f | loss_res_gcn: %f | image_loss: %f | text_loss: %f | unpaired_text_loss: %f" % (step, 
                                                                                                                         np.mean(losses), 
                                                                                                                         info['loss_res_gcn'], 
                                                                                                                         info['image_loss'], 
                                                                                                                         info['text_loss'], 
                                                                                                                         info['unpaired_text_loss']))

        if (step+1) % steps_per_epoch == 0:
            scheduler.step()
    
    torch.save(model, f"./saved_models/Unimodal_{args.dataset}.pth")

if __name__ == "__main__":
    args = read_options()
    main(args)