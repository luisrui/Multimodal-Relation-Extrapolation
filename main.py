import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
import json
import pickle

from args import read_options
from module.utils import (set_random_seed, generate_m3ae_embed, generate_batchdata)
from module.model import UnifiedModel
from module.data import MMKGDataset
from module.NegativeSampling import NegativeSampling
from module.loss import MarginLoss

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    e_id = json.load(open(os.path.join(data_path, 'entity2ids_clear.json')))
    r_id = json.load(open(os.path.join(data_path, 'relation2ids_clear.json')))
    with open(os.path.join(data_path, 'train.tsv'), 'r') as f:
        h, r, t = list(), list(), list()
        for line in f.readlines():
            head, rel, tail = line[:-1].split('\t')
            if head in e_id.keys() and tail in e_id.keys() and rel in r_id.keys():
                h.append(e_id[head])
                r.append(r_id[rel])
                t.append(e_id[tail])
        triples = [h, r, t]
    with open(os.path.join(data_path, 'MultiModalInfo_clear.pkl'), 'rb') as f:
        mm_info = pickle.load(f)

    if not args.evaluate:
        print('Start dataset preprocessing!')
        graph_train_dataset = MMKGDataset(
            config=MMKGDataset.get_default_config(),
            train_file='train_tasks_clear.json',
            name=args.dataset,  
            root=data_path,
            mode='train',
            mm_info=mm_info,
        )
        print('Entity Number:', graph_train_dataset.num_nodes)
        print('Finish dataset preprocessing!')

        print('Start Model Instantiation!')
        part_model = UnifiedModel(
            args=args,
            hidden_channels=200,
            dataset=graph_train_dataset,
            num_relations=235
        )
        model = NegativeSampling(
            args = args,
            whole_triples=triples,
            model= part_model,
            loss_fn = MarginLoss(margin=3.0),
            neg_rel = 0,
            neg_ent = 1,
            sampling_mode = 'normal'
        ).to(device)
        if args.load_from_pretrained:
            print('Loading pretrained model')
            model.load_checkpoint(f"./saved_models/{args.saved_model_name}.ckpt")
        print('Finish Model Instantiation!')

        graph = graph_train_dataset.get_struc_dataset()
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

        def infinite_iterator(iterator):
            inf_iterator = itertools.cycle(iterator)
            while True:
                yield next(inf_iterator)
        dataloader = infinite_iterator(dataloader)

        losses = deque([], 10)
        print('Start Fusion Training!\n')
        model.train()
        for step, data in zip(step_counter, dataloader):
            batch_size, n_id, adjs = data
            batch_data = graph_train_dataset.generate_batch(n_id)
            batch_data['image'] = batch_data['image'].to(device)
            batch_data['text'] = batch_data['text'].to(device)
            batch_data['text_padding_mask'] = batch_data['text_padding_mask'].to(device)
            optimizer.zero_grad()
            loss, info = model(
                local_global_id={k: v.item() for k, v in zip(range(len(n_id)), n_id)},
                edge_index=adjs.edge_index.to(device),
                edge_type=graph.edge_type[adjs.e_id], 
                batch=batch_data
                )
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step_counter.set_description("Step %d | loss: %f | struc_loss: %.2f | image_loss: %.2f | text_loss: %.2f " % (
                step, 
                np.mean(losses), 
                info['struc_loss'], 
                info['image_loss'], 
                info['text_loss'],
                )
            )
            if (step+1) % steps_per_epoch == 0:
                scheduler.step()
        
        print('Finish Training\n')
        model.save_checkpoint(f"./saved_models/{args.saved_model_name}.ckpt")
    
    else:
        hits_at_k = [1, 5, 10]

        print('Start dataset preprocessing!')
        graph_test_dataset = MMKGDataset(
            config=MMKGDataset.get_default_config(),
            train_file='test_tasks_clear.json',
            name=args.dataset,  
            root=data_path, 
            mode='test',
            mm_info=mm_info
        )
        test_graph = graph_test_dataset.get_struc_dataset()
        test_dataloader = NeighborSampler(
            test_graph.edge_index,
            sizes=[args.eval_sample_size],
            batch_size=args.eval_batch_size,
            shuffle=True,
            num_workers=args.dataloader_n_workers
        )
        print('Finish dataset preprocessing!')

        part_model = UnifiedModel(
            args=args,
            hidden_channels=200,
            dataset=graph_test_dataset,
            num_relations=235
        )
        model = NegativeSampling(
            args = args,
            whole_triples=triples,
            model= part_model,
            loss_fn = MarginLoss(margin=10.0),
            neg_rel = 0,
            neg_ent = 50, #building test candidates
            sampling_mode = 'normal'
        ).to(device)
        model.load_checkpoint(f"./saved_models/{args.saved_model_name}.ckpt")
        print('Finish model Instantiation!')

        print('Start evaluation!\n')
        model.eval()
        ranks = []
        step_counter = len(test_dataloader)
        step_counter = trange(0, len(test_dataloader), ncols=0)
        for step, data in zip(step_counter, test_dataloader):
            batch_size, n_id, adjs = data
            batch_data = graph_test_dataset.generate_batch(n_id)
            batch_data['image'] = batch_data['image'].to(device)
            batch_data['text'] = batch_data['text'].to(device)
            batch_data['text_padding_mask'] = batch_data['text_padding_mask'].to(device)
            with torch.no_grad():
                p_score, n_score = model.evaluate(
                    local_global_id={k: v.item() for k, v in zip(range(len(n_id)), n_id)},
                    edge_index=adjs.edge_index.to(device),
                    edge_type=test_graph.edge_type[adjs.e_id], 
                    batch=batch_data
                    )
                raw_ranks = torch.sum(n_score < p_score, dim=1, dtype=torch.long)
                num_ties = torch.sum(n_score == p_score, dim=1, dtype=torch.long)
                branks = raw_ranks + num_ties // 2
                ranks.extend((branks + 1).tolist())
            temp_rank = (branks + 1).tolist()
            temp_mrr = sum([1.0/rank for rank in temp_rank]) / len(temp_rank)
            step_counter.set_description("Step %d | temp_mrr: %f " % (step, temp_mrr))

        mrr = sum([1.0/rank for rank in ranks])/len(ranks)
        hits = []
        for k in hits_at_k:
            hits.append(sum([1.0 if rank <= k else 0.0 for rank in ranks]) / len(ranks))
        hits_at_1, hits_at_5, hits_at_10 = hits
        print(f'[Final Scores] '
              f'MRR: {mrr} \t'
              f'Hits@1: {hits_at_1} \t'
              f'Hits@5: {hits_at_5} \t'
              f'Hits@10: {hits_at_10}')

        print('Finish evaluation!\n')
if __name__ == "__main__":
    args = read_options()
    main(args)
    # model = torch.load(f"./saved_models/Unimodal_{args.dataset}.pth")
    # model.save_checkpoint(f"./saved_models/Unimodal_{args.dataset}.ckpt")