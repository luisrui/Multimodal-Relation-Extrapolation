import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os import path as osp
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborSampler

from tqdm.auto import tqdm, trange
from collections import deque
import numpy as np
import itertools
import json
import pickle
import wandb

from args import read_options
#from args_exp import read_options
#from args_openbg import read_options
#from autodl_args import read_options
from module.utils import (
    set_random_seed, WandBLogger, create_log_images, 
    load_appendix_data, generate_ent_embed, generate_rel_embed, load_pretrained_CC12M
)
from module.model import UnifiedModel, patch_predict_fn
from module.zsl_module import ZSLmodule
from module.data import MMKGDataset
from module.NegativeSampling import NegativeSampling
from module.loss import MarginLoss, SigmoidLoss

def main(args): 
    # logger = WandBLogger(
    #     config=WandBLogger.get_default_config(),
    #     variant=args,
    # )
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    set_random_seed(args.seed)
    data_path = osp.join('./origin_data', args.dataset)

    triples, mm_info, rel_des_file, e2id, r2id = load_appendix_data(data_path, mode='train')
    #### Train Procedure
    print('Start dataset preprocessing!')
    graph_train_dataset = MMKGDataset(
        config=MMKGDataset.get_default_config(),
        train_file='train_tasks_zsl.json',
        name=args.dataset,  
        root=data_path,
        mode='train',
        mm_info=mm_info,
        rel_des_file=rel_des_file
    )
    print('Entity Number:', graph_train_dataset.num_nodes)
    print('Finish dataset preprocessing!')

    print('Start Model Instantiation!')

    part_model = UnifiedModel(
        args=args,
        hidden_channels=200,
        dataset=graph_train_dataset,
        num_relations=graph_train_dataset.num_relations,
        noise_dim = args.noise_dim
    )
    load_pretrained_CC12M(part_model.M3AEmodel, './m3ae/checkpoints/m3ae_small.pkl')
    model = NegativeSampling(
        args = args,
        whole_triples = triples,
        model= part_model,
        loss_fn = MarginLoss(margin=3.0),
        neg_ent = 10,
        sampling_mode = 'normal'
    ).to(device)
    if args.pretrained_model_name != '':
        print(f'Loading pretrained model:{args.pretrained_model_name}')
        state_dict = torch.load(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt", map_location=device)
        del state_dict['model.generate_fc_layer.weight_orig']
        del state_dict['model.generate_fc_layer.weight_v']
        model.load_state_dict(state_dict, strict=False)
        #model.load_checkpoint(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt", device=device)
    
    zslmodule = ZSLmodule(
        args=args,
        data_path=data_path,
        r2id=r2id,
        e2id=e2id,
        device=device,
        dataset=graph_train_dataset
    ).to(device)
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
    print('Average steps per epoch is:', steps_per_epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_maximum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_warmup_epochs * steps_per_epoch // args.accumulate_grad_steps,
        T_mult=2,
        eta_min=args.lr_minimum
    )
    
    start_step = 0

    losses = deque([], steps_per_epoch)
    losses_struc = deque([], steps_per_epoch)
    losses_image = deque([], steps_per_epoch)
    losses_text = deque([], steps_per_epoch)
    losses_con = deque([], steps_per_epoch)
    
    print('Start Fusion Training!\n')
    epoch_counter = trange(start_step, args.epochs, ncols=0)

    for epoch in epoch_counter:
        model.train()
        model.model.train()
        for step, data in enumerate(dataloader):
            batch_size, n_id, adjs = data
            batch_rels = graph.edge_type[adjs.e_id]
            batch_data = graph_train_dataset.generate_batch(n_id, batch_rels)
            
            if torch.numel(batch_data['image']) == 0:
                batch_data['image'] = None
            else:
                batch_data['image'] = batch_data['image'].to(device)
            if torch.numel(batch_data['text']) == 0:
                batch_data['text'] = None
            else:
                batch_data['text'] = batch_data['text'].to(device)
            # batch_data['image'] = None
            # batch_data['text'] = None 
            batch_data['text_padding_mask'] = batch_data['text_padding_mask'].to(device)
            batch_data['rel_des'] = batch_data['rel_des'].to(device)
            batch_data['rel_des_padding_mask'] = batch_data['rel_des_padding_mask'].to(device)
            optimizer.zero_grad()
            if len(adjs.edge_index[0]) != 0:
                loss, info = model(
                    local_global_id={k: v.item() for k, v in zip(range(len(n_id)), n_id)},
                    edge_index=adjs.edge_index.to(device),
                    edge_type=graph.edge_type[adjs.e_id], 
                    batch=batch_data
                )
                loss = loss.to(device)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch * steps_per_epoch + step)

                losses.append(loss.item())
                epoch_counter.set_description("Epoch %d |loss: %.3f |gcn_loss: %.2f |image_loss: %.2f |text_loss: %.2f |con_loss: %.2f " % (
                    epoch + args.start_epoch + 1,
                    np.mean(losses), 
                    info['gcn_loss'], 
                    info['image_loss'], 
                    info['text_loss'],
                    info['contrastive_loss']
                    )
                )
                losses_struc.append(info['struct_loss'].item())
                try:
                    losses_image.append(info['image_loss'].item())
                except:
                    losses_image.append(0.0)
                try:
                    losses_text.append(info['text_loss'].item())
                except:
                    losses_text.append(0.0)
                try:
                    losses_con.append(info['contrastive_loss'].item())
                except:
                    losses_con.append(0.0)
            else: print(adjs)
        print(f'epoch{epoch + args.start_epoch + 1} loss is {np.mean(losses)}!')
        log_metrics = {
            'epoch' : epoch + args.start_epoch + 1,
            'whole loss' : np.mean(losses),
            'structure loss' : np.mean(losses_struc),
            'image loss' : np.mean(losses_image),
            'text loss' : np.mean(losses_text),
            'contrastive_loss' : np.mean(losses_con)
        }
        #logger.log(log_metrics)
        losses.clear()
        losses_struc.clear()
        losses_image.clear()
        losses_text.clear()
        losses_con.clear()
        # if not graph_train_dataset.config.text_only: 
        #     log_image = create_log_images(
        #         patch_predict_fn(model.model.M3AEmodel, args.patch_size, batch_data),
        #         mean=graph_train_dataset.image_mean, std=graph_train_dataset.image_std
        #     )
        #     logger.log({"image_prediction": wandb.Image(log_image)})

        if (epoch + args.start_epoch + 1) % args.save_epochs == 0:
            print(f'\n save model at epoch{epoch + args.start_epoch + 1}!')
            model.save_checkpoint(f"./saved_models/{args.dataset}/epoch{epoch + args.start_epoch + 1}_{args.saved_model_name}.ckpt")
            model.model.set_evaluate(True)
            ent_embs = generate_ent_embed(args, graph_train_dataset, model, device)
            rel_embs = generate_rel_embed(graph_train_dataset, model, None, device, 'seen')
            zslmodule.update_embed(ent_embs, rel_embs)
            zslmodule.train(model.model)
            for param in model.model.parameters():
                param.requires_grad = True
            model.model.set_evaluate(False)
    print('Finish Training\n')
    model.save_checkpoint(f"./saved_models/{args.saved_model_name}.ckpt")

def evaluate(args, ent_embs, rel_embs, e2id, r2id, model, mode='test'):
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    hits_at_k = [1, 3, 10]

    with open(os.path.join(data_path, f'{mode}/{mode}_candidates.json'), 'r') as f:
        test_candidates = json.load(f)

    print('Start evaluation!\n')
    model.eval()
    model.model.eval()
    model.model.set_evaluate(True)

    ranks = []
    #rela_counter = tqdm(range(len(test_candidates.keys())))
    for query in test_candidates.keys():
        temp_rank = []
        for e1_rel, tail_candidates in test_candidates[query].items():
            head, rela, _ = e1_rel.split("\t")
            #true = tail_candidates[0]
            head_id, rela_id = e2id[head], r2id[rela]
            head_emb = ent_embs[head_id]
            head_embs = head_emb.repeat(len(tail_candidates), 1)
            rela_emb = rel_embs[rela_id]
            rela_embs = rela_emb.repeat(len(tail_candidates), 1)
            tail_embs = torch.rand(len(tail_candidates), model.model.dim)
            for idx, tail in enumerate(tail_candidates):
                tail_embs[idx] = ent_embs[e2id[tail]]
            scores = model.evaluate(h=head_embs, r=rela_embs, t=tail_embs)
            p_score, n_score = scores[0], scores[1:]
            raw_ranks = torch.sum(n_score < p_score, dim=0, dtype=torch.long)
            num_ties = torch.sum(n_score == p_score, dim=0, dtype=torch.long)
            branks = raw_ranks + num_ties // 2
            temp_rank.append(branks + 1)
        ranks.extend(temp_rank)
        temp_mrr = sum([1.0/rank for rank in temp_rank]) / len(temp_rank)
        temp_hit_1 = sum([1.0 if rank <= 1 else 0.0 for rank in temp_rank]) / len(temp_rank)
        temp_hit_3 = sum([1.0 if rank <= 3 else 0.0 for rank in temp_rank]) / len(temp_rank)
        temp_hit_10 = sum([1.0 if rank <= 10 else 0.0 for rank in temp_rank]) / len(temp_rank)
        print("Relation: %s| Number %d | mrr: %.4f | hit1: %.4f | hit3: %.4f | hit10: %.4f " % (query,
                                                                                            len(test_candidates[query]),
                                                                                            temp_mrr, 
                                                                                            temp_hit_1, 
                                                                                            temp_hit_3, 
                                                                                            temp_hit_10))

    mrr = sum([1.0/rank for rank in ranks])/len(ranks)
    hits = []
    for k in hits_at_k:
        hits.append(sum([1.0 if rank <= k else 0.0 for rank in ranks]) / len(ranks))
    hits_at_1, hits_at_3, hits_at_10 = hits
    print(f'[Final Scores] '
            f'MRR: {mrr} \t'
            f'Hits@1: {hits_at_1} \t'
            f'Hits@3: {hits_at_3} \t'
            f'Hits@10: {hits_at_10}')

if __name__ == "__main__":
    args = read_options()
    if not args.evaluate:
        main(args)
    else:
        device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
        # device = 'cpu'
        data_path = osp.join('./origin_data', args.dataset)
        triples, mm_info, rel_des_file, e2id, r2id = load_appendix_data(data_path, mode='train')
        
        graph_train_dataset = MMKGDataset(
            config=MMKGDataset.get_default_config(),
            train_file='train_tasks_zsl.json',
            name=args.dataset,  
            root=data_path,
            mode='train',
            mm_info=mm_info,
            rel_des_file=rel_des_file
        )
        part_model = UnifiedModel(
            args=args,
            hidden_channels=200,
            dataset=graph_train_dataset,
            num_relations=graph_train_dataset.num_relations,
            noise_dim=args.noise_dim
        )
        model = NegativeSampling(
            args = args,
            whole_triples=triples,
            model= part_model,
            loss_fn = MarginLoss(margin=3.0),
            neg_ent = 1,
            sampling_mode = 'normal'
        ).to(device)
        zslmodule = ZSLmodule(
            args=args,
            data_path=data_path,
            r2id=r2id,
            e2id=e2id,
            device=device,
            dataset=graph_train_dataset
        ).to(device)
        
        if args.pretrained_model_name != '':
            print(f'Loading pretrained model:{args.pretrained_model_name}')
            # state_dict = torch.load(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt")
            # del state_dict['model.generate_fc_layer.weight_orig']
            # del state_dict['model.generate_fc_layer.weight_v']
            # model.load_state_dict(state_dict, map_location=device)
            model.load_checkpoint(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt", device=device)

        ent_embs = generate_ent_embed(args, graph_train_dataset, model, device)
        rel_embs = generate_rel_embed(graph_train_dataset, model, None, device, 'seen')

        with open('./temp_ent_embs.pkl', 'wb') as fout:
            pickle.dump(ent_embs, fout)
        with open('./temp_rel_embs.pkl', 'wb') as fout:
            pickle.dump(rel_embs, fout)
        # with open('./temp_ent_embs.pkl', 'rb') as fout:
        #     ent_embs = pickle.load(fout)
        # with open('./temp_rel_embs.pkl', 'rb') as fout:
        #     rel_embs = pickle.load(fout)
        #evaluate(args, ent_embs, rel_embs, e2id, r2id, model, mode='train')
        model.model.set_evaluate(True)
        for param in model.model.parameters():
            param.requires_grad = False
        zslmodule.update_embed(ent_embs, rel_embs)
        zslmodule.train(generate_model= model.model)
        zslmodule.eval(generate_model= model.model, mode="test", meta=True, load_pretrain=False)
        # relations = ["/sports/sports_position/players./american_football/football_historical_roster_position/position_s",
        #                                         "/military/military_combatant/military_conflicts./military/military_combatant_group/combatants",
        #                                         "/music/artist/track_contributions./music/track_contribution/role",
        #                                         "/film/film/featured_film_locations",
        #                                         "/music/performance_role/guest_performances./music/recording_contribution/performance_role"]
        # entity_pair_embs, rels, target_idx = zslmodule.generate_entity_pair_emb(relations, model.model)
        # entity_pair_embs = torch.cat(entity_pair_embs, dim = 0)
        # print(entity_pair_embs.shape)
        # np.savez(f'./RZSMAE_select_entpair_embs.npz', eM=entity_pair_embs, rl=rels, tidx=target_idx)