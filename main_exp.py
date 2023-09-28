import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os import path as osp
import torch
import torch.nn as nn

from tqdm.auto import tqdm, trange
from ml_collections import ConfigDict
from collections import deque
import numpy as np
import wandb
import json

from args_exp import read_options
from module.utils import (
    set_random_seed, load_appendix_data, WandBLogger, create_log_images
)
from module.model import ExpModel, patch_predict_fn
from module.data import MultiModalKnowledgeGraphDataset
from module.NegativeSamplingEXP import NegativeSampling
from module.loss import MarginLoss, SigmoidLoss

def main(args):
    # logger = WandBLogger(
    #     config=WandBLogger.get_default_config(),
    #     variant=args,
    # )
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    triples, mm_info, rel_des_file, e2id, r2id = load_appendix_data(data_path, mode='train')
    print('Start dataset preprocessing!')
    graph_train_dataset = MultiModalKnowledgeGraphDataset(
        config=MultiModalKnowledgeGraphDataset.get_default_config(),
        e2id=e2id,
        r2id=r2id,
        triples=triples,
        mm_info=mm_info,  
        rel_des_file=rel_des_file
    )
    print('Entity Number:', graph_train_dataset.num_nodes)
    print('Finish dataset preprocessing!')

    print('Start Model Instantiation!')
    part_model = ExpModel(
        args=args,
        dataset=graph_train_dataset,
    )
    model = NegativeSampling(
        args = args,
        whole_triples=triples,
        model= part_model,
        loss_fn = SigmoidLoss(),
        neg_ent = 10,
        sampling_mode = 'normal'
    ).to(device)
    if args.pretrained_model_name != '':
        print(f'Loading pretrained model:{args.pretrained_model_name}')
        model.load_checkpoint(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt", device=device)
    print('Finish Model Instantiation!')

    dataloader = torch.utils.data.DataLoader(
        graph_train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        shuffle=True
    )

    steps_per_epoch = len(dataloader)
    print('Average steps per epoch is:', steps_per_epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_maximum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_warmup_epochs * steps_per_epoch // args.accumulate_grad_steps,
        T_mult=1,
        eta_min=args.lr_minimum
    )

    start_step = 0
    losses = deque([], steps_per_epoch)
    losses_struc = deque([], steps_per_epoch)
    losses_image = deque([], steps_per_epoch)
    losses_text = deque([], steps_per_epoch)
    losses_con = deque([], steps_per_epoch)

    print('Start Fusion Training!\n')
    model.train()
    epoch_counter = trange(start_step, args.epochs, ncols=0)

    for epoch in epoch_counter:
        for step, data in enumerate(dataloader):
            batch_data = dict()
            triple, image_head, text_head, text_pad_mask_head, rel_des, rel_des_pad_mask = data

            batch_data['triples'] = triple.to(device)
            batch_data['image'] = image_head.to(device)
            batch_data['text'] = text_head.to(device)
            batch_data['text_padding_mask'] = text_pad_mask_head.to(device)
            batch_data['rel_des'] = rel_des.to(device)
            batch_data['rel_des_padding_mask'] = rel_des_pad_mask.to(device)

            optimizer.zero_grad()
            loss, info = model(batch_data)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            epoch_counter.set_description("epoch %d | loss: %f | struct_loss: %.2f | image_loss: %.2f | text_loss: %.2f |con_loss: %.2f" % (
                epoch + args.start_epoch + 1, 
                np.mean(losses), 
                info['struct_loss'], 
                info['image_loss'], 
                info['text_loss'],
                info['contrastive_loss']
                )
            )
            losses_struc.append(info['struct_loss'].item())
            losses_image.append(info['image_loss'].item())
            try:
                losses_text.append(info['text_loss'].item())
            except:
                losses_text.append(0.0)
            try:
                losses_con.append(info['contrastive_loss'].item())
            except:
                losses_con.append(0.0)
        log_metrics = {
            'epoch' : epoch + args.start_epoch + 1,
            'whole loss' : np.mean(losses),
            'structure loss' : np.mean(losses_struc),
            'image loss' : np.mean(losses_image),
            'text loss' : np.mean(losses_text),
            'contrastive_loss' : np.mean(losses_con)
        }
        print(f'epoch{epoch + args.start_epoch + 1} loss is {np.mean(losses)}!')
        #logger.log(log_metrics)
        losses.clear()
        losses_struc.clear()
        losses_image.clear()
        losses_text.clear()
        losses_con.clear()
        if not graph_train_dataset.config.text_only: 
            log_image = create_log_images(
                patch_predict_fn(model.model.M3AEmodel, args.patch_size, batch_data),
                mean=graph_train_dataset.image_mean, std=graph_train_dataset.image_std
            )
            #logger.log({"image_prediction": wandb.Image(log_image)})
        if (epoch + 1) % args.save_epochs == 0:
            print(f'\n save model at epoch{epoch + args.start_epoch + 1}!')
            model.save_checkpoint(f"./saved_models/{args.dataset}/epoch{epoch + args.start_epoch + 1}_{args.saved_model_name}.ckpt")
            model.model.set_evaluate(True)
            evaluate(args, e2id, r2id, model, device)
            model.model.set_evaluate(False)
    print('Finish Training\n')
    model.save_checkpoint(f"./saved_models/{args.saved_model_name}.ckpt")

def evaluate(args, e2id, r2id, model, device, mode='test'):
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = osp.join('./origin_data', args.dataset)
    hits_at_k = [1, 3, 10]

    triples, mm_info, rel_des_file, e2id, r2id = load_appendix_data(data_path, mode='test')
    graph_test_dataset = MultiModalKnowledgeGraphDataset(
        config=MultiModalKnowledgeGraphDataset.get_default_config(),
        e2id=e2id,
        r2id=r2id,
        triples=triples,
        mm_info=mm_info,  
        rel_des_file=rel_des_file
    )
    with open(os.path.join(data_path, f'{mode}/{mode}_candidates.json'), 'r') as f:
        test_candidates = json.load(f)

    print('Start evaluation!')
    model.eval()
    batch_size = 256

    ranks = []
    #rela_counter = tqdm(range(len(test_candidates.keys())))
    for query in test_candidates.keys():
        temp_rank, hs, rs, ts = [], [], [], []
        head_embs = torch.rand((len(test_candidates[query].keys()), args.emb_dim))
        rel_embs = torch.rand((len(test_candidates[query].keys()), args.emb_dim))
        for e1_rel in test_candidates[query].keys():
            head, rela, tail = e1_rel.split("\t")
            head_id, rela_id, tail_id = e2id[head], r2id[rela], e2id[tail]
            hs.append(head_id)
            rs.append(rela_id)
            ts.append(tail_id)
        num_epoch = len(hs) // batch_size if len(hs) % batch_size != 0 else len(hs) // batch_size -1
        for i in range(num_epoch):
            if (i+1) * batch_size <= len(hs):
                batch_data = graph_test_dataset.get_batch([hs[i*batch_size:(i+1)*batch_size], rs[i*batch_size:(i+1)*batch_size], ts[i*batch_size:(i+1)*batch_size]])
            else:
                batch_data = graph_test_dataset.get_batch([hs[i*batch_size:], rs[i*batch_size:], ts[i*batch_size:]])
            for key, item in batch_data.items():
                batch_data[key] = item.to(device)
            with torch.no_grad():
                head_embs_part, rel_embs_part = model.model(batch_data)
            if (i+1) * batch_size <= len(hs):
                head_embs[i*batch_size:(i+1)*batch_size], rel_embs[i*batch_size:(i+1)*batch_size] = head_embs_part, rel_embs_part
            else:
                head_embs[i*batch_size], rel_embs[i*batch_size] = head_embs_part, rel_embs_part

        for idx, (e1_rel, tail_candidates) in enumerate(test_candidates[query].items()):
            h_emb = head_embs[idx]
            h_embs = h_emb.repeat(len(tail_candidates), 1).to(device)
            r_emb = rel_embs[idx]
            r_embs = r_emb.repeat(len(tail_candidates), 1).to(device)
            tail_ids = torch.tensor(np.array([e2id[tail] for tail in tail_candidates]), dtype = torch.int64).to(device)
            with torch.no_grad():
                t_embs = model.ent_embedding(tail_ids)
                scores = model._calc(h=h_embs, r=r_embs, t=t_embs)
            p_score, n_score = scores[0], scores[1:]
            raw_ranks = torch.sum(n_score > p_score, dim=0, dtype=torch.long)
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
        #device = 'cpu'
        data_path = osp.join('./origin_data', args.dataset)
        triples, mm_info, rel_des_file, e2id, r2id = load_appendix_data(data_path, mode='train')

        graph_dataset = MultiModalKnowledgeGraphDataset(
            config=MultiModalKnowledgeGraphDataset.get_default_config(),
            e2id=e2id,
            r2id=r2id,
            triples=triples,
            mm_info=mm_info,  
            rel_des_file=rel_des_file
        )
        part_model = ExpModel(
            args=args,
            dataset=graph_dataset,
        )
        model = NegativeSampling(
            args = args,
            whole_triples=None,
            model= part_model,
            loss_fn = SigmoidLoss(),
            neg_ent = 1,
            sampling_mode = 'normal'
        ).to(device)
        if args.pretrained_model_name != '':
            try:
                print(f'Loading pretrained model:{args.pretrained_model_name}')
                model.load_checkpoint(f"./saved_models/{args.dataset}/{args.pretrained_model_name}.ckpt", device=device)
            except:
                print('Pretrained model invalid!')
                exit()
        evaluate(args, e2id, r2id, model, device)
    # model = torch.load(f"./saved_models/Unimodal_{args.dataset}.pth")
    # model.save_checkpoint(f"./saved_models/Unimodal_{args.dataset}.ckpt")