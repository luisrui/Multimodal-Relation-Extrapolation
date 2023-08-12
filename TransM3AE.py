import os
import json
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm.auto import tqdm, trange
from ml_collections import ConfigDict
from collections import deque
import numpy as np
import itertools

from args import read_options
from utils import (
    set_random_seed, generate_m3ae_embed
)
from model import (
    MaskedMultimodalAutoencoder, extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy,
    mask_intersection, all_mask, mask_not
)
from data import (
    ImageTextDataset, TextDataset
)
from TransAE.module import module as transae
from TransAE.dataloader.PyTorchTrainDataLoader import PyTorchTrainDataLoader
from main_DDP import first_fusion_train

def m3aeFusion(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    ) 
    torch.cuda.set_device(gpu)
    assert args.batch_size % args.gpus == 0
    process_index = dist.get_rank()
    process_batch_size = args.batch_size // args.world_size
    device_batch_size = process_batch_size // args.gpus
    lr_scale = args.batch_size / 256
    n_devices = torch.cuda.device_count()
    assert process_batch_size % n_devices == 0

    set_random_seed(args.seed * (process_index + 1))

    dataset = ImageTextDataset(ImageTextDataset.get_default_config(), process_index / args.world_size)
    image_text_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    paired_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=process_batch_size,
        drop_last=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        sampler=image_text_sampler
    )
    image_patch_dim = args.patch_size * args.patch_size * 3
    image_sequence_length = (dataset.config.image_size // args.patch_size) ** 2

    unpaired_text_dataset = TextDataset(TextDataset.get_default_config(), process_index / args.world_size)
    unpaired_text_sampler = torch.utils.data.distributed.DistributedSampler(
        unpaired_text_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    unpaired_text_dataloader = torch.utils.data.DataLoader(
        unpaired_text_dataset,
        batch_size=process_batch_size,
        drop_last=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        sampler=unpaired_text_sampler
    )

    steps_per_epoch = int(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.m3ae_epochs

    tokenizer_params, encode_image, decode_image, image_vocab_size = (
        None, None, None, -1
    )
    image_output_dim = image_patch_dim

    model = MaskedMultimodalAutoencoder(
        text_vocab_size=dataset.vocab_size,
        image_output_dim=image_output_dim,
        config_updates=ConfigDict(dict(model_type=args.model_type, image_mask_ratio=args.image_mask_ratio, text_mask_ratio=args.text_mask_ratio)),
    ).to(gpu)

    model = DDP(model, device_ids=[gpu])
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
    paired_dataloader = infinite_iterator(paired_dataloader)
    unpaired_text_dataloader = infinite_iterator(unpaired_text_dataloader)

    losses = deque([], 100)
    for step, (image, text, text_padding_mask), (unpaired_text, unpaired_text_padding_mask) in zip(step_counter, paired_dataloader, unpaired_text_dataloader):
        batch = {}
        batch['image'] = image.to(torch.float32)
        batch['text'] = text.to(torch.int32)
        batch['text_padding_mask'] = text_padding_mask.to(torch.float32)
        batch['unpaired_text'] = unpaired_text.to(torch.int32)
        batch['unpaired_text_padding_mask'] = unpaired_text_padding_mask.to(torch.float32)

        for key, value in batch.items():
            batch[key] = value.to(gpu)
        loss, info = first_fusion_train(model, batch, args, gpu)
        loss = loss.to(gpu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= args.world_size
        losses.append(loss.item())
        step_counter.set_description("Step %d | loss: %f | textloss: %f | imageloss: %f | textacc: %f" % (step, 
                                                                                                        np.mean(losses), 
                                                                                                        info['text_loss'], 
                                                                                                        info['image_loss'], 
                                                                                                        info['text_accuracy']))

        if (step+1) % steps_per_epoch == 0:
            scheduler.step()
    
    if gpu == 0:
        torch.save(model, f"./saved_models/M3AE_{args.dataset}.pth")
    # Train the TransAE model using the pretrained M3AE embeddings
    if gpu == 0:
        image_patch_dim = args.patch_size * args.patch_size * 3
        image_output_dim = image_patch_dim

        data_path = os.path.join('./origin_data', args.dataset)
        predict_model = torch.load(f"./saved_models/M3AE_{args.dataset}.pth")
        unpaired_text_dataset.random_start_offset = 0
        dataset.random_start_offset = 0
        print('Generate pretraining embeddings')
        generate_m3ae_embed(data_path, args, predict_model.module.to('cpu'), dataset, unpaired_text_dataset)

    dist.destroy_process_group()

def TransAE(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = f"./origin_data/{args.dataset}/"
    with open(os.path.join(data_path, "entity2ids.json"), 'r') as fin:
        ent_id = json.load(fin)
    with open(os.path.join(data_path, "M3AE_embed.pkl"), 'rb') as fin:
        m3ae_tokens = pickle.load(fin)
    with open(os.path.join(data_path, 'entity2id.txt'), 'r') as fp:
        ents_count = int(fp.readline()[:-1])
    with open(os.path.join(data_path, 'relation2id.txt'), 'r') as fp:
        rels_count = int(fp.readline()[:-1])

    transe = transae.TransE(
        dataset_name=args.dataset,
        m3ae_token=m3ae_tokens,
        device = device,
        ent_tot = ents_count,
        rel_tot = rels_count,
        ent_id = ent_id,
        dim = 200, 
        p_norm = 1, 
        norm_flag = True,
    )
    
    train_dataloader = PyTorchTrainDataLoader(
        in_path = f'./TransAE/data/{args.dataset}/', 
        nbatches = 1000,
        threads = 8, 
        sampling_mode = "normal", 
        bern_flag = 1, 
        filter_flag = 1, 
        neg_ent = 25,
        neg_rel = 25
    )

    if os.path.exists(f'./saved_models/{args.dataset}.ckpt'):
        transe.load_checkpoint(f'./saved_models/{args.dataset}.ckpt')
    
    model = transae.NegativeSampling(
            model = transe, 
            loss = transae.MarginLoss(margin = 5.0),
            batch_size = train_dataloader.get_batch_size()
        )
    
    transe.to(device)
    trainer = transae.Trainer(model = model, data_loader = train_dataloader, train_times = 500, alpha = 0.5, use_gpu = True)
    trainer.run()
    transe.save_checkpoint(f'./saved_models/{args.dataset}.ckpt')

    # Save the mapped embedddings
    rel_embed = transe.rel_embeddings.weight.data.cpu()
    index = torch.tensor([i for i in range(ents_count)]).to(device)
    ent_embed, _ = transe.ent_embeddings(index)
    ent_embed = ent_embed.detach().cpu().numpy()

    np.savez('./origin_data/' + args.dataset + "/TransM3AE_embed.npz", rM=rel_embed, eM=ent_embed)
    print('Success Generated the Embeddings of TransM3AE!')
    
if __name__ == "__main__":
    args = read_options()
    os.environ['MASTER_ADDR'] = 'localhost'             
    os.environ['MASTER_PORT'] = '1113'
    args.world_size = args.gpus * args.nodes
    mp.spawn(m3aeFusion, args=(args,), nprocs=args.world_size) 
    #To-Do: Make the Train TransAE procedure into a DDP version
    TransAE(args)
    