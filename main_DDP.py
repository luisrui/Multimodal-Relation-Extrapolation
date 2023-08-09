import os
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
     set_random_seed
)
from model import (
    MaskedMultimodalAutoencoder, extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy,
    mask_intersection, all_mask, mask_not
)
from data import (
    ImageTextDataset, TextDataset
)

def first_fusion_train(model, batch, args, gpu):
    image = batch['image']
    text = batch['text']
    text_padding_mask = batch['text_padding_mask']
    unpaired_text = batch['unpaired_text']
    unpaired_text_padding_mask = batch['unpaired_text_padding_mask']
    image_patches = extract_patches(image, args.patch_size)
    # Forward Propogation
    image_output, text_output, image_mask, text_mask = model(
        image_patches,
        text,
        text_padding_mask,
        deterministic=False,
    )
    _, unpaired_text_output, _, unpaired_text_mask = model(
        None,
        unpaired_text,
        unpaired_text_padding_mask,
        deterministic=False,
    )
    #Missing discretized image optimization
    image_loss = patch_mse_loss(
        image_output, image_patches,
        None if args.image_all_token_loss else image_mask
    )
    image_accuracy = 0.0

    text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
        text_output, text,  
        mask_intersection(
            all_mask(text) if args.text_all_token_loss else text_mask,
            mask_not(text_padding_mask)
        )
    )

    unpaired_text_loss, unpaired_text_accuracy = cross_entropy_loss_and_accuracy(
        unpaired_text_output, unpaired_text,
        mask_intersection(
            all_mask(unpaired_text) if args.text_all_token_loss else unpaired_text_mask,
            mask_not(unpaired_text_padding_mask)
        )
    )
    loss = (
        args.image_loss_weight * image_loss
        + args.text_loss_weight * text_loss
        + args.unpaired_text_loss_weight * unpaired_text_loss
    )
    average_text_length = torch.mean(torch.sum(mask_not(text_padding_mask), dim=-1))
    average_unpaired_text_length = torch.mean(torch.sum(mask_not(unpaired_text_padding_mask), dim=-1))
    info = dict(
        image_loss=image_loss,
        text_loss=text_loss,
        loss=loss,
        image_accuracy=image_accuracy,
        text_accuracy=text_accuracy,
        text_token_ratio=torch.mean(torch.sum(mask_not(text_padding_mask), dim=-1) / text_mask.shape[-1]),
        average_text_length=average_text_length,
    )
    if args.unpaired_text_loss_weight > 0.0:
        info['unpaired_text_loss'] = unpaired_text_loss
        info['unpaired_text_accuracy'] = unpaired_text_accuracy
        info['average_unpaired_text_length'] = average_unpaired_text_length
    return loss, info

    
def main(gpu, args):
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

    # logger = WandBLogger(
    #     config=FLAGS.logging,
    #     variant=variant,
    #     enable=FLAGS.log_all_worker or (process_index == 0),
    # )
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
        # if FLAGS.discretized_image:
            #     encoded_image = encode_image(state.tokenizer_params, image)
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
        torch.save(model, path=f"./saved_models/M3AE_{args.dataset}.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    args = read_options()
    os.environ['MASTER_ADDR'] = 'localhost'             
    os.environ['MASTER_PORT'] = '1113'
    args.world_size = args.gpus * args.nodes
    # torch.multiprocessing.set_start_method("spawn")
    mp.spawn(main, args=(args,), nprocs=args.world_size)