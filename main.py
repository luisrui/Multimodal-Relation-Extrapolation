import os
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
    MaskedMultimodalAutoencoder, extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy,
    mask_intersection, all_mask, mask_not, UnifiedModel, first_fusion_train
)
from module.data import (
    ImageTextDataset, TextDataset, MMKGDataset
)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    # Instantiation of multimodal Graph Dataset
    data_path = os.path.join('./origin_data', args.dataset)
    graph_dataset = MMKGDataset(config=MMKGDataset.get_default_config(), name='FB15K-237', root='./origin_data/FB15K-237', device=device)
    #graph_dataset.generate_batch(torch.arange(1, 100))
    model = UnifiedModel(args=args,
                         hidden_channels=400,
                         dataset=graph_dataset,
                         num_relations=237)
    num_nodes = graph_dataset.num_nodes
    node_list_all = torch.arange(0, num_nodes)
    # dataset = ImageTextDataset(ImageTextDataset.get_default_config(), 0)
    # paired_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=args.dataloader_n_workers,
    #     prefetch_factor=2,
    #     persistent_workers=True,
    # )
    # image_patch_dim = args.patch_size * args.patch_size * 3
    # image_sequence_length = (dataset.config.image_size // args.patch_size) ** 2

    # unpaired_text_dataset = TextDataset(TextDataset.get_default_config(), 0)
    # unpaired_text_dataloader = torch.utils.data.DataLoader(
    #     unpaired_text_dataset,
    #     batch_size=args.batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=args.dataloader_n_workers,
    #     prefetch_factor=2,
    #     persistent_workers=True,
    # )

    # steps_per_epoch = int(len(dataset) / args.batch_size)
    # total_steps = steps_per_epoch * args.m3ae_epochs
    # tokenizer_params, encode_image, decode_image, image_vocab_size = (
    #     None, None, None, -1
    # )
    # image_output_dim = image_patch_dim

    # model = MaskedMultimodalAutoencoder(
    #     text_vocab_size=dataset.vocab_size,
    #     image_output_dim=image_output_dim,
    #     config_updates=ConfigDict(dict(model_type=args.model_type, image_mask_ratio=args.image_mask_ratio, text_mask_ratio=args.text_mask_ratio)),
    # ).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=args.lr_warmup_epochs * steps_per_epoch // args.accumulate_grad_steps,
    #     T_mult=1,
    #     eta_min=args.lr_minimum
    # )

    # start_step = 0
    # step_counter = trange(start_step, total_steps, ncols=0)

    # def infinite_iterator(iterator):
    #     inf_iterator = itertools.cycle(iterator)
    #     while True:
    #         yield next(inf_iterator)
    # paired_dataloader = infinite_iterator(paired_dataloader)
    # unpaired_text_dataloader = infinite_iterator(unpaired_text_dataloader)

    # losses = deque([], 100)
    # for step, (image, text, text_padding_mask), (unpaired_text, unpaired_text_padding_mask) in zip(step_counter, paired_dataloader, unpaired_text_dataloader):
    #     batch = {}
    #     batch['image'] = image.to(torch.float32)
    #     batch['text'] = text.to(torch.int32)
    #     batch['text_padding_mask'] = text_padding_mask.to(torch.float32)
    #     batch['unpaired_text'] = unpaired_text.to(torch.int32)
    #     batch['unpaired_text_padding_mask'] = unpaired_text_padding_mask.to(torch.float32)
    #     # if FLAGS.discretized_image:
    #         #     encoded_image = encode_image(state.tokenizer_params, image)
    #     for key, value in batch.items():
    #         batch[key] = value.to(device)
    #     loss, info = first_fusion_train(model, batch, args)
    #     loss = loss.to(device)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     losses.append(loss.item())
    #     step_counter.set_description("Step %d | loss: %f | textloss: %f | imageloss: %f | textacc: %f | imageacc: %f" % (step, 
    #                                                                                                                      np.mean(losses), 
    #                                                                                                                      info['text_loss'], 
    #                                                                                                                      info['image_loss'], 
    #                                                                                                                      info['text_accuracy'], 
    #                                                                                                                      info['image_accuracy']))

    #     if (step+1) % steps_per_epoch == 0:
    #         scheduler.step()
    
    # model.save(path=f"./saved_models/M3AE_{args.dataset}.pth")
    # image_patch_dim = args.patch_size * args.patch_size * 3
    # image_output_dim = image_patch_dim

    # data_path = os.path.join('./origin_data', args.dataset)
    # predict_model = torch.load(f"./saved_models/M3AE_{args.dataset}.pth")
    # unpaired_text_dataset.random_start_offset = 0
    # dataset.random_start_offset = 0
    # print('Generate pretraining embeddings')
    # generate_m3ae_embed(data_path, args, predict_model.module.to('cpu'), dataset, unpaired_text_dataset)

if __name__ == "__main__":
    args = read_options()
    main(args)