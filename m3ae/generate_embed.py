import os
import json
import dataclasses
import pprint
from functools import partial

import absl.app
import absl.flags
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb
import pickle
import einops

from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from tqdm.auto import tqdm, trange

from .data import ImageTextDataset, TextDataset
from .jax_utils import (
    JaxRNG, get_metrics, next_rng, accumulated_gradient,
    sync_state_across_devices
)
from .model import (
    MaskedMultimodalAutoencoder, extract_patches,
    merge_patches, cross_entropy_loss_and_accuracy,
    patch_mse_loss, M3AETrainState, mask_intersection, mask_not,
    mask_select, all_mask
)
from .model_pytorch import MaskedMultimodalAutoencoder_pytorch

from .utils import (
    WandBLogger, define_flags_with_default, get_user_flags,
    image_float2int, load_pickle, set_random_seed, create_log_images
)
from .vqgan import get_image_tokenizer

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict

FLAGS_DEF = define_flags_with_default(
    dataset_name='FB15K-237',
    seed=42,
    batch_size=2,
    accumulate_grad_steps=1,
    patch_size=16,
    discretized_image=False,
    image_tokenizer_type='maskgit',
    load_checkpoint="./m3ae/checkpoints/m3ae_small.pkl",
    m3ae=MaskedMultimodalAutoencoder.get_default_config(),
    data=ImageTextDataset.get_default_config(),
    unpaired_text_data=TextDataset.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=True,
)

FLAGS = absl.flags.FLAGS
# variant = get_user_flags(FLAGS, FLAGS_DEF)

# variant["jax_process_index"] = jax_process_index = jax.process_index()
# variant["jax_process_count"] = jax_process_count = jax.process_count()


# print(variant["jax_process_index"], variant["jax_process_count"])
def main(argv):
    dataset = ImageTextDataset(FLAGS.data)
    unpaired_text_dataset = TextDataset(FLAGS.unpaired_text_data)

    image_patch_dim = FLAGS.patch_size * FLAGS.patch_size * 3
    image_output_dim = image_patch_dim

    model = MaskedMultimodalAutoencoder_pytorch(
        text_vocab_size=dataset.vocab_size,
        config_updates=FLAGS.m3ae,
    )

    if FLAGS.load_checkpoint != '':
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        jax_weights = checkpoint_data['state']
        jax_config = checkpoint_data['variant']

    #load the pretrained weights
    for n, p in model.named_parameters():
        if n in ['cls_token', 'encoder_image_type_embedding', 'encoder_text_type_embedding']:
            p.data.copy_(torch.from_numpy(jax_weights.params['params'][n].copy()))
        elif n == 'image_embedding.weight':
            p.data.copy_(torch.from_numpy(jax_weights.params['params']['image_embedding']['kernel'].copy()).t())
        elif n == 'image_embedding.bias':
            p.data.copy_(torch.from_numpy(jax_weights.params['params']['image_embedding']['bias'].copy()).t())
        elif n == 'text_embedding.weight':
            p.data.copy_(torch.from_numpy(jax_weights.params['params']['text_embedding']['embedding'].copy()))
        elif n == 'encoder.layer_norm.weight':
            p.data.copy_(torch.from_numpy(jax_weights.params['params']['encoder']['LayerNorm_0']['scale'].copy()))
        elif n == 'encoder.layer_norm.bias':
            p.data.copy_(torch.from_numpy(jax_weights.params['params']['encoder']['LayerNorm_0']['bias'].copy()))
        elif n.startswith('encoder.blocks.'):
            block_num = n.split('encoder.blocks.')[1].split('.')[0]
            jax_block_weights = jax_weights.params['params']['encoder'][f"Block_{block_num}"]
            if n == f"encoder.blocks.{block_num}.layer_norm1.weight":
                v = jax_block_weights['LayerNorm_0']['scale']
            elif n == f"encoder.blocks.{block_num}.layer_norm1.bias":
                v = jax_block_weights['LayerNorm_0']['bias']
            elif n == f"encoder.blocks.{block_num}.layer_norm2.weight":
                v = jax_block_weights['LayerNorm_1']['scale']
            elif n == f"encoder.blocks.{block_num}.layer_norm2.bias":
                v = jax_block_weights['LayerNorm_1']['bias']
            elif n == f"encoder.blocks.{block_num}.attention.qkv_linear.weight":
                v = jax_block_weights['Attention_0']['Dense_0']['kernel']
            elif n == f"encoder.blocks.{block_num}.attention.qkv_linear.bias":
                v = jax_block_weights['Attention_0']['Dense_0']['bias']
            elif n == f"encoder.blocks.{block_num}.attention.fc.weight":
                v = jax_block_weights['Attention_0']['Dense_1']['kernel']
            elif n == f"encoder.blocks.{block_num}.attention.fc.bias":
                v = jax_block_weights['Attention_0']['Dense_1']['bias']
            elif n == f"encoder.blocks.{block_num}.transformer_mlp.fc1.weight":
                v = jax_block_weights['TransformerMLP_0']['fc1']['kernel']
            elif n == f"encoder.blocks.{block_num}.transformer_mlp.fc1.bias":
                v = jax_block_weights['TransformerMLP_0']['fc1']['bias']
            elif n == f"encoder.blocks.{block_num}.transformer_mlp.fc2.weight":
                v = jax_block_weights['TransformerMLP_0']['fc2']['kernel']
            elif n == f"encoder.blocks.{block_num}.transformer_mlp.fc2.bias":
                v = jax_block_weights['TransformerMLP_0']['fc2']['bias']
            else:
                raise False
            p.data.copy_(torch.from_numpy(v.copy()).t())
        else:
            raise False
        
    unpaired_text_dataset.random_start_offset = 0
    dataset.random_start_offset = 0

    #load the id of each entity
    data_path = os.path.join('./origin_data', FLAGS.dataset_name)
    ### load the entity to id json file first
    with open(data_path + "/entity2ids.json", 'r') as fin:
        ent_id = json.load(fin)
    ### load the entity with text description file 
    with open(data_path + "/entity2textlong.txt", 'r') as fin:
        entity_text = dict()
        for line in fin.readlines():
            if line[-1] == '\n':
                ent, text = line[:-1].split('\t')
            else:
                ent, text = line.split('\t')
            entity_text.update({ent : text})
        #### split the entity into image-pair category and text-only category
        paired_entity, unpaired_entity = [], []
        for filename in os.listdir(os.path.join(data_path, "images")):
            #entity = filename[1:]
            entity = '/' + filename.replace('.', '/')
            assert entity in entity_text.keys()
            paired_entity.append(entity)
        for ent in entity_text.keys():
            if ent not in paired_entity:
                unpaired_entity.append(ent)

    #Read image-text pair in each embedding
    embedding = [None] * len(ent_id)

    for idx, ent_name in tqdm(enumerate(paired_entity)):
        image, text_token, mask = dataset.__getitem__(idx)
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        image_patches = einops.rearrange(image, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=FLAGS.patch_size,
            p2=FLAGS.patch_size)
        text_token = torch.from_numpy(text_token).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            representation = model.forward_representation(image=image_patches, text=text_token, text_padding_mask=mask, deterministic=True)
        embedding[ent_id[ent_name]] = representation.numpy()

    #Read unpaired text entity in each embedding
    for idx, ent_name in tqdm(enumerate(unpaired_entity)):
        unpaired_text, unpaired_text_padding_mask = unpaired_text_dataset.__getitem__(idx)
        unpaired_text = torch.from_numpy(unpaired_text).unsqueeze(0)
        unpaired_text_padding_mask = torch.from_numpy(unpaired_text_padding_mask).unsqueeze(0)
        with torch.no_grad():
            representation = model.forward_representation(image=None, text=unpaired_text, text_padding_mask=unpaired_text_padding_mask, deterministic=True)
        embedding[ent_id[ent_name]] = representation.numpy()

    #print(embedding)
    with open(os.path.join(data_path, 'M3AE_embed.pkl'), 'wb') as fout:
        pickle.dump(embedding, fout)

if __name__ == '__main__':
    absl.app.run(main)