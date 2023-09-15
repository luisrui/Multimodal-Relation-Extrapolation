from io import BytesIO
import skimage.io
from skimage.color import gray2rgb, rgba2rgb
import numpy as np
from PIL import Image

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from tqdm import tqdm
import os
import json
import einops

from .submodule import (MLP, Transformer)
from .utils import get_transformer_by_config

def first_fusion_train(model, batch : dict, args : dict):
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

def extract_patches(image, patch_size):
    batch, height, width, channels = image.shape
    height, width = height // patch_size, width // patch_size
    x = image.view(batch, height, patch_size, width, patch_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch, height * width, patch_size**2 * channels)
    return x

def index_sequence(x, ids):
    return x[:, ids, ...]

def random_masking(x, keep_len, padding_mask=None):
    batch, length, dim = x.shape
    noise = torch.rand((length,), dtype=torch.float32)
    _, ids_shuffle = torch.sort(noise)
    _, ids_restore = torch.sort(ids_shuffle)
    kept = index_sequence(x, ids_shuffle[:keep_len])
    mask = torch.ones([batch, length], dtype=torch.float32)
    mask[:, :keep_len] = 0.0
    mask = index_sequence(mask, ids_restore)

    if padding_mask is None:
        return kept, mask, ids_restore

    padding_mask_kept = index_sequence(padding_mask, ids_shuffle[:keep_len])
    return kept, mask, ids_restore, padding_mask_kept

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.view(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
        embed_dim, torch.arange(length, dtype=torch.float32)
    )
    pos_embed = torch.unsqueeze(pos_embed, 0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, length, patch_size):
    grid_size_w = patch_size
    grid_size_h = length // patch_size
    assert grid_size_h * grid_size_w == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid).unsqueeze(0)
    return pos_embed

def mask_intersection(mask1, mask2):
    return torch.logical_and(mask1 > 0, mask2 > 0).to(torch.float32)

def mask_not(mask):
    return 1.0 - mask

def all_mask(x):
    return torch.ones(x.shape[:2])

def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = all_mask(tokens)
    device = logits.device
    valid = valid.to(device)
    valid_text_length = torch.max(torch.sum(valid, dim=-1), torch.tensor(1e-5))
    token_log_prob = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1).to(torch.int64)).squeeze()
    token_log_prob = torch.where(valid > 0.0, token_log_prob, torch.tensor(0.0).to(device))
    loss = -torch.mean(torch.sum(token_log_prob, dim=-1) / valid_text_length)
    correct = torch.where(
        valid > 0.0,
        torch.argmax(logits, dim=-1) == tokens,
        torch.tensor(False).to(device)
    )
    accuracy = torch.mean(torch.sum(correct, dim=-1) / valid_text_length)
    return loss, accuracy

def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = torch.sum(valid, dim=-1) / valid.shape[-1]
    device = patch_output.device
    return torch.mean(
        torch.mean(
            torch.where(
                valid > 0.0,
                torch.mean(torch.square(patch_target - patch_output), dim=-1),
                torch.tensor(0.0).to(device),
            ),
            dim=-1,
        ) / valid_ratio
    )



# @title PyTorch MaskedMultimodalAutoencoder
class MaskedMultimodalAutoencoder(nn.Module):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.model_type = 'small'
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    def __init__(self, text_vocab_size, patch_size, image_output_dim = 768, config_updates=None):
        super(MaskedMultimodalAutoencoder, self).__init__()
        self.text_vocab_size = text_vocab_size
        self.config = self.get_default_config(config_updates)
        assert self.text_vocab_size > 0

        self.patch_size = patch_size
        self.text_embedding = nn.Embedding(self.text_vocab_size, 
                                           self.config.emb_dim)
        self.text_embedding.weight.data.normal_(0.0, 1.0)
        self.image_output_dim = image_output_dim
        self.image_embedding = nn.Linear(image_output_dim, self.config.emb_dim)
        nn.init.xavier_uniform_(self.image_embedding.weight)
        
        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            )
            self.encoder_text_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            )
            self.decoder_image_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.dec_emb_dim).normal_(0.02)
            )
            self.decoder_text_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.dec_emb_dim).normal_(0.02)
            )

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
        )
        self.image_mask_embedding = nn.Parameter(
            torch.empty(1, 1, self.config.dec_emb_dim).normal_(0.02)
        )
        self.text_mask_embedding = nn.Parameter(
            torch.empty(1, 1, self.config.dec_emb_dim).normal_(0.02)
        )

        self.encoder = Transformer(
            emb_dim=self.config.emb_dim,
            depth=self.config.depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder = Transformer(
            emb_dim=self.config.dec_emb_dim,
            depth=self.config.dec_depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.dec_num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder_input_projection = nn.Linear(self.config.emb_dim, self.config.dec_emb_dim)
        nn.init.xavier_uniform_(self.decoder_input_projection.weight)

        self.decoder_image_output = MLP(
            self.config.dec_emb_dim,
            self.image_output_dim,
            self.config.output_head_depth,
            input_norm=self.config.output_head_depth > 0,
        )

        self.decoder_text_output = MLP(
            self.config.dec_emb_dim,
            self.text_vocab_size,
            self.config.output_head_depth,
            input_norm=self.config.output_head_depth > 0,
        )

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'encoder_text_type_embedding': self.encoder_text_type_embedding,
                'decoder_image_type_embedding': self.decoder_image_type_embedding,
                'decoder_text_type_embedding': self.decoder_text_type_embedding,
            }[name]
        else:
            return 0.0
        
    def get_model_device(self):
        return next(self.parameters()).device
    
    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.state_dict(), path + "M3AE")

    def forward_representation(self, image, text, text_padding_mask, deterministic=True):
        if image is not None:
            batch_size = image.shape[0]
        else:
            batch_size = text.shape[0]
        model_device = self.get_model_device()
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32, device=model_device)]
        if image is not None:
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1], self.patch_size).to(model_device)
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image.shape[1]), dtype=torch.float32, device=model_device))

        if text is not None:
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1]).to(model_device)
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)

        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.encoder(x, deterministic, padding_mask)

        cls_x = x[:, :1, :]

        return cls_x, x

    def forward_encoder(self, image, text, text_padding_mask, deterministic=False):
        if image is not None:
            batch_size = image.shape[0]
        else:
            batch_size = text.shape[0]
        model_device = self.get_model_device()
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32, device=model_device)]
        if image is not None:
            image_keep_length = int(
                image.shape[1] * (1.0 - self.config.image_mask_ratio)
            )
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1], self.patch_size).to(model_device)
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            image_x, image_mask, image_ids_restore = random_masking(
                image_x, image_keep_length
            )
            input_tensors.append(image_x.to(model_device))
            padding_masks.append(torch.zeros((batch_size, image_keep_length), dtype=torch.float32, device=model_device))
            image_mask = image_mask.to(model_device)
        else:
            image_mask = image_ids_restore = None
        
        if text is not None:
            text_keep_length = int(
                text.shape[1] * (1.0 - self.config.text_mask_ratio)
            )
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1]).to(model_device)
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            text_x, text_mask, text_ids_restore, text_padding_mask = random_masking(
                text_x,
                text_keep_length,
                text_padding_mask,
            )
            input_tensors.append(text_x.to(model_device))
            padding_masks.append(text_padding_mask.to(model_device))
            text_mask = text_mask.to(model_device)
        else:
            text_mask = text_ids_restore = text_padding_mask = None
        
        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.encoder(x, deterministic, padding_mask)

        cls_x = x[:, :1, :]
        if image is None:
            image_x = None
            text_x = x[:, 1:, :]
        elif text is None:
            image_x = x[:, 1:, :]
            text_x = None
        else:
            image_x = x[:, 1:image_keep_length + 1, :]
            text_x = x[:, image_keep_length + 1:, :]

        return cls_x, image_x, text_x, image_mask, text_mask, image_ids_restore, text_ids_restore
    
    def forward_decoder(
        self,
        cls_x,
        image_x,
        text_x,
        image_ids_restore,
        text_ids_restore,
        text_padding_mask,
        deterministic=False,                                                                                                                                                                                                   
    ):
        model_device = self.get_model_device()
        batch_size = cls_x.shape[0]
        input_tensors = [self.decoder_input_projection(cls_x)]
        padding_masks = [torch.zeros((batch_size,  1), dtype=torch.float32, device=model_device)]

        if image_x is not None:
            image_keep_length = int(
                image_ids_restore.shape[0] * (1.0 - self.config.image_mask_ratio)
            )
            image_x = self.decoder_input_projection(image_x)
            masked_image_x = self.image_mask_embedding.expand(
                batch_size,
                image_ids_restore.shape[0] - image_keep_length,
                self.config.dec_emb_dim,
            )
            image_x = index_sequence(
                torch.cat([image_x, masked_image_x], dim=1), image_ids_restore
            )
            image_x = (
                image_x.to(model_device)
                + get_2d_sincos_pos_embed(self.config.dec_emb_dim, image_ids_restore.shape[0], self.patch_size).to(model_device)
                + self.get_type_embedding('decoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image_ids_restore.shape[0]), dtype=torch.float32, device=model_device))

        if text_x is not None:
            text_keep_length = int(
                text_ids_restore.shape[0] * (1.0 - self.config.text_mask_ratio)
            )
            text_x = self.decoder_input_projection(text_x)
            masked_text_x = self.text_mask_embedding.expand(
                batch_size,
                text_ids_restore.shape[0] - text_keep_length,
                self.config.dec_emb_dim,
            )
            text_x = index_sequence(
                torch.cat([text_x, masked_text_x], dim=1), text_ids_restore
            )
            text_x = (
                text_x.to(model_device)
                + get_1d_sincos_pos_embed(self.config.dec_emb_dim, text_ids_restore.shape[0]).to(model_device)
                + self.get_type_embedding('decoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask.to(model_device))

        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.decoder(x, deterministic, padding_mask)

        cls_x = x[:, :1, :]
        if image_x is None:
            image_output = None
            text_output = self.decoder_text_output(x[:, 1:, :])
        elif text_x is None:
            image_output = self.decoder_image_output(x[:, 1:, :])
            text_output = None
        else:
            image_output = self.decoder_image_output(x[:, 1:image_ids_restore.shape[0] + 1, :])
            text_output = self.decoder_text_output(x[:, image_ids_restore.shape[0] + 1:, :])

        return image_output, text_output
    
    def __call__(self, image, text, text_padding_mask, deterministic=False):
        (
            cls_x,
            image_x,
            text_x,
            image_mask,
            text_mask,
            image_ids_restore,
            text_ids_restore,
        ) = self.forward_encoder(image, text, text_padding_mask, deterministic)
        image_output, text_output = self.forward_decoder(
            cls_x,
            image_x,
            text_x,
            image_ids_restore,
            text_ids_restore,
            text_padding_mask,
            deterministic,
        )
        return image_output, text_output, image_mask, text_mask

class UnifiedModel(nn.Module):
    def __init__(self, args, hidden_channels, dataset, num_relations):
        super(UnifiedModel, self).__init__()
        self.M3AEmodel = MaskedMultimodalAutoencoder(
            text_vocab_size=dataset.vocab_size,
            patch_size=args.patch_size,
            image_output_dim=args.patch_size * args.patch_size * 3,
            config_updates=ConfigDict(dict(model_type=args.model_type, image_mask_ratio=args.image_mask_ratio, text_mask_ratio=args.text_mask_ratio)),
        )
        self.is_evaluate = args.evaluate
        self.patch_size = args.patch_size
        #self.mm_info = mm_info

        self.paired_tokenizer_max_length = dataset.config.tokenizer_max_length
        self.token_num = dataset.config.unpaired_tokenizer_max_length
        self.transform_image = dataset.transform_image
        self.tokenizer = dataset.tokenizer
        
        self.num_relations = num_relations
        self.num_nodes = dataset.num_nodes
        self.reduced_dim = self.M3AEmodel.config.emb_dim
        self.dim = args.emb_dim

        #self.node_embedding = nn.Embedding(self.num_nodes, self.reduced_dim)
        self.dim_reduce1 = nn.Linear(self.token_num, 1) # 320 -> 1
        self.conv1 = RGCNConv(in_channels=self.reduced_dim, out_channels=hidden_channels, num_relations=self.num_relations, num_bases=30)
        self.conv2 = RGCNConv(in_channels=hidden_channels, out_channels=self.dim, num_relations=self.num_relations, num_bases=30)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def get_model_device(self):
        return next(self.parameters()).device
    
    def get_unmasked_features(self, x):
        new_matched_features = []
        model_device = self.get_model_device()
        for node_info in x:
            if len(node_info) == 3:
                image, text, text_padding_mask = node_info
                # image, text, text_padding_mask = self.mm_info[node]
                image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
                image_patches = einops.rearrange(image, 
                    'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                    p1=self.patch_size,
                    p2=self.patch_size).to(model_device)
                text_token = torch.from_numpy(text).unsqueeze(0).to(model_device)
                mask = torch.from_numpy(text_padding_mask).unsqueeze(0).to(model_device)
                #with torch.enable_grad():
                with torch.no_grad():
                    representation = self.M3AEmodel.forward_representation(image=image_patches, text=text_token, text_padding_mask=mask, deterministic=True)
            else:
                unpaired_text, unpaired_text_padding_mask = node_info
                unpaired_text = torch.from_numpy(unpaired_text).unsqueeze(0).to(model_device)
                unpaired_text_padding_mask = torch.from_numpy(unpaired_text_padding_mask).unsqueeze(0).to(model_device)
                #with torch.enable_grad():
                with torch.no_grad():
                    representation = self.M3AEmodel.forward_representation(image=None, text=unpaired_text, text_padding_mask=unpaired_text_padding_mask, deterministic=True)
            if new_matched_features.__len__()!=0:
                assert representation.shape == new_matched_features[-1].shape
            new_matched_features.append(representation)
        return torch.stack(new_matched_features, dim=0).to(model_device)

    def gcn_forward_encoder(self, x, edge_index, edge_type):
        #x = self.dim_reduce1(x.transpose(-2, -1))
        #x = x.transpose(-2, -1)
        x = x.view(x.shape[0], -1)
        x = self.conv1(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.conv2(x, edge_index, edge_type)
        return x
    
    def forward(self, edge_index, edge_type, batch, deterministic=False):
        image = batch['image']
        text = batch['text']
        text_padding_mask = batch['text_padding_mask']
        if image is not None:
            image_patches = extract_patches(image, self.patch_size)
        else:
            image_patches = None
        # if image_patches is None and text is None:
        #     cls_x = self.node_embedding(n_id)
        # else:
        cls_x, x = self.M3AEmodel.forward_representation(
            image=image_patches,  text=text, text_padding_mask=text_padding_mask, deterministic=True
        )

        x_ecd = self.gcn_forward_encoder(cls_x, edge_index, edge_type) 
        
        # batch_output = dict(
        #     image_output=0,
        #     text_output=0,
        #     image_mask=0,
        #     text_mask=0
        # )
        # return x_ecd, batch_output
    
        if not self.is_evaluate:
            image_output, text_output, image_mask, text_mask = self.M3AEmodel(
                image_patches, 
                text, 
                text_padding_mask, 
                deterministic
            )
            batch_output = dict(
                image_output=image_output,
                text_output=text_output,
                image_mask=image_mask,
                text_mask=text_mask
            )
            return x_ecd, batch_output
        else:
            return x_ecd
