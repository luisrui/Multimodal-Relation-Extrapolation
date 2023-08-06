import torch 
import torch.nn as nn
from torch.nn import functional as F
#!pip install ml_collections
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
import pickle
import einops 
import pprint
import transformers
import numpy as np

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

def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid).unsqueeze(0)
    return pos_embed

# @title Model size config
def get_transformer_by_config(model_type, config):
    if model_type == 'small':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'base':
        config.emb_dim = 768
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 12
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'large':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'huge':
        config.emb_dim = 1280
        config.dec_emb_dim = 512
        config.depth = 32
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'debug':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 2
        config.dec_depth = 2
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    else:
        raise ValueError('Unsupported model type!')

def mask_not(mask):
    return 1.0 - mask

def all_mask(x):
    return torch.ones(x.shape[:2])

def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = all_mask(tokens)
    valid_text_length = torch.max(torch.sum(valid, dim=-1), torch.tensor(1e-5))

    token_log_prob = jnp.squeeze

def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = torch.sum(valid, dim=-1) / valid.shape[-1]
    return torch.mean(
        torch.mean(
            torch.where(
                valid > 0.0,
                torch.mean(torch.square(patch_target - patch_output), dim=-1),
                torch.tensor(0.0),
            ),
            dim=-1,
        ) / valid_ratio
    )
# @title PyTorch transformer definition
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, depth, input_norm=True):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.input_norm = input_norm

        if self.input_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = inputs
        if self.input_norm:
            x = self.layer_norm(x)

        for i in range(self.depth):
            y = self.dense_layers[i](x)
            y = F.gelu(y)
            y = nn.LayerNorm(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = self.output_layer(x)
        return x


class DropPath(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super(DropPath, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, input, deterministic=False):
        if deterministic:
            return input

        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32)
        random_tensor = random_tensor.floor()
        return input.div(keep_prob) * random_tensor


class TransformerMLP(nn.Module):
    def __init__(self, dim=256, out_dim=256, dropout=0.0, kernel_init=None):
        super(TransformerMLP, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_

        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, out_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, deterministic=False):
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, use_bias=False, att_drop=0, proj_drop=0, kernel_init=None):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.att_drop = att_drop
        self.proj_drop = proj_drop
        self.scale = (dim // num_heads) ** -0.5
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_

        self.qkv_linear = nn.Linear(dim, dim * 3, bias=use_bias)
        self.fc = nn.Linear(dim, dim)
        self.att_drop_layer = nn.Dropout(att_drop)
        self.proj_drop_layer = nn.Dropout(proj_drop)

    def forward(self, inputs, deterministic=False, padding_mask=None):
        batch, n, channels = inputs.shape
        qkv = self.qkv_linear(inputs)
        qkv = qkv.view(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask = padding_mask.expand(attention.shape)
            attention = torch.where(padding_mask > 0, torch.tensor(-1e7), attention)

        attention = F.softmax(attention, dim=-1)
        attention = self.att_drop_layer(attention)

        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).reshape(batch, n, channels)
        x = self.fc(x)
        x = self.proj_drop_layer(x)
        return x


class Block(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, mlp_ratio=4, att_drop=0.0, drop=0.0, drop_path=0.0):
        super(Block, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.att_drop = att_drop
        self.drop = drop
        self.drop_path = drop_path

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.attention = Attention(emb_dim, num_heads, True, att_drop, drop)
        self.drop_path_layer1 = DropPath(drop_path)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.transformer_mlp = TransformerMLP(emb_dim, emb_dim, drop)
        self.drop_path_layer2 = DropPath(drop_path)

    def forward(self, inputs, deterministic=False, padding_mask=None):
        x = self.layer_norm1(inputs)
        x = self.attention(x, deterministic, padding_mask)
        x = self.drop_path_layer1(x)
        inputs = inputs + x

        x = self.layer_norm2(inputs)
        x = self.transformer_mlp(x, deterministic)
        x = self.drop_path_layer2(x)
        return inputs + x


class Transformer(nn.Module):
    def __init__(self, emb_dim=1024, depth=24, att_drop=0, drop=0, drop_path=0, num_heads=16, mlp_ratio=4):
        super(Transformer, self).__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.att_drop = att_drop
        self.drop = drop
        self.drop_path = drop_path
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.blocks = nn.ModuleList([
            Block(emb_dim, num_heads, mlp_ratio, att_drop, drop, drop_path)
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, deterministic=False, padding_mask=None):
        for block in self.blocks:
            x = block(x, deterministic, padding_mask)

        x = self.layer_norm(x)
        return x

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

    def __init__(self, text_vocab_size, image_output_dim = 768, config_updates=None):
        super(MaskedMultimodalAutoencoder, self).__init__()
        self.text_vocab_size = text_vocab_size
        self.config = self.get_default_config(config_updates)
        assert self.text_vocab_size > 0

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

    def forward_representation(self, image, text, text_padding_mask, deterministic=False):
        input_tensors = []   
        padding_masks = []
        if image is not None:
            batch_size = image.shape[0]
            # cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
            # input_tensors = [cls_token]
            # padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32)]
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image.shape[1]), dtype=torch.float32))

        if text is not None:
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)

        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.encoder(x, deterministic, padding_mask)
        return x

    def forward_encoder(self, image, text, text_padding_mask, deterministic=False):
        if image is not None:
            batch_size = image.shape[0]
        else:
            batch_size = text.shape[0]
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32)]
        if image is not None:
            image_keep_length = int(
                image.shape[1] * (1.0 - self.config.image_mask_ratio)
            )
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            image_x, image_mask, image_ids_restore = random_masking(
                image_x, image_keep_length
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image_keep_length), dtype=torch.float32))
        else:
            image_mask = image_ids_restore = None
        
        if text is not None:
            text_keep_length = int(
                text.shape[1] * (1.0 - self.config.text_mask_ratio)
            )
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            text_x, text_mask, text_ids_restore, text_padding_mask = random_masking(
                text_x,
                text_keep_length,
                text_padding_mask,
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)
        else:
            text_mask = text_ids_restore = text_padding_mask = None
        
        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, axis=1)

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
        batch_size = cls_x.shape[0]
        input_tensors = [self.decoder_input_projection(cls_x)]
        padding_masks = [torch.zeros((batch_size,  1), dtype=torch.float32)]

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
                torch.cat([image_x, masked_image_x], axis=1), image_ids_restore
            )
            image_x = (
                image_x
                + get_2d_sincos_pos_embed(self.config.dec_emb_dim, image_ids_restore.shape[0])
                + self.get_type_embedding('decoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image_ids_restore.shape[0]), dtype=torch.float32))

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
                torch.cat([text_x, masked_text_x], axis=1), text_ids_restore
            )
            text_x = (
                text_x
                + get_1d_sincos_pos_embed(self.config.dec_emb_dim, text_ids_restore.shape[0])
                + self.get_type_embedding('decoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)

        x = torch.cat(input_tensors, axis=1)
        padding_mask = torch.cat(padding_masks, axis=1)
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

if __name__ == '__main__':
    from ml_collections import ConfigDict
    from data import ImageTextDataset, TextDataset
    dataset = ImageTextDataset(ImageTextDataset.get_default_config(), 0)
    unpaired_text_dataset = TextDataset(TextDataset.get_default_config(), 0)
    model = MaskedMultimodalAutoencoder(
        text_vocab_size=dataset.vocab_size,
        image_output_dim=16*16*3,
        config_updates=ConfigDict(dict(model_type='small', image_mask_ratio=0.75, text_mask_ratio=0.75)),
    )
    paired_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        drop_last=True,
        num_workers=8,
        shuffle = True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    for data in paired_dataloader:
        image, text, text_padding_mask = data
        break
    image_patches = extract_patches(image, 16)
    #unpaired_text, unpaired_text_padding_mask = unpaired_text_dataset[1]
    image_output, text_output, image_mask, text_mask = model(
        image_patches,
        text,
        text_padding_mask,
        deterministic=False,
    )
    