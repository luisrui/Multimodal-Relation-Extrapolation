import torch 
import torch.nn as nn
import os
import json
from torch.nn import functional as F

class BaseModule(nn.Module):

	def __init__(self):
		super(BaseModule, self).__init__()
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

	def load_checkpoint(self, path, device):
		self.load_state_dict(torch.load(os.path.join(path), map_location=device))
		self.eval()

	def save_checkpoint(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		f = open(path, "r")
		parameters = json.loads(f.read())
		f.close()
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

	def save_parameters(self, path):
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def get_parameters(self, mode = "numpy", param_dict = None):
		all_param_dict = self.state_dict()
		if param_dict == None:
			param_dict = all_param_dict.keys()
		res = {}
		for param in param_dict:
			if mode == "numpy":
				res[param] = all_param_dict[param].cpu().numpy()
			elif mode == "list":
				res[param] = all_param_dict[param].cpu().numpy().tolist()
			else:
				res[param] = all_param_dict[param]
		return res

	def set_parameters(self, parameters):
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
             
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
        device = input.device
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32).to(device)
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

        device = inputs.device
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask = padding_mask.expand(attention.shape)
            attention = torch.where(padding_mask > 0, torch.tensor(-1e7).to(device), attention)

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
    
class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        nn.init.xavier_normal_(self.proj1.weight)
        nn.init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)
