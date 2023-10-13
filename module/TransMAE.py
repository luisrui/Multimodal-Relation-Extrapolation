from module.NegativeSampling import NegativeSampling
from module.submodule import BaseModule, LayerNormalization, Transformer
from module.model import get_1d_sincos_pos_embed
import torch
from torch import nn as nn

class DistillModel(BaseModule):
    def __init__(self, emb_dim, transformer_emb_dim, text_type_embedding, learned_text_embedding):
        super(DistillModel, self).__init__()
        self.dim = emb_dim
        self.transformer_emb_dim = transformer_emb_dim
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(transformer_emb_dim, 2 * emb_dim)
        self.fc2 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc3 = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.LeakyReLU()
        self.layer_norm = LayerNormalization(emb_dim)
        self.criterion = nn.MSELoss()
        self.text_type_embedding = text_type_embedding
        self.text_embedding = learned_text_embedding

    def get_model_device(self):
        return next(self.parameters()).device
    
    def forward(self, rel_tokens, rel_embs):
        device = self.get_model_device()
        with torch.no_grad():
            x = (
                self.text_embedding(rel_tokens) + 
                get_1d_sincos_pos_embed(self.transformer_emb_dim, rel_tokens.shape[1]).to(device) +
                self.text_type_embedding
                )
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=-2)
        x = torch.squeeze(x, dim=-2)
        x = self.fc3(x)
        loss = self.criterion(x, rel_embs)
        return loss
    
    def predict(self, rel_tokens):
        device = self.get_model_device()
        with torch.no_grad():
            x = (
                self.text_embedding(rel_tokens) + 
                get_1d_sincos_pos_embed(self.transformer_emb_dim, rel_tokens.shape[1]).to(device) +
                self.text_type_embedding
                )
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=-2)
        x = torch.squeeze(x, dim=-2)
        x = self.fc3(x)
        return x
    
    def update(self, text_type_embedding, learned_text_embedding):
        self.text_type_embedding = text_type_embedding
        self.text_embedding = learned_text_embedding

        

# class TransMAE(BaseModule):
#     def __init__(self, KGCmodel:NegativeSampling, Distillmodel:DistillModel, KGC_epoch:int, Dst_epoch:int):
#         super(TransMAE, self).__init__()
#         self.KGCmodel = KGCmodel
#         self.DistillModel = Distillmodel
#         self.K_epoch = KGC_epoch
#         self.D_epoch = Dst_epoch
    
#     def train(self, total_epochs):

