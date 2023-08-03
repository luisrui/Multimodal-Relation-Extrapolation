from cover import Tester, TransE
import numpy as np
import sys 
sys.path.append("../OpenKE/") 
import openke
from openke.data import TrainDataLoader

dataset = "FB15K-237"
data_path = f"./data/{dataset}/"

train_dataloader = TrainDataLoader(
	in_path = data_path, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# define the model
transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True,
)

transe.load_checkpoint("./checkpoints/" + dataset + ".ckpt")

rel_embed = transe.rel_embeddings.weight.data
ent_embed = transe.ent_embeddings.weight.data

print('rel_embed', rel_embed.shape)
print('ent_embed', ent_embed.shape)

np.savez(dataset + "_TransE_embed.npz", rM=rel_embed, eM=ent_embed)
