from cover import Tester, TransE
import numpy as np

# define the model
transe = TransE(
    ent_tot=27909,
    rel_tot=136,
    dim=200,
    p_norm=1,
    norm_flag=True,
)

transe.load_checkpoint("./checkpoints/OpenBG-IMG.ckpt")

rel_embed = transe.rel_embeddings.weight.data
ent_embed = transe.ent_embeddings.weight.data

np.savez("OpenBG_TransE_embed.npz", rM=rel_embed, eM=ent_embed)
