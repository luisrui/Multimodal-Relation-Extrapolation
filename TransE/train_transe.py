import sys 
sys.path.append("../OpenKE/") 
import openke
from openke.config import Trainer
from cover import Tester, TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import os

dataset = "FB15K-237"
data_path = f"./data/{dataset}/"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = data_path, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
#test_dataloader = TestDataLoader(data_path, "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

#load the checkpoints
checkpoint_path = f'./checkpoints/{dataset}.ckpt'
if os.path.exists(checkpoint_path):
	transe.load_checkpoint(checkpoint_path)

if not os.path.exists('./checkpoints'):
	os.mkdir('./checkpoints/')

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

#train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transe.save_checkpoint(checkpoint_path)

# # test the model
# transe.load_checkpoint('./checkpoints/OpenBG-IMG.ckpt')
# tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)