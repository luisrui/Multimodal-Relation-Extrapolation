
import torch
import torch.nn as nn
import os
import json
import numpy as np
from torch.nn import functional as F
import random

from module.model import (
	extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy, 
	mask_intersection, all_mask, mask_not
)

class BaseModule(nn.Module):

	def __init__(self):
		super(BaseModule, self).__init__()
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

	def load_checkpoint(self, path):
		self.load_state_dict(torch.load(os.path.join(path)))
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
		
class NegativeSampling(BaseModule):
	def __init__(
			self,
			args,
			model = None, 
			loss_fn = None, 
			regul_rate = 0.0, 
			l3_regul_rate = 0.0,
			neg_rel = 0, 
			neg_ent = 1, 
			sampling_mode = "normal",
			bern_flag = False,
			filter_flag = True,
			score_norm_flag = False
			):
		super(NegativeSampling, self).__init__()
		#model configs
		self.args = args
		self.model = model
		self.loss_fn = loss_fn
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

		#sample configs
		self.rel_total = self.model.num_relations if model is not None else 237
		self.neg_rel = neg_rel
		self.neg_ent = neg_ent
		self.bern_flag = bern_flag
		self.filter_flag = filter_flag
		self.sampling_mode = sampling_mode
		self.score_norm_flag = score_norm_flag
		self.p_norm = 1
		if self.sampling_mode == "normal":
			self.cross_sampling_flag = None
		else:
			self.cross_sampling_flag = 0
		
		self.rel_embedding = nn.Embedding(self.model.num_relations, self.model.dim)
	
	def __count_htr(self, head, rel, tail):
		self.h_of_tr = {}
		self.t_of_hr = {}
		self.r_of_ht = {}
		self.h_of_r = {}
		self.t_of_r = {}
		self.freqRel = {}
		self.lef_mean = {}
		self.rig_mean = {}

		triples = zip(head, tail, rel)
		for h, t, r in triples:
			if (h, r) not in self.t_of_hr:
				self.t_of_hr[(h, r)] = []
			self.t_of_hr[(h, r)].append(t)
			if (t, r) not in self.h_of_tr:
				self.h_of_tr[(t, r)] = []
			self.h_of_tr[(t, r)].append(h)
			if (h, t) not in self.r_of_ht:
				self.r_of_ht[(h, t)] = []
			self.r_of_ht[(h, t)].append(r)
			if r not in self.freqRel:
				self.freqRel[r] = 0
				self.h_of_r[r] = {}
				self.t_of_r[r] = {}
			self.freqRel[r] += 1.0
			self.h_of_r[r][h] = 1
			self.t_of_r[r][t] = 1

		for t, r in self.h_of_tr:
			self.h_of_tr[(t, r)] = np.array(list(set(self.h_of_tr[(t, r)])))
		for h, r in self.t_of_hr:
			self.t_of_hr[(h, r)] = np.array(list(set(self.t_of_hr[(h, r)])))
		for h, t in self.r_of_ht:
			self.r_of_ht[(h, t)] = np.array(list(set(self.r_of_ht[(h, t)])))
		# for r in rel:
		# 	self.h_of_r[r] = np.array(list(self.h_of_r[r].keys()))
		# 	self.t_of_r[r] = np.array(list(self.t_of_r[r].keys()))
		# 	self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
		# 	self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])
	
	def scoring_fn(self, x, edge_index, edge_type):
		batch_head = x[edge_index[0]]
		batch_tail = x[edge_index[1]]
		batch_rel = self.rel_embedding(edge_type)
		if self.score_norm_flag:
			batch_head = F.normalize(batch_head, 2, -1)
			batch_tail = F.normalize(batch_tail, 2, -1)
			batch_rel = F.normalize(batch_rel, 2, -1)
		score = (batch_head + batch_rel) - batch_tail
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score
	
	def get_model_device(self):
		return next(self.parameters()).device
	
	def neg_sample_fn(self, node_list, edge_index, edge_type):
		assert edge_index.shape[0] == 2
		batch_h, batch_t, batch_r = edge_index[0], edge_index[1], edge_type
		len_triples = batch_h.__len__()
		self.__count_htr(batch_h.numpy(), batch_r.numpy(), batch_t.numpy())
		batch_data = {}
		if self.sampling_mode == "normal":
			batch_data['mode'] = "normal"
			batch_h_sample = np.repeat(batch_h.view(-1, 1).numpy(), 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_t_sample = np.repeat(batch_t.view(-1, 1).numpy(), 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_r_sample = np.repeat(batch_r.view(-1, 1).numpy(), 1 + self.neg_ent + self.neg_rel, axis = -1)
			for idx, (h, t, r) in enumerate(zip(batch_h, batch_t, batch_r)):
				last = 1
				if self.neg_ent > 0:
					neg_head, neg_tail = self.__normal_batch(node_list, h, t, r, self.neg_ent)
					if len(neg_head) > 0:
						batch_h_sample[idx][last:last + len(neg_head)] = neg_head
						last += len(neg_head)
					if len(neg_tail) > 0:
						batch_t_sample[idx][last:last + len(neg_tail)] = neg_tail
						last += len(neg_tail)
				if self.neg_rel > 0:
					neg_rel = self.__rel_batch(h, r, t, self.neg_rel)
					batch_r_sample[idx][last:last + len(neg_rel)] = neg_rel
					last += len(neg_rel)
			batch_h = batch_h_sample.transpose()
			batch_t = batch_t_sample.transpose()
			batch_r = batch_r_sample.transpose()

		#batch_y = np.concatenate([np.ones((len_triples, 1)), np.zeros((len_triples, self.neg_ent + self.neg_rel))], -1).transpose()
		# expand_edge_h = torch.tensor(batch_h.squeeze().flatten())
		# expand_edge_t = torch.tensor(batch_t.squeeze().flatten())
		expand_edge_index = torch.tensor(np.array([batch_h.squeeze().flatten(), batch_t.squeeze().flatten()]), dtype=torch.int64)
		expand_edge_type = torch.tensor(batch_r.squeeze().flatten(), dtype=torch.int64)
		return expand_edge_index, expand_edge_type
	
	def _get_positive_score(self, score, num_pos_samples):
		positive_score = score[:num_pos_samples]
		positive_score = positive_score.view(-1, num_pos_samples).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, num_pos_samples):
		negative_score = score[num_pos_samples:]
		negative_score = negative_score.view(-1, num_pos_samples).permute(1, 0)
		return negative_score
	
	#Computing the loss of the whole model
	def forward(self, node_list, x, edge_index, edge_type, batch, deterministic=False):
		device = self.get_model_device()
		x_gcn, batch_output = self.model(
			node_list, x, edge_index, edge_type, batch, deterministic)
		edge_index_expand, edge_type_expand = self.neg_sample_fn(node_list, edge_index, edge_type)
		edge_index_expand = edge_index_expand.to(device)
		edge_type_expand = edge_type_expand.to(device)
		score = self.scoring_fn(x_gcn, edge_index_expand, edge_type_expand)
		pos_score = self._get_positive_score(score, len(edge_type))
		neg_score = self._get_negative_score(score, len(edge_type))
		loss_res_gcn = self.loss_fn(pos_score, neg_score)
		# if self.regul_rate != 0:
		# 	loss_res += self.regul_rate * self.model.regularization(data)
		# if self.l3_regul_rate != 0:
		# 	loss_res += self.l3_regul_rate * self.model.l3_regularization()

		image = batch['image']
		text = batch['text']
		text_padding_mask = batch['text_padding_mask']
		unpaired_text = batch['unpaired_text']
		unpaired_text_padding_mask = batch['unpaired_text_padding_mask']
		image_output = batch_output['image_output']
		image_mask = batch_output['image_mask']
		text_output = batch_output['text_output']
		text_mask= batch_output['text_mask']
		unpaired_text_output = batch_output['unpaired_text_output']
		unpaired_text_mask = batch_output['unpaired_text_mask']
		image_patches = extract_patches(image, self.args.patch_size)
		# Forward Propogation
		#Missing discretized image optimization
		image_loss = patch_mse_loss(
			image_output, image_patches,
			None if self.args.image_all_token_loss else image_mask
		)
		text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
			text_output, text,  
			mask_intersection(
				all_mask(text) if self.args.text_all_token_loss else text_mask,
				mask_not(text_padding_mask)
			)
		)
		if unpaired_text_output is not None:
			unpaired_text_loss, unpaired_text_accuracy = cross_entropy_loss_and_accuracy(
				unpaired_text_output, unpaired_text,
				mask_intersection(
					all_mask(unpaired_text) if self.args.text_all_token_loss else unpaired_text_mask,
					mask_not(unpaired_text_padding_mask)
				)
			)
		else:
			unpaired_text_loss = 0

		loss_image_text = (
			self.args.image_loss_weight * image_loss
			+ self.args.text_loss_weight * text_loss
			+ self.args.unpaired_text_loss_weight * unpaired_text_loss
		)
		
		loss = loss_image_text + self.args.gcn_part_loss * loss_res_gcn
		info = dict(
			loss_res_gcn=loss_res_gcn,
			loss_image_text=loss_image_text,
			image_loss=image_loss,
			text_loss=text_loss,
			unpaired_text_loss=unpaired_text_loss
		)
		return loss, info
	
	def __normal_batch(self, node_list, h, t, r, neg_size):
		neg_size_h = 0
		neg_size_t = 0
		prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
		for i in range(neg_size):
			if random.random() < prob:
				neg_size_h += 1
			else:
				neg_size_t += 1

		neg_list_h = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_h:
			neg_tmp_h = self.__corrupt_head(node_list, t, r, num_max = (neg_size_h - neg_cur_size) * 2)
			neg_list_h.append(neg_tmp_h)
			neg_cur_size += len(neg_tmp_h)
		if neg_list_h != []:
			neg_list_h = np.concatenate(neg_list_h)
			
		neg_list_t = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_t:
			neg_tmp_t = self.__corrupt_tail(h, r, node_list, num_max = (neg_size_t - neg_cur_size) * 2)
			neg_list_t.append(neg_tmp_t)
			neg_cur_size += len(neg_tmp_t)
		if neg_list_t != []:
			neg_list_t = np.concatenate(neg_list_t)

		return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]
	
	def __rel_batch(self, h, t, r, neg_size):
		neg_list = []
		neg_cur_size = 0
		while neg_cur_size < neg_size:
			neg_tmp = self.__corrupt_rel(h, r, t, num_max = (neg_size - neg_cur_size) * 2)
			neg_list.append(neg_tmp)
			neg_cur_size += len(neg_tmp)
		return np.concatenate(neg_list)[:neg_size]
	
	def __corrupt_head(self, node_list, t, r, num_max = 1):
		tmp = torch.tensor(random.sample(node_list.numpy().tolist(), k=num_max))
		t_index, r_index= t.item(), r.item()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.h_of_tr[(t_index, r_index)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_tail(self, h, r, node_list, num_max = 1):
		tmp = torch.tensor(random.sample(node_list.numpy().tolist(), k=num_max))
		h_index, r_index= h.item(), r.item()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.t_of_hr[(h_index, r_index)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_rel(self, h, r, t, num_max = 1):
		tmp = random.sample(r.numpy().tolist(), k=num_max)
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg
	
if __name__ == '__main__':
	from torch_geometric.utils import subgraph
	from data import MMKGDataset
	import torch
	node_list = torch.arange(128)
	device = 'cpu'
	graph_dataset = MMKGDataset(config=MMKGDataset.get_default_config(), 
			     name='FB15K-237', 
				 root='../origin_data/FB15K-237', 
				 device=device)
	edge_index, edge_type = subgraph(node_list, graph_dataset.struc_dataset.edge_index, graph_dataset.struc_dataset.edge_type)
	test_sample = NegativeSampling()
	batch = test_sample.neg_sample_fn(node_list, edge_index, edge_type)
