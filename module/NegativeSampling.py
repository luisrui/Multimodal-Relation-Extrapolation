import torch
import torch.nn as nn
import os
import json
import numpy as np
from torch.nn import functional as F
import random
from collections import defaultdict
#from muse import MaskGitVQGAN

from sklearn.metrics.pairwise import cosine_similarity
from module.model import (
	extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy, 
	mask_intersection, all_mask, mask_not
)
from module.vqgan import get_image_tokenizer
from module.submodule import BaseModule
		
class NegativeSampling(BaseModule):
	def __init__(
			self,
			args,
			whole_triples,
			model = None,
			loss_fn = None, 
			regul_rate = 0.5, 
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

		#sample configs
		self.rel_total = self.model.num_relations if model is not None else 237
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
		
		#self.rel_embedding = nn.Embedding(self.model.num_relations, self.model.dim)
		#self.ent_embedding = nn.Embedding(self.model.num_nodes, self.model.dim)
		self.feature_extractor = nn.Linear(self.model.dim * 2, self.model.dim)
		if whole_triples is not None:
			h, r, t = whole_triples
			self.__count_htr(h, r, t)

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
		
	
	def scoring_fn(self, local_global_id, x, relations, edge_index, edge_type):
		#batch_head = self.ent_embedding(self._global_mapping(local_global_id, edge_index[0]).to(self.get_model_device()))
		batch_head = x[edge_index[0].long()]
		#batch_tail = self.ent_embedding(self._global_mapping(local_global_id, edge_index[1]).to(self.get_model_device()))
		batch_tail = x[edge_index[1].long()]
		# batch_rel = self.rel_embedding(edge_type)
		score = self._calc(h=batch_head, t=batch_tail, r=relations)
		return score
	
	def get_model_device(self):
		return next(self.parameters()).device
	
	def neg_sample_fn(self, local_global_id, node_list, edge_index, edge_type):
		assert edge_index.shape[0] == 2
		batch_h, batch_t, batch_r = edge_index[0], edge_index[1], edge_type
		#len_triples = batch_h.__len__()
		batch_data = {}
		if self.sampling_mode == "normal":
			batch_data['mode'] = "normal"
			batch_h_sample = np.repeat(batch_h.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
			batch_t_sample = np.repeat(batch_t.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
			batch_r_sample = np.repeat(batch_r.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
			for idx, (h, t, r) in enumerate(zip(batch_h, batch_t, batch_r)):
				last = 1
				if self.neg_ent > 0:
					neg_head, neg_tail = self.__normal_batch(local_global_id, node_list, h, t, r, self.neg_ent)
					if len(neg_head) > 0:
						batch_h_sample[idx][last:last + len(neg_head)] = neg_head
						last += len(neg_head)
					if len(neg_tail) > 0:
						batch_t_sample[idx][last:last + len(neg_tail)] = neg_tail
						last += len(neg_tail)
			batch_h = batch_h_sample.transpose()
			batch_t = batch_t_sample.transpose()
			batch_r = batch_r_sample.transpose()

		expand_edge_index = torch.tensor(np.array([batch_h.squeeze().flatten(), batch_t.squeeze().flatten()]), dtype=torch.int32)
		expand_edge_type = torch.tensor(batch_r.squeeze().flatten(), dtype=torch.int32)
		return expand_edge_index, expand_edge_type
	
	def _calc(self, h, t, r, mode='normal', score_model='transe'):
		if score_model == 'transe':
			if self.score_norm_flag:
				h = F.normalize(h, 2, -1)
				r = F.normalize(r, 2, -1)
				t = F.normalize(t, 2, -1)
			if mode != 'normal':
				h = h.view(-1, r.shape[0], h.shape[-1])
				t = t.view(-1, r.shape[0], t.shape[-1])
				r = r.view(-1, r.shape[0], r.shape[-1])
			if mode == 'head_batch':
				score = h + (r - t)
			else:
				score = (h + r) - t
			score = torch.norm(score, self.p_norm, -1).flatten()
			return score
		elif score_model == 'distmult':
			if mode != 'normal':
				h = h.view(-1, r.shape[0], h.shape[-1])
				t = t.view(-1, r.shape[0], t.shape[-1])
				r = r.view(-1, r.shape[0], r.shape[-1])
			if mode == 'head_batch':
				score = h * (r * t)
			else:
				score = (h * r) * t
			score = torch.sum(score, -1).flatten()
			return score
		
	def _global_mapping(self, local_global_id:dict, tranfer_list:torch.tensor):
		return torch.tensor([local_global_id[i.item()] for i in tranfer_list])
	
	def _get_positive_score(self, score, num_pos_samples):
		positive_score = score[:num_pos_samples]
		positive_score = positive_score.view(-1, num_pos_samples).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, num_pos_samples):
		negative_score = score[num_pos_samples:]
		negative_score = negative_score.view(-1, num_pos_samples).permute(1, 0)
		return negative_score
	
	def centroid_loss(self, x, edge_index, edge_type, rel_embs):
		# rel_list = torch.unique(edge_type)
		# centroids = dict()
		# for h, r, t in zip(edge_index[0], edge_type, edge_index[1]):
		# 	if r not in centroids.keys(): centroids[r] = list()
		# 	pair_emb = torch.cat((x[h], x[t]), dim = 0)
		# 	pair_emb = self.feature_extractor(pair_emb)
		# 	centroids[r].append(pair_emb.cpu().numpy())
		# for rel, points in centroids:
		# 	points = torch.tensor(np.array(points))
		# 	centroids[rel] = torch.sum(points, dim = 0)
		pair_embs = []
		for h, r, t in zip(edge_index[0], edge_type, edge_index[1]):
			pair_emb = torch.cat((x[h], x[t]), dim = 0)
			pair_embs.append(pair_emb)
		pair_embs = torch.stack(pair_embs, dim = 0).to(self.get_model_device())
		pair_embs = self.feature_extractor(pair_embs)
		cen_score = F.cosine_similarity(pair_embs, rel_embs, dim=1)
		return cen_score
	
	#Computing the loss of the whole model
	def forward(self, local_global_id, edge_index, edge_type, batch, deterministic=False):
		device = self.get_model_device()
		#################test
		#n_id = torch.tensor(np.array(list(local_global_id.values())), dtype=torch.int32).to(device)
		x_gcn, rel_emb, batch_output = self.model(
			edge_index, edge_type, batch, deterministic)
		mapped_node_list = torch.arange(torch.max(edge_index))
		edge_index_expand, edge_type_expand = self.neg_sample_fn(local_global_id, mapped_node_list, edge_index, edge_type)
		edge_index_expand = edge_index_expand.to(device)
		edge_type_expand = edge_type_expand.to(device)
		rel_emb_expand = rel_emb.repeat(1 + self.neg_ent, 1).to(device)

		# cen_score = self.centroid_loss(x_gcn, edge_index_expand, edge_type_expand, rel_emb_expand)
		# c_pos_score = self._get_positive_score(cen_score, len(edge_type))
		# c_neg_score = self._get_negative_score(cen_score, len(edge_type))
		# loss_rel_center = self.loss_fn(c_pos_score, c_neg_score)

		score = self.scoring_fn(local_global_id, x_gcn, rel_emb_expand, edge_index_expand, edge_type_expand)
		pos_score = self._get_positive_score(score, len(edge_type))
		neg_score = self._get_negative_score(score, len(edge_type))
		loss_res_gcn = self.loss_fn(pos_score, neg_score)

		#struct_loss = loss_rel_center + loss_res_gcn
		struct_loss = loss_res_gcn
		if self.regul_rate != 0:
			struct_loss += self.regul_rate * self.regularization(x_gcn, rel_emb_expand, edge_index_expand, edge_type_expand)

		image = batch['image']
		text = batch['text']
		text_padding_mask = batch['text_padding_mask']
		image_output = batch_output['image_output']
		image_mask = batch_output['image_mask']
		text_output = batch_output['text_output']
		text_mask= batch_output['text_mask']
		contrastive_loss = batch_output['contrastive_loss']

		if image is not None:
			if self.args.discretized_image:
				print(image.shape)
				encoded_image = self.vq_model.encode(image)
				#encoded_image = encode_image(tokenizer_params, image)
				print(encoded_image.shape)
				image_loss, image_accuracy = cross_entropy_loss_and_accuracy(
                    image_output, encoded_image,
                    None if self.args.image_all_token_loss else image_mask
                )
			else:
				image_patches = extract_patches(image, self.args.patch_size)
				#Missing discretized image optimization
				image_loss = patch_mse_loss(
					image_output, image_patches,
					None if self.args.image_all_token_loss else image_mask
				)
		else:
			image_loss = 0.0

		if text is not None:
			text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
				text_output, text,  
				mask_intersection(
					all_mask(text) if self.args.text_all_token_loss else text_mask,
					mask_not(text_padding_mask).to(device)
				)
			)
		else:
			text_loss = 0.0

		loss_image_text = (
			self.args.image_loss_weight * image_loss
			+ self.args.text_loss_weight * text_loss
		)

		loss = loss_image_text + self.args.gcn_loss_weight * loss_res_gcn + self.args.contrastive_loss_weight * contrastive_loss
		# info = dict(
		# 	struc_loss=loss_res_gcn,
		# 	loss_image_text=loss_image_text,
		# 	image_loss=image_loss,
		# 	text_loss=text_loss,
		# 	contrastive_loss=contrastive_loss
		# )
		info = dict(
			struct_loss = struct_loss,
			gcn_loss=loss_res_gcn,
			loss_image_text=loss_image_text,
			image_loss=image_loss,
			text_loss=text_loss,
			contrastive_loss=contrastive_loss
		)
		return loss, info
	
	def evaluate(self, h, r, t, score_model='transe'):
		if score_model == 'transe':
			if self.score_norm_flag:
				h = F.normalize(h, 2, -1)
				r = F.normalize(r, 2, -1)
				t = F.normalize(t, 2, -1)
			score = (h + r) - t
			score = torch.norm(score, self.p_norm, -1).flatten()
			return score
		else:
			print('invalid scoring model!')
			return None

	def regularization(self, x, relations, edge_index, edge_type):
		batch_head = x[edge_index[0].long()]
		batch_tail = x[edge_index[1].long()]
		#batch_rel = self.rel_embedding(edge_type)
		regul = (torch.mean(batch_head ** 2) + 
                 torch.mean(batch_tail ** 2) + 
                 torch.mean(relations ** 2)) / 3
		return regul

	def generate_eval_list(self, local_global_id, edge_index, edge_type):
		mapped_node_list = torch.arange(torch.max(edge_index))
		edge_index_expand, edge_type_expand = self.neg_sample_fn(local_global_id, mapped_node_list, edge_index, edge_type)
		return edge_index_expand, edge_type_expand
	
	def __normal_batch(self, local_global_id, node_list, h, t, r, neg_size):
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
			neg_tmp_h = self.__corrupt_head(local_global_id, node_list, t, r, num_max = (neg_size_h - neg_cur_size) * 2)
			neg_list_h.append(neg_tmp_h)
			neg_cur_size += len(neg_tmp_h)
		if neg_list_h != []:
			neg_list_h = np.concatenate(neg_list_h)
			
		neg_list_t = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_t:
			neg_tmp_t = self.__corrupt_tail(local_global_id, h, r, node_list, num_max = (neg_size_t - neg_cur_size) * 2)
			neg_list_t.append(neg_tmp_t)
			neg_cur_size += len(neg_tmp_t)
		if neg_list_t != []:
			neg_list_t = np.concatenate(neg_list_t)

		return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]
	  
	def __corrupt_head(self, local_global_id, node_list, t, r, num_max = 1):
		try:
			tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=num_max))
		except:
			tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=len(node_list)))
		t_index, r_index= t.item(), r.item()
		if not self.filter_flag:
			return tmp
		compare_list = torch.tensor([local_global_id[num.item()] for num in tmp])
		mask = np.in1d(compare_list, self.h_of_tr[(local_global_id[t_index], r_index)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_tail(self, local_global_id, h, r, node_list, num_max = 1):
		try:
			tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=num_max))
		except:
			tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=len(node_list)))
		h_index, r_index= h.item(), r.item()
		if not self.filter_flag:
			return tmp
		compare_list = torch.tensor([local_global_id[num.item()] for num in tmp])
		mask = np.in1d(compare_list, self.t_of_hr[(local_global_id[h_index], r_index)], assume_unique=True, invert=True)
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
