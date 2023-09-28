import torch
import torch.nn as nn
import os
import json
import numpy as np
from torch.nn import functional as F
import random
from collections import defaultdict
#from muse import MaskGitVQGAN

from module.model import (
	extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy, 
	mask_intersection, all_mask, mask_not, Extractor
)
from module.submodule import BaseModule
from module.loss import MarginLoss
		
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
		self.ent_total = self.model.num_nodes if model is not None else 14208
		self.neg_ent = neg_ent
		self.bern_flag = bern_flag
		self.filter_flag = filter_flag
		self.sampling_mode = sampling_mode
		self.score_norm_flag = score_norm_flag
		self.center_margin_loss = MarginLoss(margin=1.0)
		self.p_norm = 1
		if self.sampling_mode == "normal":
			self.cross_sampling_flag = None
		else:
			self.cross_sampling_flag = 0

		self.extractor = Extractor(embed_dim=self.args.emb_dim)
		self.ent_embedding = nn.Embedding(self.model.num_nodes, self.args.emb_dim)
		self.embedding_range = nn.Parameter(
			torch.Tensor([(self.args.emb_dim / 10 + 2) / self.args.emb_dim]), requires_grad=False
		)
		nn.init.uniform_(
			tensor = self.ent_embedding.weight.data, 
			a = -self.embedding_range.item(), 
			b = self.embedding_range.item()
		)
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
	
	def scoring_fn(self, h, r, t, entity_embs, relations_embs):
		#batch_head = self.ent_embedding(self._global_mapping(local_global_id, edge_index[0]).to(self.get_model_device()))
		batch_tail = self.ent_embedding(t)
		# batch_rel = self.rel_embedding(edge_type)
		score = self._calc(h=entity_embs, t=batch_tail, r=relations_embs)
		return score
	
	def get_model_device(self):
		return next(self.parameters()).device
	
	def neg_sample_fn(self, triples):
		batch_h, batch_t, batch_r = triples[:, 0], triples[:, 2], triples[:, 1]
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
					neg_head, neg_tail = self.__normal_batch(h, t, r, self.neg_ent)
					if len(neg_head) > 0:
						batch_h_sample[idx][last:last + len(neg_head)] = neg_head
						last += len(neg_head)
					if len(neg_tail) > 0:
						batch_t_sample[idx][last:last + len(neg_tail)] = neg_tail
						last += len(neg_tail)
			batch_h = batch_h_sample.transpose()
			batch_t = batch_t_sample.transpose()
			batch_r = batch_r_sample.transpose()

		expand_h = torch.tensor(batch_h.squeeze().flatten(), dtype=torch.int32)
		expand_r = torch.tensor(batch_r.squeeze().flatten(), dtype=torch.int32)
		expand_t = torch.tensor(batch_t.squeeze().flatten(), dtype=torch.int32)
		return expand_h, expand_r, expand_t
	
	def _calc(self, h, t, r, mode='normal', score_model='distmult'):
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
		
	def _get_positive_score(self, score, num_pos_samples):
		positive_score = score[:num_pos_samples]
		positive_score = positive_score.view(-1, num_pos_samples).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, num_pos_samples):
		negative_score = score[num_pos_samples:]
		negative_score = negative_score.view(-1, num_pos_samples).permute(1, 0)
		return negative_score
	
	def forward_center(self, h, r, t, ent_embs, rel_embs):
		tail_embs = self.ent_embedding(t)
		pair_embs = self.extractor(ent_embs, tail_embs)
		cen_score = -1 * F.cosine_similarity(pair_embs, rel_embs, dim=1)
		return cen_score
	
	#Computing the loss of the whole model
	def forward(self, batch, deterministic=False):
		device = self.get_model_device()
		#################test
		#n_id = torch.tensor(np.array(list(local_global_id.values())), dtype=torch.int32).to(device)
		x_gcn, rel_emb, batch_output = self.model(batch, deterministic)

		expand_h, expand_r, expand_t = self.neg_sample_fn(batch['triples'])
		expand_h = expand_h.to(device)
		expand_r = expand_r.to(device)
		expand_t = expand_t.to(device)
		x_gcn_expand = x_gcn.repeat(1 + self.neg_ent, 1).to(device)
		rel_emb_expand = rel_emb.repeat(1 + self.neg_ent, 1).to(device)
		loss_rel_center = 0
		
		cen_score = self.forward_center(expand_h, expand_r, expand_t, x_gcn_expand, rel_emb_expand)
		c_pos_score = self._get_positive_score(cen_score, batch['triples'].shape[0])
		c_neg_score = self._get_negative_score(cen_score, batch['triples'].shape[0])
		loss_rel_center = self.center_margin_loss(c_pos_score, c_neg_score)

		score = self.scoring_fn(expand_h, expand_r, expand_t, x_gcn_expand, rel_emb_expand)
		pos_score = self._get_positive_score(score, batch['triples'].shape[0])
		neg_score = self._get_negative_score(score, batch['triples'].shape[0])
		loss_res_gcn = self.loss_fn(pos_score, neg_score)

		struct_loss = loss_rel_center + loss_res_gcn
		if self.regul_rate != 0:
			struct_loss += self.regul_rate * self.regularization(expand_h, expand_r, expand_t, x_gcn_expand, rel_emb_expand)

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
		info = dict(
			struct_loss = struct_loss,
			gcn_loss=loss_res_gcn,
			center_loss=loss_rel_center,
			loss_image_text=loss_image_text,
			image_loss=image_loss,
			text_loss=text_loss,
			contrastive_loss=contrastive_loss
		)
		return loss, info
	
	def regularization(self, h, r, t, entity_embs, relations_embs):
		batch_tail = self.ent_embedding(t)
		#batch_rel = self.rel_embedding(edge_type)
		regul = (torch.mean(entity_embs ** 2) + 
                 torch.mean(batch_tail ** 2) + 
                 torch.mean(relations_embs ** 2)) / 3
		return regul
	
	def __normal_batch(self, h, t, r, neg_size):
		neg_size_h = 0
		neg_size_t = 0
		#prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
		for i in range(neg_size):
			neg_size_t += 1
			# if random.random() < prob:
			# 	neg_size_h += 1
			# else:
			# 	neg_size_t += 1

		neg_list_h = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_h:
			neg_tmp_h = self.__corrupt_head(t, r, num_max = (neg_size_h - neg_cur_size) * 2)
			neg_list_h.append(neg_tmp_h)
			neg_cur_size += len(neg_tmp_h)
		if neg_list_h != []:
			neg_list_h = np.concatenate(neg_list_h)
			
		neg_list_t = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_t:
			neg_tmp_t = self.__corrupt_tail(h, r, num_max = (neg_size_t - neg_cur_size) * 2)
			neg_list_t.append(neg_tmp_t)
			neg_cur_size += len(neg_tmp_t)
		if neg_list_t != []:
			neg_list_t = np.concatenate(neg_list_t)

		return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]
	  
	def __corrupt_head(self, t, r, num_max = 1):
		tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
		#t_index, r_index= t.item(), r.item()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.h_of_tr[(t, r)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_tail(self, h, r, num_max = 1):
		tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
		h_index, r_index= h.item(), r.item()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.t_of_hr[(h_index, r_index)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

