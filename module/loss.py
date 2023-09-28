import torch
from torch import nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0):
        super(MarginLoss,self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
            
    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

class SigmoidLoss(nn.Module):

	def __init__(self, adv_temperature = None):
		super(SigmoidLoss, self).__init__()
		self.criterion = nn.LogSigmoid()
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
		else:
			return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()