# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def load(cl, args, path):
	model = cl(**args)
	model.load_state_dict(torch.load(path))
	return model

def save(model, path):
	path = path + '.pth'
	torch.save(model.state_dict(), path)


class NMLinear2(nn.Module):
	# neuromodulator here inverts the sign of of standard neuron's weighted sum of standard neuron.
	# neuromodulator layer is an rnn
	def __init__(self, in_features, out_features, nmout_features):
		super(NMLinear2, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.nmout_features = nmout_features
		self.fc = nn.Linear(in_features, out_features)
		self.nm = nn.RNNCell(in_features, nmout_features, True, 'tanh')
		self.hx = torch.zeros(1, nmout_features) # hx: hidden state/memory of rnn, dim 0: batch size
		self.nmfc = nn.Linear(nmout_features, out_features) # neuromod rnn to linear output

	def forward(self, x):
		std = self.fc(x)
		self.hx = self.nm(x, self.hx)
		hmod = torch.tanh(self.nmfc(self.hx))
		sgn = torch.sign(hmod)
		sgn[sgn==0.] = 1.
		#print('x=', x)
		#print('sgn=', sgn)
		#print('std=', std)
		#print('std*sgn=', std*sgn)
		#print()
		return std*sgn, self.hx
	
	def reset(self, batch_size=None):
		batch_size = 1 if batch_size is None else batch_size
		self.hx = torch.zeros(batch_size, self.nmout_features)
	
	def extra_repr(self):
		raise NotImplementedError

class Policy(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Policy, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.nmfc1 = NMLinear(input_dim, 8, 2)
		self.nmfc2 = NMLinear(8, 8, 2)
		self.nmfc3 = NMLinear(8, output_dim, 2)
	
	def forward(self, x):
		# NOTE, not yet sure of the best non-linearity
		x, hx1 = self.nmfc1(x)
		x, hx2 = self.nmfc2(F.relu(x)) 
		logits, hx3 = self.nmfc3(F.relu(x))
		dist = torch.distributions.Categorical(logits=logits)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action, log_prob
	
	def reset(self, batch_size=None):
		batch_size = 1 if batch_size is None else batch_size
		self.nmfc1.reset(batch_size)
		self.nmfc2.reset(batch_size)
		self.nmfc3.reset(batch_size)

class EvoPolicy(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(EvoPolicy, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		#self.nmfc1 = NMLinear(input_dim, 12, 4)
		#self.nmfc2 = NMLinear(12, output_dim, 4)
		self.nmfc1 = NMLinear(input_dim, 24, 12)
		self.nmfc2 = NMLinear(24, output_dim, 12)
		# disable gradient computation
		for param in self.parameters():
			param.requires_grad = False
	
	def forward(self, x):
		# NOTE, not yet sure of the best non-linearity
		x, hx1 = self.nmfc1(x)
		logits, hx2 = self.nmfc2(F.relu(x))
		#logits, hx2 = self.nmfc2(F.tanh(x))
		return logits

	def reset(self, batch_size=None):
		batch_size = 1 if batch_size is None else batch_size
		self.nmfc1.reset(batch_size)
		self.nmfc2.reset(batch_size)
	
	### methods specific for cma-es
	def get_parameters(self):
		# get parameters of model that are flattened and concatenated together in a single dim
		params = []
		for p in self.parameters():
			params.append(copy.deepcopy(p.data))
		params = [p.view(-1,) for p in params]
		params = torch.cat(params, dim=0)
		return params
	
	def set_parameters(self, params):
		params = copy.deepcopy(params)
		if not isinstance(params, torch.Tensor):
			params = torch.tensor(params, dtype=torch.float32)
		idx = 0
		for p in self.parameters():
			sz = p.shape.numel() # prod of shape
			p.data = (params[idx : idx + sz]).view(p.shape)
			idx += sz
	
	### methods below added for compatibility with the current training code in the repo.
	def perform_action(self, x):
		x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
		if x.dim() == 1:
			x = x.view(1, -1) # include a batch dim
		logits = self.forward(x)
		action = logits.argmax(dim=1)
		if action.shape[0] == 1: action = int(action)
		return action
		#output = torch.tanh(self.forward(x))
		#if output > 0.33: return 2
		#elif output < -0.33: return 1
		#else: return 0

	def draw_network(self, path=None, prune=False):
		return
	
	def enable_neurons_output_logging(self):
		return
	
	def get_reward(self):
		return self.reward
	
	def set_reward(self, reward):
		self.reward = reward
	
	def get_params_copy(self):
		return {'input_dim':self.input_dim, 'output_dim':self.output_dim}

# unit testing
if __name__ == '__main__':
	print('Policy testing')
	policy = Policy(6, 3)
	policy.reset(4)
	x = torch.rand(4, 6)
	action, log_prob = policy(x)
	print('action:', action)
	print('log_prob:', log_prob)

	print('\nEvolution Policy')
	policy = EvoPolicy(6, 3)
	#print('current parameters')
	#for p in policy.parameters(): print(p)
	#params = policy.get_parameters()
	#newparams = torch.rand(*params.shape)
	#policy.set_parameters(newparams)
	#print('\nnew parameters')
	#for p in policy.parameters(): print(p)
	policy.reset(4)
	x = torch.rand(4, 6)
	action, log_prob = policy(x)
	print('action:', action)
	print('log_prob:', log_prob)
