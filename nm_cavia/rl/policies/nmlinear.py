#-*- coding: utf-8 -*-
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import torch.nn as nn

class NMLinear(Module):
	'''
		PyTorch Linear (fully connected) layer with an extra hidden component for
		neuromodulators (modulatory neurons). Based on modified implementation of 
		Linear layer from:
		https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py

		Args:
			in_features: size of each input sample
			out_features: size of each output sample
			out_actfn: the activation function for the output
			bias: If set to `False`, the layer will not have a bias neuron. Default is
				True.
			nm_info:{
					'1':
						{
						 'in_activation': activation_fn,
						 'out_activation': activation_fn,
						 'features': 1,
						 'lr': 0.01
						 'plasticity_coefficients': [0.1, 0.2, 0.3, 0.2]
						 'min_weight': -10.
						 'max_weight': 10.
						}
					...
				}
		Note: bias not affected by neuromodulators
	'''
	NM_GATEPLASTIC = '1'
	NM_GATEACT = '2'
	NM_NOISE = '3'
	NM_INVERTACT = '4'
	def __init__(self, in_features, out_features, out_actfn, bias=True, nm_info=None):
		super(NMLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(Tensor(out_features, in_features))
		self.hebb_weight = None
		if bias: self.bias = Parameter(Tensor(out_features))
		else: self.register_parameter('bias', None)

		self.out_actfn = out_actfn
		self.nm_info = nm_info
		active_nm_types = []
		nm_types = [NMLinear.NM_GATEPLASTIC, NMLinear.NM_GATEACT, NMLinear.NM_NOISE,\
			NMLinear.NM_INVERTACT]
		if nm_info is not None:
			for nm_type, value in nm_info.items():
				if value['features'] <= 0: continue
				else: active_nm_types.append(nm_type)
				self.register_parameter('in_mod'+nm_type, Parameter(Tensor(value['features'], in_features)))
				self.register_parameter('out_mod'+nm_type, Parameter(Tensor(out_features, value['features'])))
		inactive_nm_types = set(nm_types).difference(set(active_nm_types))
		for nm_type in inactive_nm_types:
			self.register_parameter('in_mod'+nm_type, None)
			self.register_parameter('out_mod'+nm_type, None)
		self.reset_parameters()
	
	def reset_parameters(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
		if self.in_mod1 is not None: init.kaiming_uniform_(self.in_mod1, a=math.sqrt(5))
		if self.out_mod1 is not None: init.kaiming_uniform_(self.out_mod1, a=math.sqrt(5))
		if self.in_mod2 is not None: init.kaiming_uniform_(self.in_mod2, a=math.sqrt(5))
		if self.out_mod2 is not None: init.kaiming_uniform_(self.out_mod2, a=math.sqrt(5))
		if self.in_mod3 is not None: init.kaiming_uniform_(self.in_mod3, a=math.sqrt(5))
		if self.out_mod3 is not None: init.kaiming_uniform_(self.out_mod3, a=math.sqrt(5))
		if self.in_mod4 is not None: init.kaiming_uniform_(self.in_mod4, a=math.sqrt(5))
		if self.out_mod4 is not None: init.kaiming_uniform_(self.out_mod4, a=math.sqrt(5))
	
	def forward(self, input_):
		_w = self.hebb_weight if self.hebb_weight is not None else self.weight
		output = F.linear(input_, _w, self.bias)
		
		in_key, out_key = 'in_activation', 'out_activation'
		if self.in_mod2 is not None:
			nm_type = NMLinear.NM_GATEACT
			act_fn = self.nm_info[nm_type][in_key]
			mod_features = act_fn(F.linear(input_, self.in_mod2, None))
			act_fn = self.nm_info[nm_type][out_key]
			output *= act_fn(F.linear(mod_features, self.out_mod2, None))
		if self.in_mod3 is not None:
			nm_type = NMLinear.NM_NOISE 
			act_fn = self.nm_info[nm_type][in_key]
			mod_features = act_fn(F.linear(input_, self.in_mod3, None))
			act_fn = self.nm_info[nm_type][out_key]
			std = act_fn(F.linear(mod_features, self.out_mod3, None))
			if act_fn == F.relu: std[std==0.] += 0.0001 # small epsilon
			#print('output before\t', output)
			#print('mod value\t', std)
            # TODO this needs to be fix as distributions are not differentiable. use VAE trick
            # sample from a standard 0 mean, 1 std guassian and multiply by `std` mod activation.
			output += torch.normal(torch.zeros_like(std), std)
		if self.in_mod4 is not None:
			nm_type = NMLinear.NM_INVERTACT 
			act_fn = self.nm_info[nm_type][in_key]
			mod_features = act_fn(F.linear(input_, self.in_mod4, None))
			act_fn = self.nm_info[nm_type][out_key]
			sign_ = torch.sign(act_fn(F.linear(mod_features, self.out_mod4, None)))
			sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
			output *= sign_
		output = self.out_actfn(output)
		if self.in_mod1 is not None: self.hebb_plasticity(input_, output)
		return output
	
	def hebb_plasticity(self, input_, output):
		nm_type = NMLinear.NM_GATEPLASTIC 
		A, B, C, D = self.nm_info[nm_type]['plasticity_coefficients']
		min_, max_ = self.nm_info[nm_type]['min_weight'], self.nm_info[nm_type]['max_weight']
		act_fn = self.nm_info[nm_type]['in_activation']
		mod_features = act_fn(F.linear(input_, self.in_mod1, None))
		act_fn = self.nm_info[nm_type]['out_activation']
		out_mod_features = act_fn(F.linear(mod_features, self.out_mod1, None))
		for i in range(input_.shape[0]):
			inp, out, out_mfeat = input_[i], output[i], out_mod_features[i]
			weight_delta = (A*torch.ger(out, inp)) + (B*inp) + (C*out.view(-1, 1)) + D
			weight_delta *= self.nm_info[nm_type]['lr']
			weight_delta *= out_mfeat.view(-1, 1)
			inactive_weight = self.weight == 0.
			if self.bias is not None: inactive_weight2 = self.bias == 0.
			if self.hebb_weight is None: self.hebb_weight = self.weight + weight_delta
			else: self.hebb_weight = self.hebb_weight + weight_delta
			self.hebb_weight = torch.clamp(self.hebb_weight, min=min_, max=max_)
			self.hebb_weight[inactive_weight] = 0.
			if self.bias is not None: self.bias[inactive_weight2] = 0.
	
	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}, mod_info={}'.format(self.in_features, \
			self.out_features, self.bias is not None, self.nm_info)

class NMLinearReduced(Module):
    ''' only includes neuromodulator that inverts sign of neural activity (weighted sum of input) '''
    def __init__(self, in_features, out_features, nm_features, bias=True):
        super(NMLinearReduced, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nm_features = nm_features
        self.in_nm_act = F.relu # NOTE hardcoded activation function
        self.out_nm_act = torch.tanh # NOTE hardcoded activation function

        self.std = nn.Linear(in_features, out_features, bias=bias)
        self.in_nm = nn.Linear(in_features, nm_features, bias=bias)
        self.out_nm = nn.Linear(nm_features, out_features, bias=bias)

    def forward(self, data, params=None):
        assert False, 'NMLinearReduced.forward(...). We should never get here'
        output = self.std(data)
        mod_features = self.in_nm_act(self.in_nm(data))
        sign_ = torch.sign(self.out_nm_act(self.out_nm(mod_features)))
        sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
        output *= sign_
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, nm_features={}'.format(self.in_features,\
                self.out_features, self.nm_features)
