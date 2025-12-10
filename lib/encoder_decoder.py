###

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model_functs as utils

# -----

# GRU ref:https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-4/

class GRU_unit(nn.Module):

	def __init__(self, input_dim, latent_dim,
		update_gate = None,
		reset_gate = None,
		new_state_netw = None,
		n_unit = None):

		super(GRU_unit, self).__init__()   # initialization of the parent class

		if update_gate is None:
			self.update_gate = nn.Sequential(
				nn.Linear(latent_dim * 2 + input_dim, n_unit),   # latent mean, latent std
				nn.Tanh(),
				nn.Linear(n_unit, latent_dim),
				nn.Sigmoid())
			utils.init_netw_weights(self.update_gate)   # initialize weights in linear layers
		else: 
			self.update_gate = update_gate

		if reset_gate is None:
			self.reset_gate = nn.Sequential(
				nn.Linear(latent_dim * 2 + input_dim, n_unit),
				nn.Tanh(),
				nn.Linear(n_unit, latent_dim),
				nn.Sigmoid())
			utils.init_netw_weights(self.reset_gate)
		else: 
			self.reset_gate = reset_gate

		if new_state_netw is None:
			self.new_state_netw = nn.Sequential(
				nn.Linear(latent_dim * 2 + input_dim, n_unit),
				nn.Tanh(),
				nn.Linear(n_unit, latent_dim * 2))
			utils.init_netw_weights(self.new_state_netw)
		else: 
			self.new_state_netw = new_state_netw


	def forward(self, y_mean, y_std, x, masked_update = True):
		# y_mean, y_std - latent states; shape: n_traj x n_subj x latent_dim
		# x - input; shape: n_traj x n_subj x input_dim

		y_concat = torch.cat([y_mean, y_std, x], -1)   # n_traj x n_subj x (latent_dim x 2 + input_dim)

		update_gate = self.update_gate(y_concat)   # n_traj x n_subj x latent_dim
		reset_gate = self.reset_gate(y_concat)   # n_traj x n_subj x latent_dim

		y_concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)   # n_traj x n_subj x (latent_dim x 2 + input_dim)
		
		new_mean, new_std = utils.split_last_dim(self.new_state_netw(y_concat))
		# n_traj x n_subj x (latent_dim x 2) --> two n_traj x n_subj x latent_dim

		new_std = F.softplus(new_std) + 1e-6 #added by bo

		new_mean = (1-update_gate) * new_mean + update_gate * y_mean
		new_std = (1-update_gate) * new_std + update_gate * y_std

		if masked_update:   # x concatenates data and mask
			n_dim = x.size(-1)//2
			mask = x[:,:,n_dim:]   # n_traj x n_subj x n_dim
			utils.check_mask(x[:,:,:n_dim], mask)

			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
			# n_traj x n_subj x 1; if all input dims are masked out, no update to the latent states

			new_mean = mask * new_mean + (1-mask) * y_mean
			new_std = mask * new_std + (1-mask) * y_std

		new_std = new_std.abs()

		return new_mean, new_std   # shape: n_traj x n_subj x latent_dim


class Encoder_z0_ODE_RNN(nn.Module):

	def __init__(self, input_dim, latent_dim, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, n_gru_unit = None):

		super(Encoder_z0_ODE_RNN, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		if GRU_update is None:
			self.GRU_update = GRU_unit(input_dim, latent_dim, n_unit = n_gru_unit)   # initialize GRU unit
		else:
			self.GRU_update = GRU_update

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.z0_diffeq_solver = z0_diffeq_solver

		self.transform_z0 = nn.Sequential(
			nn.Linear(latent_dim * 2, n_gru_unit),
			nn.Tanh(),
			nn.Linear(n_gru_unit, self.z0_dim * 2))   # transform latent states (latent mean and std) to z0 (mean and std)
		utils.init_netw_weights(self.transform_z0)


	def forward(self, data, obs_tps):
		# data shape: n_subj x n_tp x input_dim
		# 'data' includes concatenated observations and mask

		n_subj, n_tp, n_dim = data.size()
		device = data.device

		if len(obs_tps) == 1:
			prev_y = torch.zeros((1, n_subj, self.latent_dim), device=device)   # initialize latent mean; shape: 1 x n_subj x latent_dim
			prev_std = torch.zeros((1, n_subj, self.latent_dim), device=device)   # initialize latent std; shape: 1 x n_subj x latent_dim

			xi = data[:,0,:].unsqueeze(0)   # insert a dimension of depth 1 at dimension 0; shape: 1 x n_subj x input_dim

			last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)   # if only one observation time point, use the GRU unit to update latent states
		else:
			last_yi, last_yi_std, y_traj = self.run_odernn(data, obs_tps)
			# if more than one observation time point, run ODE backwards between observation time points and use the GRU unit to combine latent states and observations

		mean_z0 = last_yi.reshape(1, n_subj, self.latent_dim)   # 1 x n_subj x latent_dim
		std_z0 = last_yi_std.reshape(1, n_subj, self.latent_dim)   # 1 x n_subj x latent_dim

		mean_z0, std_z0 = utils.split_last_dim(self.transform_z0(torch.cat((mean_z0, std_z0), -1)))
		# 1 x n_subj x (latent_dim * 2) -> 1 x n_subj x (z0_dim * 2) -> two 1 x n_subj x z0_dim

		std_z0 = std_z0.abs() 

		return mean_z0, std_z0   # shape: 1 x n_subj x z0_dim


	def run_odernn(self, data, obs_tps):
		# data shape: n_subj x n_tp x input_dim
		# 'data' includes concatenated observations and mask

		n_subj, n_tp, n_dim = data.size()
		device = data.device

		intv_len = obs_tps[-1] - obs_tps[0]   # interval length of observations
		min_step = intv_len / 100   # minimum step size for running ODE solver


		prev_y = torch.zeros((1, n_subj, self.latent_dim), device=device)   # initialize latent mean; shape: 1 x n_subj x latent_dim
		prev_std = torch.zeros((1, n_subj, self.latent_dim), device=device)   # initialize latent std; shape: 1 x n_subj x latent_dim

		prev_t, t_i = obs_tps[-1] + 0.01, obs_tps[-1]   # set the initial previous time point to the last time point + a small value

		y_traj = []
		tp_iter = range(0, len(obs_tps))
		tp_iter = reversed(tp_iter)   # running ODE backwards from the last time point to the first time point

		for ii in tp_iter:
			if (prev_t - t_i) < min_step:
				tps = torch.stack((prev_t, t_i))
				ode_sol = prev_y + self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)   # Euler step - assuming the step is small enough
				ode_sol = torch.stack((prev_y, ode_sol), 2)   # shape: 1 x n_subj x n_tp x latent_dim
			else:
				n_intm_tp = max(2, ((prev_t - t_i) / min_step).int())   # number of intermediate time points

				tps = torch.linspace(prev_t, t_i, n_intm_tp, device=device)   # intermediate linearly spaced time points to evaluate
				ode_sol = self.z0_diffeq_solver(prev_y, tps)   # solve ODE and evaluate at specified time points; shape: 1 x n_subj x n_tp x latent_dim

			yi_ode = ode_sol[:,:,-1,:]   # solution of the ODE at the end of the current time interval t_i; shape: 1 x n_subj x latent_dim
			xi = data[:,ii,:].unsqueeze(0)   # observation at t_i; shape: 1 x n_subj x input_dim
			
			yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)   # combine latent states and observation at t_i using the GRU unit; shape: 1 x n_subj x latent_dim

			prev_y, prev_std = yi, yi_std   # initial values for the next interval
			prev_t, t_i = obs_tps[ii], obs_tps[ii-1]   # start and end time for the next interval

			y_traj.append(yi)   # save the latent state trajectory


		y_traj = torch.stack(y_traj, 1)   # shape: 1 x n_tp x n_subj x latent_dim

		return yi, yi_std, y_traj


class Decoder(nn.Module):

	def __init__(self, latent_dim, input_dim):

		super(Decoder, self).__init__()

		decoder = nn.Sequential(nn.Linear(latent_dim, input_dim))   # linear transformation from latent space to data space

		utils.init_netw_weights(decoder)

		self.decoder = decoder


	def forward(self, data):

		return self.decoder(data)



