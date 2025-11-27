###

import time
import numpy as np

import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git   # PyTorch implementation of differentiable ODE solvers
from torchdiffeq import odeint as odeint   # interface that contains general-purpose algorithms for solving initial value problems (IVP)
# from torchdiffeq import odeint_adjoint as odeint

# -----

class DiffeqSolver(nn.Module):

	def __init__(self, ode_func, method, odeint_rtol = 1e-4, odeint_atol = 1e-5):

		super(DiffeqSolver, self).__init__()

		self.ode_func = ode_func   # any callable implementing the vector field of the ODE
		self.ode_method = method   # ODE solver

		self.odeint_rtol = odeint_rtol   # relative tolerance
		self.odeint_atol = odeint_atol   # absolute tolerance


	def forward(self, init_val, tps_to_eval):
		# init_val shape: n_traj x n_subj x n_dim

		n_traj, n_subj = init_val.size()[0], init_val.size()[1]

		pred_y = odeint(
			self.ode_func, 
			init_val, 
			tps_to_eval, 
			rtol = self.odeint_rtol, 
			atol = self.odeint_atol, 
			method = self.ode_method,
			# adjoint_rtol=self.odeint_rtol,
			# adjoint_atol=self.odeint_atol,
			# adjoint_method=self.ode_method,
			# adjoint_options={'norm': 'seminorm'},  # More efficient for adaptive solvers
			# adjoint_params=tuple(self.ode_func.parameters())  # Only compute gradients for ODE function
		)

		pred_y = pred_y.permute(1,2,0,3)   # n_traj x n_subj x n_tp x n_dim

		assert(pred_y.size()[0] == n_traj)
		assert(pred_y.size()[1] == n_subj)

		return pred_y


	def sample_traj_from_prior(self, init_val_enc, tps_to_eval):
		# init_val_enc shape: n_traj x n_subj x n_dim

		pred_y = odeint(
			self.ode_func, 
			init_val_enc, 
			tps_to_eval, 
			rtol = self.odeint_rtol, 
			atol = self.odeint_atol, 
			method = self.ode_method
		)

		pred_y = pred_y.permute(1,2,0,3)   # n_traj x n_subj x n_tp x n_dim

		return pred_y



