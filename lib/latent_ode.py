###

import time
import numpy as np

import torch
import torch.nn as nn

import utils.model_functs as utils
from lib.encoder_decoder import *
from lib.likelihood import *
from lib.base_vae import VAE_Baseline

# -----

class LatentODE(VAE_Baseline):

	def __init__(self, encoder_z0, decoder, diffeq_solver, z0_dim, z0_prior, obsrv_std,
			n_label = 1, n_unit = 100, classif = False, classif_w_recon = True):

		super(LatentODE, self).__init__(
			z0_dim = z0_dim,
			z0_prior = z0_prior,
			obsrv_std = obsrv_std,
			n_label = n_label,
			n_unit = n_unit,
			classif = classif,
			classif_w_recon = classif_w_recon)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder


	def get_recons(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):

		data_w_mask = obs_data
		if mask is not None:
			data_w_mask = torch.cat((obs_data, mask), -1)
		
		fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)   # encode data into latent representations; shape: 1 x n_subj x z0_dim
		# fp_std = fp_std.abs()		

		mean_z0 = fp_mu.repeat(n_traj, 1, 1)   # n_traj x n_subj x z0_dim
		std_z0 = fp_std.repeat(n_traj, 1, 1)   # n_traj x n_subj x z0_dim
		fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)   # sample n_traj initial values; shape: n_traj x n_subj x z0_dim

		sol_y = self.diffeq_solver(fp_enc, tps_to_pred)   # get latent states at evaluation time points; shape: n_traj x n_subj x n_tp x z0_dim

		pred_y = self.decoder(sol_y)   # decode latent states into data space; shape: n_traj x n_subj x n_tp x input_dim


		info = {"fp": (fp_mu, fp_std, fp_enc),
			"latent_traj": sol_y.detach()
		}

		if self.classif:
			if z_last:
				lp_enc = sol_y[:, :, -1, :] 
				info["label_pred"] = self.classifier(lp_enc).squeeze(-1)
			else:
				info["label_pred"] = self.classifier(fp_enc).squeeze(-1)   # n_traj x n_subj x n_label


		return pred_y, info


	def sample_traj_from_prior(self, tps_to_pred, n_traj = 1):

		init_val_enc = self.z0_prior.sample([n_traj, 1, self.z0_dim]).squeeze(-1)   # sample from z0 prior (univariate standard normal); shape: n_traj x 1 x z0_dim

		sol_y = self.diffeq_solver.sample_traj_from_prior(init_val_enc, tps_to_pred)   # n_traj x 1 x n_tp x z0_dim

		return self.decoder(sol_y)   # n_traj x 1 x n_tp x input_dim


	def get_classification_only(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):
		"""
		Classification-only inference - no reconstruction decoder called.
		"""
		
		# Encode observations to latent space (same as get_recons)
		data_w_mask = obs_data
		if mask is not None:
			data_w_mask = torch.cat((obs_data, mask), -1)
		
		fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)   # shape: 1 x n_subj x z0_dim
		
		mean_z0 = fp_mu.repeat(n_traj, 1, 1)   # n_traj x n_subj x z0_dim
		std_z0 = fp_std.repeat(n_traj, 1, 1)   # n_traj x n_subj x z0_dim
		fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)   # sample n_traj initial values
		
		# Solve ODE to get latent states
		sol_y = self.diffeq_solver(fp_enc, tps_to_pred)   # n_traj x n_subj x n_tp x z0_dim
		
		# Skip decoder - don't compute pred_y = self.decoder(sol_y)
		
		info = {"fp": (fp_mu, fp_std, fp_enc),
				"latent_traj": sol_y.detach()}
		
		# Only compute classification predictions
		if self.classif:
			if z_last:
				lp_enc = sol_y[:, :, -1, :]  # Use last time point
				info["label_pred"] = self.classifier(lp_enc).squeeze(-1)
			else:
				info["label_pred"] = self.classifier(fp_enc).squeeze(-1)   # n_traj x n_subj
		
		# Return None for pred_y since we don't compute reconstructions
		return None, info

