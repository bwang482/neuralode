###

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import utils.model_functs as utils
from lib.encoder_decoder import *
from lib.likelihood import *

# -----

class VAE_Baseline(nn.Module):

	def __init__(self, z0_dim, z0_prior, obsrv_std, n_label = 1, n_unit = 100, classif = False, classif_w_recon = True):

		super(VAE_Baseline, self).__init__()

		self.obsrv_std = obsrv_std
		self.z0_prior = z0_prior
		self.n_label = n_label
		self.classif = classif
		self.classif_w_recon = classif_w_recon

		if self.classif:
			self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_unit),
					nn.ReLU(),
					nn.Linear(n_unit, n_unit),
					nn.ReLU(),
					nn.Linear(n_unit, n_label))

			utils.init_netw_weights(self.classifier)


	def get_gaussian_likelihood(self, truth, pred_y, mask = None):
		# truth: n_subj x n_tp x n_dim
		# pred_y: n_traj x n_subj x n_tp x n_dim
		# mask: n_subj x n_tp x n_dim

		n_subj, n_tp, n_dim = truth.size()

		truth_rep = truth.repeat(pred_y.size(0), 1, 1, 1)   # n_traj x n_subj x n_tp x n_dim

		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)   # n_traj x n_subj x n_tp x n_dim

		log_prob = masked_gaussian_log_likelihood(pred_y, self.obsrv_std, truth_rep, mask = mask)   # n_traj x n_subj

		log_prob = torch.mean(log_prob, 1)   # average across individuals; shape: n_traj x 1

		return log_prob


	def get_mse(self, truth, pred_y, mask = None):
		# truth: n_subj x n_tp x n_dim
		# pred_y: n_traj x n_subj x n_tp x n_dim
		# mask: n_subj x n_tp x n_dim

		n_subj, n_tp, n_dim = truth.size()

		truth_rep = truth.repeat(pred_y.size(0), 1, 1, 1)    # n_traj x n_subj x n_tp x n_dim
		
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)   # n_traj x n_subj x n_tp x n_dim

		mse = masked_mse(pred_y, truth_rep, mask = mask)   # scalar

		return mse


	def compute_all_losses(self, batch_dict, n_traj = 1, kl_coef = 1., ce_weight=100, use_joint_loss=True, z_last=False):

		pred_y, info = self.get_recons(batch_dict["tps_to_pred"], 
			batch_dict["obs_data"], batch_dict["obs_tps"], 
			mask = batch_dict["obs_mask"], n_traj = n_traj, z_last = z_last)

		fp_mu, fp_std, fp_enc = info["fp"]   # fp_mu, fp_std: 1 x n_subj xcb#.fDPc5XFn$U2 z0_dim; fp_enc: n_traj x n_subj x z0_dim
		# fp_std = fp_std.abs()
		fp_std = F.softplus(fp_std) + 1e-6 #added by bo
		fp_distr = Normal(fp_mu, fp_std) # approximate posterior distribution

		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)   # KL divergence between encoded states and prior; shape: 1 x n_subj x z0_dim
		kldiv_z0 = torch.mean(kldiv_z0, (1,2))   # average across individuals and latent dimensions; scalar

		# compute a reconstruction likelihood by comparing the observed data (batch_dict["data_to_pred"])
		# with the model's predictions (pred_y)
		recon_likelihood = self.get_gaussian_likelihood(batch_dict["data_to_pred"], pred_y, mask = batch_dict["pred_mask"])   # shape: n_traj x 1

		mse = self.get_mse(batch_dict["data_to_pred"], pred_y, mask = batch_dict["pred_mask"])

		if (batch_dict["labels"] is not None) and self.classif:
			ce_loss = compute_binary_loss(info["label_pred"], batch_dict["labels"])   # cross-entropy loss averaged across samples and trajectories
		else:
			ce_loss = torch.Tensor([0.])


		# evidence lower bound (ELBO): reconstruction likelihood - weighted KL divergence
		loss = - torch.logsumexp(recon_likelihood - kl_coef * kldiv_z0, 0)   # product across trajectories
		if torch.isnan(loss):
			loss = - torch.mean(recon_likelihood - kl_coef * kldiv_z0, 0)   # average across trajectories; scalar

		if self.classif:
			if self.classif_w_recon and use_joint_loss:
				loss = loss + ce_loss * ce_weight   # recon_loss + classification_loss * ce_weight
			else:
				loss = ce_loss   # classification_loss only


		metrics = {}
		metrics["loss"] = torch.mean(loss)
		metrics["likelihood"] = torch.mean(recon_likelihood).detach()
		metrics["ce_loss"] = torch.mean(ce_loss).detach()
		metrics["mse"] = torch.mean(mse).detach()
		metrics["kl_div"] = torch.mean(kldiv_z0).detach()
		metrics["fp_std"] = torch.mean(fp_std).detach()

		if batch_dict["labels"] is not None and self.classif:
			metrics["label_pred"] = info["label_pred"].detach()


		return metrics


	def compute_classification_loss_only(self, batch_dict, n_traj=1, z_last=False):
		"""
		Compute only classification metrics for efficient test inference.
		No reconstruction loss or MSE computed.
		"""
		
		# Use the classification-only forward pass
		_, info = self.get_classification_only(
			batch_dict["tps_to_pred"], 
			batch_dict["obs_data"], 
			batch_dict["obs_tps"], 
			mask=batch_dict["obs_mask"], 
			n_traj=n_traj, 
			z_last=z_last
		)
		
		# Only compute classification loss
		if (batch_dict["labels"] is not None) and self.classif:
			ce_loss = compute_binary_loss(info["label_pred"], batch_dict["labels"])
		else:
			ce_loss = torch.Tensor([0.])
		
		# Return minimal metrics for classification
		metrics = {}
		metrics["loss"] = torch.mean(ce_loss)
		metrics["ce_loss"] = torch.mean(ce_loss).detach()
		
		if batch_dict["labels"] is not None and self.classif:
			metrics["label_pred"] = info["label_pred"].detach()
		
		# Set unused metrics to zero (for compatibility)
		metrics["likelihood"] = torch.tensor(0.)
		metrics["mse"] = torch.tensor(0.)
		metrics["kl_div"] = torch.tensor(0.)
		metrics["fp_std"] = torch.tensor(0.)
		
		return metrics


