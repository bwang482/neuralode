###

import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

import utils.model_functs as utils

# -----

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std):
	# mu_2d: predictions
	# data_2d: observations

	n_dp = mu_2d.size()[-1]   # number of data points

	if n_dp > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_dp)), 1) ### causing OOM error!!!
		log_prob = gaussian.log_prob(data_2d)   # each row is a multidimensional sample ### causing OOM error!!!
		log_prob = log_prob / n_dp   # average across time points to account for different number of observations between samples
	else:
		log_prob = torch.zeros([1]).squeeze()

	return log_prob


# def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std):
#     var = obsrv_std ** 2
#     n_dp = mu_2d.size(-1)
    
#     if n_dp == 0:
#         return torch.zeros([1], device=mu_2d.device).squeeze()

#     residual = data_2d - mu_2d
#     log_prob = -0.5 * ((residual ** 2) / var).sum(dim=-1)
#     log_prob += -0.5 * n_dp * torch.log(2 * torch.pi * var)

#     # Average across data points
#     log_prob = log_prob / n_dp

#     return log_prob


# def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, chunk_size=5000):
#     """
#     Memory-efficient Gaussian log-likelihood computation.
    
#     Args:
#         mu_2d: Tensor [batch_size, num_points], predicted means.
#         data_2d: Tensor [batch_size, num_points], observed data.
#         obsrv_std: Scalar, observation standard deviation.
#         chunk_size: Number of points to process per chunk.

#     Returns:
#         Tensor [batch_size]: log-likelihood for each sample in batch.
#     """
#     n_dp = mu_2d.size(-1)
#     batch_size = mu_2d.size(0)

#     if n_dp == 0:
#         return torch.zeros(batch_size, device=mu_2d.device)

#     variance = obsrv_std ** 2
#     log_term = math.log(2 * math.pi) + torch.log(variance)

#     log_prob = torch.zeros(batch_size, device=mu_2d.device)

#     for start_idx in range(0, n_dp, chunk_size):
#         end_idx = min(start_idx + chunk_size, n_dp)
#         current_chunk_size = end_idx - start_idx

#         mu_chunk = mu_2d[:, start_idx:end_idx]
#         data_chunk = data_2d[:, start_idx:end_idx]

#         squared_diff = (data_chunk - mu_chunk) ** 2
#         normalized_diff_sum = squared_diff.sum(dim=-1) / variance

#         chunk_log_prob = -0.5 * (normalized_diff_sum + current_chunk_size * log_term)
#         log_prob += chunk_log_prob

#     return log_prob


def mse(mu_2d, data_2d):
        # mu_2d: predictions
        # data_2d: observations

        n_dp = mu_2d.size()[-1]   # number of data points

        if n_dp > 0:
                mse = nn.MSELoss()(mu_2d, data_2d)
        else:
                mse = torch.zeros([1]).squeeze()

        return mse


# def compute_masked_func(mu, data, mask, func):

# 	n_traj, n_subj, n_tp, n_dim = data.size()

# 	val = []
# 	for ii in range(n_traj):
# 		for kk in range(n_subj):
# 			for jj in range(n_dim):
# 				data_masked = torch.masked_select(data[ii,kk,:,jj], mask[ii,kk,:,jj].bool())
# 				mu_masked = torch.masked_select(mu[ii,kk,:,jj], mask[ii,kk,:,jj].bool())
# 				val.append(func(mu_masked, data_masked))

# 	val = torch.stack(val, 0)
# 	val = val.reshape((n_traj, n_subj, n_dim))   # n_traj x n_subj x n_dim
# 	val = torch.mean(val, -1)   # average across dims; n_traj x n_subj

# 	return val


# def masked_gaussian_log_likelihood(mu, obsrv_std, data, mask = None):
# 	# mu: predictions
# 	# data: observations

# 	if len(mu.size()) == 3:
# 		mu = mu.unsqueeze(0)   # add additional dimension for trajectory samples

# 	if len(data.size()) == 2:
# 		data = data.unsqueeze(0).unsqueeze(2)   # add additional dimensions for trajectory samples and time points
# 	elif len(data.size()) == 3:
# 		data = data.unsqueeze(0)   # add additional dimension for trajectory samples

# 	if mask is None:   # complete data
# 		n_traj, n_subj, n_tp, n_dim = mu.size()
# 		mu_flat = mu.reshape(n_traj * n_subj, n_tp * n_dim)

# 		n_traj, n_subj, n_tp, n_dim = data.size()
# 		data_flat = data.reshape(n_traj * n_subj, n_tp * n_dim)

# 		log_prob = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std = obsrv_std)
# 		log_prob = log_prob.reshape(n_traj, n_subj)   # n_traj x n_subj
# 	else:
# 		func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
# 		log_prob = compute_masked_func(mu, data, mask, func)   # n_traj x n_subj

# 	return log_prob


# def masked_mse(mu, data, mask = None):

# 	if len(mu.size()) == 3:
# 		mu = mu.unsqueeze(0)   # add additional dimension for trajectory samples

# 	if len(data.size()) == 2:
# 		data = data.unsqueeze(0).unsqueeze(2)   # add additional dimensions for trajectory samples and time points
# 	elif len(data.size()) == 3:
# 		data = data.unsqueeze(0)   # add additional dimension for trajectory samples

# 	if mask is None:
# 		n_traj, n_subj, n_tp, n_dim = mu.size()
# 		mu_flat = mu.reshape(n_traj * n_subj, n_tp * n_dim)

# 		n_traj, n_subj, n_tp, n_dim = data.size()
# 		data_flat = data.reshape(n_traj * n_subj, n_tp * n_dim)

# 		mse_loss = mse(mu_flat, data_flat)   # scalar
# 	else:
# 		mse_loss = compute_masked_func(mu, data, mask, mse)   # n_traj x n_subj
# 		mse_loss = torch.mean(mse_loss)   # scalar

# 	return mse_loss


def compute_binary_loss(label_pred, label):

	label = label.reshape(-1)   # vectorization into a row vector

	if len(label_pred.size()) == 1:
		label_pred = label_pred.unsqueeze(0)   # n_traj x n_subj
 
	n_traj = label_pred.size(0)
	label_pred = label_pred.reshape(n_traj, -1)   # n_traj x n_subj

	# assert(torch.sum(label == 0.) != 0 and torch.sum(label == 1.) != 0)

	# Check if we have both positive and negative samples
	has_positive = torch.sum(label == 1.) > 0
	has_negative = torch.sum(label == 0.) > 0

	# Handle the case where we don't have both classes
	if not (has_positive and has_negative):
		# Option 1: Return zero loss with gradient
		dummy_loss = torch.zeros(1, device=label_pred.device, requires_grad=True)
		
		# Or Option 2: Log a warning
		if not has_positive:
			print("Warning: Batch contains no positive samples (label=1)")
		if not has_negative:
			print("Warning: Batch contains no negative samples (label=0)")
			
		return dummy_loss
	
	label = label.repeat(n_traj, 1)   # n_traj x n_subj
	ce_loss = nn.BCEWithLogitsLoss()(label_pred, label)   # average loss across elements

	return ce_loss


def masked_gaussian_log_likelihood(mu, obsrv_std, data, mask = None):
	# mu: predictions
	# data: observations

	if len(mu.size()) == 3:
		mu = mu.unsqueeze(0)   # add additional dimension for trajectory samples

	if len(data.size()) == 2:
		data = data.unsqueeze(0).unsqueeze(2)   # add additional dimensions for trajectory samples and time points
	elif len(data.size()) == 3:
		data = data.unsqueeze(0)   # add additional dimension for trajectory samples

	if mask is None:   # complete data
		n_traj, n_subj, n_tp, n_dim = mu.size()
		mu_flat = mu.reshape(n_traj * n_subj, n_tp * n_dim)

		n_traj, n_subj, n_tp, n_dim = data.size()
		data_flat = data.reshape(n_traj * n_subj, n_tp * n_dim)

		log_prob = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std = obsrv_std)
		log_prob = log_prob.reshape(n_traj, n_subj)   # n_traj x n_subj
	else:
		# VECTORIZED IMPLEMENTATION
		n_traj, n_subj, n_tp, n_dim = mu.size()
		
		# Ensure mask has same shape as mu
		if len(mask.size()) == 3:
			mask = mask.unsqueeze(0).expand(n_traj, -1, -1, -1)
		
		# Apply mask
		mu_masked = mu * mask
		data_masked = data * mask
		
		# Compute squared differences
		var = obsrv_std ** 2
		diff_squared = (mu_masked - data_masked) ** 2
		
		# Sum over time and features where mask=1
		# We need to normalize by the number of observed points
		mask_sum = mask.sum(dim=[2, 3])  # Sum over time and features
		mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero
		
		# Compute log likelihood
		log_prob = -0.5 * (diff_squared / var).sum(dim=[2, 3])
		log_prob -= 0.5 * mask_sum * torch.log(2 * torch.pi * var)
		
		# Normalize by number of observed points
		log_prob = log_prob / mask_sum  # Shape: (n_traj, n_subj)

	return log_prob


def masked_mse(mu, data, mask = None):

	if len(mu.size()) == 3:
		mu = mu.unsqueeze(0)   # add additional dimension for trajectory samples

	if len(data.size()) == 2:
		data = data.unsqueeze(0).unsqueeze(2)   # add additional dimensions for trajectory samples and time points
	elif len(data.size()) == 3:
		data = data.unsqueeze(0)   # add additional dimension for trajectory samples

	if mask is None:
		n_traj, n_subj, n_tp, n_dim = mu.size()
		mu_flat = mu.reshape(n_traj * n_subj, n_tp * n_dim)

		n_traj, n_subj, n_tp, n_dim = data.size()
		data_flat = data.reshape(n_traj * n_subj, n_tp * n_dim)

		mse_loss = mse(mu_flat, data_flat)   # scalar
	else:
		# VECTORIZED IMPLEMENTATION
		n_traj, n_subj, n_tp, n_dim = mu.size()
		
		# Ensure mask has same shape as mu
		if len(mask.size()) == 3:
			mask = mask.unsqueeze(0).expand(n_traj, -1, -1, -1)
		
		# Apply mask
		mu_masked = mu * mask
		data_masked = data * mask
		
		# Compute squared differences
		diff_squared = (mu_masked - data_masked) ** 2
		
		# Sum and normalize by number of observed points
		mask_sum_total = mask.sum()
		mask_sum_total = torch.clamp(mask_sum_total, min=1)  # Avoid division by zero
		
		mse_loss = diff_squared.sum() / mask_sum_total

	return mse_loss
