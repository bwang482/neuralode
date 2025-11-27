###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime
from scipy.sparse import csr_matrix, vstack
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import data_dict, random_state, feat_params
import utils.data_functs as utils
from utils.general_functs import save_pickle, load_pickle
from data.feature_compute import compute_features

#####################################################################################################
def parse_datasets(args, device):

	dataset_name = args.dataset
	
	##################################################################
	# RPDRml dataset

	if dataset_name == "rpdrml":
		trainpath = data_dict['train_data_file']
		testpath = data_dict['test_data_file']
		df_train = load_pickle(trainpath)
		df_test = load_pickle(testpath)
		X_train_cov, X_test_cov, cov_names, X_train_obs, X_test_obs, obs_names = compute_features(
			df_train, df_test, 
			feat_type=feat_params["feat_type"], 
			feat_transform=feat_params["feat_transform"],
			feat_lvl=feat_params["feat_lvl"], 
			max_df=feat_params["max_df"], 
			min_df=feat_params["min_df"]
		)

		time_since = []
		patient_ids = []
		labels = []
		for i, row in df_train.iterrows():
			assert( len(row['dx_dates'])==X_train_obs[i].shape[0] )
			time_since.append( [(d-row['dx_dates'][0]).days for d in row['dx_dates']] )
			patient_ids.append( row['pid'] )
			labels.append(row['label'])
			
		assert(len(time_since) == len(patient_ids))
		assert(len(patient_ids) == len(X_train_obs))

		X_train_obs = [x.toarray() for x in X_train_obs] # commented by Bo
		train_data = [
			(
				str(patient_ids[i]),
				torch.tensor(time_since[i]).to(device), # commented by Bo
				torch.from_numpy(X_train_obs[i]).to(device), # commented by Bo
				torch.tensor((X_train_obs[i] > 0).astype(float)).to(device), # commented by Bo
				torch.tensor(labels[i]).to(device)
				
			) for i in range(len(patient_ids))
		]

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(
			train_data, train_size=0.8, 
			random_state=random_state, shuffle=True
		)

		record_id, tt, vals, mask, labels = train_data[0]

		n_samples = len(train_data)
		input_dim = vals.size(-1) # commented by Bo

		batch_size = min(len(train_data), args.batch_size)
		data_min, data_max = get_data_min_max(train_data, device)

		train_dataloader = DataLoader(
			train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(
				batch, args, device, 
				data_min=data_min, data_max=data_max
			)
		)
		test_dataloader = DataLoader(
			test_data, batch_size= n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(
				batch, args, device, 
				data_min=data_min, data_max=data_max
			)
		)

		data_objects = {
			# "dataset_obj": train_dataset_obj,
			"train_dataloader": utils.inf_generator(train_dataloader), 
			"test_dataloader": utils.inf_generator(test_dataloader),
			"input_dim": input_dim,
			"n_train_batches": len(train_dataloader),
			"n_test_batches": len(test_dataloader),
			# "attr": obs_names, #optional
			# "classif_per_tp": False, #optional
			"n_labels": 1 #optional
		}

		return data_objects
	

def variable_time_collate_fn(batch, args, device, data_min=None, data_max=None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True) # commented by Bo
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)
	
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device) # commented by Bo
		vals = vals.to(device) # commented by Bo
		mask = mask.to(device) # commented by Bo
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)] ##???
		offset += len(tt)

		combined_vals[b, indices] = vals.float()
		combined_mask[b, indices] = mask.float()

		if labels is not None:
			combined_labels[b] = labels

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)
		
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args)
	return data_dict


# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records, device):
	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1] # commented by Bo
			if len(non_missing_vals) == 0: # commented by Bo
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals)) # commented by Bo
				batch_max.append(torch.max(non_missing_vals)) # commented by Bo

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max



def basic_collate_fn(batch, tps, device, args):

	data = torch.zeros([len(batch), len(tps), 1])
	labels = torch.zeros(len(batch), 1) + torch.tensor(float('nan'))   # initialize to NaN
	tps = tps.to(device)

	for bb, (ts, label) in enumerate(batch):

		data[bb] = ts
		labels[bb] = label

		data_dict = {"data": data, "tps": tps, "labels": labels}   # create data dictionary

	data_dict = utils.split_and_subsample_batch2(data_dict, args)
	# split and/or subsample train or test data
	# interpolation: condition on a subset of time points and reconstruct the full set of time points in the same time interval
	# all data -> obs_data -> subsample
	# all data -> pred_data
	# extrapolation: encode the first half of time series and reconstruct the second half
	# first half -> obs_data -> subsample
	# second half -> pred_data
	# different individuals have different observed time points

	return data_dict   # obs_data, obs_tps, data_to_pred, tps_to_pred, obs_mask, pred_mask