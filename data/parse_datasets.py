###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np
from scipy.sparse import csr_matrix, vstack

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import random_state
import utils.data_functs as utils
from utils.general_functs import load_pickle


#####################################################################################################
def parse_datasets(args, device, train_mode="train", rank=0, world_size=1):
	"""
	Parse datasets with support for distributed training.
     
	Data Pipeline:
	1. Sparse patient data (irregular times) 
	2. Collate: Create dense batch (union of all times)
	3. Model: Process dense representation
	Note: No empty timepoints possible due to union construction

	Args:
		args: Command line arguments
		device: Device to use for tensors
		rank: Process rank for distributed training
		world_size: Total number of processes for distributed training
	"""
	from config import FEAT_DIR
	dataset_name = args.dataset.lower()
	is_distributed = world_size > 1
	
	##################################################################
	# Load features
	if dataset_name in ["cad", "mdd", "adrd"]: 
		data_dict = {
			"train_feat_file": os.path.join(FEAT_DIR, dataset_name.upper(), args.feat_dir, f"train.{args.feat_type}.{args.look_back}.pk"), 
			"val_feat_file": os.path.join(FEAT_DIR, dataset_name.upper(), args.feat_dir, f"val.{args.feat_type}.{args.look_back}.pk"), 
			"test_feat_file": os.path.join(FEAT_DIR, dataset_name.upper(), args.feat_dir, f"test.{args.feat_type}.{args.look_back}.pk"),
		}
	else:
		raise ValueError("Wrong arguments: dataset")

	print("Loading train data from", data_dict['train_feat_file'])
	print("Loading val data from", data_dict['val_feat_file'])
	print("Loading test data from", data_dict['test_feat_file'])
	train_data = load_pickle(data_dict['train_feat_file'])
	val_data = load_pickle(data_dict['val_feat_file'])
	test_data = load_pickle(data_dict['test_feat_file'])

	train_data = [
		(
			int(data[0].split('.')[0]), # patient ID
			torch.tensor(data[1]).float().to(device), # time steps
			data[2], # features
			torch.tensor(data[3]).float().to(device) # labels
		) for data in train_data
	]
	val_data = [
		(
			int(data[0].split('.')[0]), # patient ID
			torch.tensor(data[1]).float().to(device), # time steps
			data[2], # features
			torch.tensor(data[3]).float().to(device) # labels
		) for data in val_data
	]
	test_data = [
		(
			int(data[0].split('.')[0]), # patient ID
			torch.tensor(data[1]).float().to(device), # time steps
			data[2], # features
			torch.tensor(data[3]).float().to(device) # labels
		) for data in test_data
	]

	record_id, tt, vals, labels = train_data[0]
	input_dim = vals.shape[-1]
	batch_size = min(len(train_data), args.batch_size)

	print(len(train_data), "training samples")
	print(len(val_data), "validation samples")
	print(len(test_data), "test samples")
	print(input_dim, "feature dimensions")

	# Find global (training data) max time for normalization
	all_train_tt = torch.cat([d[1] for d in train_data])
	global_max_t = torch.max(all_train_tt).item()
	if global_max_t == 0:
		global_max_t = 1.0 
		
	print(f"Global time normalization constant (max_t): {global_max_t}")

	print("Setting up DataLoaders")
	train_dataloader = DataLoader(
		train_data, 
		batch_size=batch_size, shuffle=True, 
		collate_fn = lambda batch: variable_time_collate_fn_optimized(
			batch, args, device, global_max_t

		)
	)
	val_dataloader = DataLoader(
		val_data, 
		batch_size=128, shuffle=False, 
		collate_fn= lambda batch: variable_time_collate_fn_optimized(
			batch, args, device, global_max_t
		)
	)
	test_dataloader = DataLoader(
		test_data, 
		batch_size=128, shuffle=False, 
		collate_fn= lambda batch: variable_time_collate_fn_optimized(
			batch, args, device, global_max_t
		)
	)

	data_objects = {
		"train_dataloader": utils.inf_generator(train_dataloader) 
								if (not is_distributed) and (train_mode=='train') else train_dataloader,
		"val_dataloader": val_dataloader,
		"test_dataloader": test_dataloader,
		"input_dim": input_dim,
		"n_train_batches": len(train_dataloader),
		"n_val_batches": len(val_dataloader),
		"n_test_batches": len(test_dataloader),
		"n_labels": 1 
	}

	return data_objects
	

def variable_time_collate_fn(batch, args, device, global_max_t):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
	"""
	D = batch[0][2].shape[1] # feature size
	# Create a tensor of all unique time points (combined_tt) and their inverse indices
	combined_tt, inverse_indices = torch.unique(
		torch.cat([ex[1] for ex in batch]), 
		sorted=True, 
		return_inverse=True
	) # added by Bo2
	combined_tt = combined_tt.to(device)

	# Pre-allocate tensors
	N_labels = 1
	dtype = torch.float32
	combined_vals = torch.zeros([len(batch), len(combined_tt), D], device=device, dtype=dtype)
	combined_labels = torch.full((len(batch), N_labels), float('nan'), device=device, dtype=dtype)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D], device=device, dtype=dtype) #Initialize mask
	
	# Process each sample in the batch
	offset = 0
	for b, (record_id, tt, vals, labels) in enumerate(batch):
		# Get time points in this sample
		n_times = len(tt)

		# Get indices in combined_tt for this sample's time points
		indices = inverse_indices[offset:offset + n_times]
		offset += n_times

		# Process values efficiently without full dense conversion
		for i in range(vals.shape[0]):  # For each row
			# Skip if this row is empty
			if i >= len(indices):
				continue
				
			# Get time index in combined tensor
			time_idx = indices[i].item()

			# Set mask to 1 for this time point
			combined_mask[b, time_idx, :] = 1.0
			
			# Get start and end indices for this row
			row_start = vals.indptr[i]
			row_end = vals.indptr[i+1]
			
			# Process non-zero elements in this row
			for j in range(row_start, row_end):
				col_idx = vals.indices[j]
				value = vals.data[j]
				combined_vals[b, time_idx, col_idx] = value

		# Add label
		if labels is not None:
			combined_labels[b] = labels

	 # Normalize the time steps -> range(0, 1)
	if (global_max_t is not None) and (global_max_t > 0):
		combined_tt = combined_tt / global_max_t
		
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"labels": combined_labels,
		"mask": combined_mask
	}

	# Split data into obs_data and data_to_pred
	# and subsample time points (by sample_tp)
	data_dict = utils.split_and_subsample_batch(data_dict, args)
	return data_dict


def get_data_min_max(records, device=None):
    """
    Calculates min/max per feature effciently from sparse matrices.
    """ 
    n_features = records[0][2].shape[1]
    
    # Initialize arrays
    data_min = np.full(n_features, np.inf)
    data_max = np.full(n_features, -np.inf)
    
    # Count total rows and non-zero values per feature
    total_rows = 0
    feature_nonzero_counts = np.zeros(n_features, dtype=int)
    
    for b, (record_id, tt, vals, labels) in enumerate(records):
        # Add to total row count
        total_rows += vals.shape[0]
        
        # Skip empty matrices
        if vals.nnz == 0:
            continue
        
        # For each row in the matrix
        for i in range(vals.shape[0]):
            row_start = vals.indptr[i]
            row_end = vals.indptr[i+1]
            
            # Process non-zero elements in this row
            for j in range(row_start, row_end):
                col_idx = vals.indices[j]
                value = vals.data[j]
                
                # Increment the non-zero count for this feature
                feature_nonzero_counts[col_idx] += 1
                
                # Update min and max for this feature
                data_min[col_idx] = min(data_min[col_idx], value)
                data_max[col_idx] = max(data_max[col_idx], value)
    
    # Set minimum to 0 for any feature that has at least one zero value
    # (i.e., any feature with fewer non-zero occurrences than total rows)
    for i in range(n_features):
        if feature_nonzero_counts[i] < total_rows:
            data_min[i] = 0
    
    # For any feature that has no values at all, set min/max to 0
    data_min[data_min == np.inf] = 0
    data_max[data_max == -np.inf] = 0
    
    # Convert to sparse format
    data_min = csr_matrix(data_min)
    data_max = csr_matrix(data_max)
    
    return data_min, data_max


_batch_stats = {
    'count': 0,
    'min_features': float('inf'),
    'max_features': 0,
    'total_features': 0,
    'min_timepoints': float('inf'),
    'max_timepoints': 0,
    'total_timepoints': 0
}


def variable_time_collate_fn_optimized(batch, args, device, global_max_t):
    """
    Memory-efficient version that processes sparse data more carefully
    """
    global _batch_stats
    
    D = batch[0][2].shape[1]  # feature size
    patient_ids = [ex[0] for ex in batch]
    
    # Create combined timeline
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), 
        sorted=True, 
        return_inverse=True
    )
    
    # Move to device only after processing
    combined_tt = combined_tt.to(device)
    
    N_labels = 1
    dtype = torch.float32
    batch_size = len(batch)
    n_timepoints = len(combined_tt)
    
    # First, determine which features are actually used in this batch
    # This can significantly reduce memory if many features are never used
    used_features = set()
    for _, _, vals, _ in batch:
        used_features.update(vals.indices)
    used_features = sorted(list(used_features))
    n_used_features = len(used_features)
    
    # print(f"Batch using {n_used_features}/{D} features, {n_timepoints} timepoints")
    _batch_stats['count'] += 1
    _batch_stats['min_features'] = min(_batch_stats['min_features'], n_used_features)
    _batch_stats['max_features'] = max(_batch_stats['max_features'], n_used_features)
    _batch_stats['total_features'] += n_used_features
    _batch_stats['min_timepoints'] = min(_batch_stats['min_timepoints'], n_timepoints)
    _batch_stats['max_timepoints'] = max(_batch_stats['max_timepoints'], n_timepoints)
    _batch_stats['total_timepoints'] += n_timepoints
    
    # Print summary every 500 batches
    if _batch_stats['count'] % 500 == 0:
        avg_features = _batch_stats['total_features'] / _batch_stats['count']
        avg_timepoints = _batch_stats['total_timepoints'] / _batch_stats['count']
        print(f"Batch {_batch_stats['count']} summary:")
        print(f"  Features: {_batch_stats['min_features']}-{_batch_stats['max_features']} (avg: {avg_features:.1f}/{D})")
        print(f"  Timepoints: {_batch_stats['min_timepoints']}-{_batch_stats['max_timepoints']} (avg: {avg_timepoints:.1f})")
    
    # Create a mapping from original to compressed feature indices
    feat_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(used_features)}
    
    # Allocate tensors only for used features initially
    combined_vals_sparse = torch.zeros([batch_size, n_timepoints, n_used_features], 
                                       device=device, dtype=dtype)
    combined_mask_sparse = torch.zeros([batch_size, n_timepoints, n_used_features], 
                                       device=device, dtype=dtype)
    
    # Process each sample
    offset = 0
    for b, (record_id, tt, vals, labels) in enumerate(batch):
        n_times = len(tt)
        indices = inverse_indices[offset:offset + n_times]
        offset += n_times
        
        # Process sparse matrix efficiently
        for i in range(vals.shape[0]):
            if i >= len(indices):
                continue
                
            time_idx = indices[i].item()
            row_start = vals.indptr[i]
            row_end = vals.indptr[i+1]
            
            for j in range(row_start, row_end):
                orig_feat_idx = vals.indices[j]
                if orig_feat_idx in feat_mapping:
                    new_feat_idx = feat_mapping[orig_feat_idx]
                    value = vals.data[j]
                    combined_vals_sparse[b, time_idx, new_feat_idx] = value
                    combined_mask_sparse[b, time_idx, new_feat_idx] = 1.0
    
    # Now expand to full feature dimension
    # This is necessary for compatibility with the model
    combined_vals = torch.zeros([batch_size, n_timepoints, D], device=device, dtype=dtype)
    combined_mask = torch.zeros([batch_size, n_timepoints, D], device=device, dtype=dtype)
    
    # Copy sparse data to full tensors
    for new_idx, orig_idx in enumerate(used_features):
        combined_vals[:, :, orig_idx] = combined_vals_sparse[:, :, new_idx]
        combined_mask[:, :, orig_idx] = combined_mask_sparse[:, :, new_idx]
    
    # Free sparse tensors
    del combined_vals_sparse, combined_mask_sparse
    torch.cuda.empty_cache()
    
    # Handle labels
    combined_labels = torch.full((batch_size, N_labels), float('nan'), device=device, dtype=dtype)
    for b, (_, _, _, labels) in enumerate(batch):
        if labels is not None:
            combined_labels[b] = labels
    
    # Normalize time steps
    if (global_max_t is not None) and (global_max_t > 0):
        combined_tt = combined_tt / global_max_t
    
    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "labels": combined_labels,
        "mask": combined_mask,
        "pids": patient_ids
    }
    
    # Split and subsample
    data_dict = utils.split_and_subsample_batch(data_dict, args)
    return data_dict