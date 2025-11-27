
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch.distributed as dist
from torch.utils.data import Sampler, Dataset


class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    

class CPUDistributedSampler(Sampler):
    """
    DistributedSampler that explicitly forces CPU generator usage
    regardless of default device settings.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        # Standard initialization (same as DistributedSampler)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate sizes
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Explicitly force CPU operations during critical generator creation
        if self.shuffle:
            # Create a generator explicitly on CPU
            g = torch.Generator(device='cpu')
            g.manual_seed(self.seed + self.epoch)
            
            # Use the CPU generator for randperm
            with torch.device('cpu'):
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            
        # Rest is standard DistributedSampler logic
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
            
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        
    def __len__(self):
        return self.num_samples
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def normalize_data(data, att_min, att_max):
    # we don't want to divide by zero
    att_max[ att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max


def split_and_subsample_batch(data_dict, args):
    
    if args.extrap:
        processed_dict = split_data_extrap(data_dict)
    else:
        processed_dict = split_data_interp(data_dict)


    # add mask
    processed_dict = add_mask(processed_dict)

    # Subsample time points
    if args.sample_tp is not None:
        processed_dict = subsample_obs_data(
            processed_dict, 
            n_tp_to_sample = args.sample_tp
        )

    return processed_dict


def split_data_extrap(data_dict):
    """
    Splits data into an observed part and a prediction part for extrapolation.
    The split is performed by taking the first half of the time steps as observed data
    and the second half as the time points to predict.

    Parameters:
        - data_dict: Dictionary with keys "data", "time_steps", etc.

    Returns:
        - split_dict: Dictionary containing:
            "obs_data": observed data (first half),
            "obs_tps": observed time points (first half),
            "data_to_pred": data to be predicted (second half),
            "tps_to_pred": time points for prediction (second half),
            "obs_mask" and "pred_mask": corresponding masks,
            "labels": labels if available,
            "mode": set to "extrap".
    """
    n_observed_tp = data_dict["data"].size(1) // 2

    # Split split split split
    split_dict = {"obs_data": data_dict["data"][:,:n_observed_tp,:].clone(),
                "obs_tps": data_dict["time_steps"][:n_observed_tp].clone(),
                "data_to_pred": data_dict["data"][:,n_observed_tp:,:].clone(),
                "tps_to_pred": data_dict["time_steps"][n_observed_tp:].clone()}

    split_dict["obs_mask"] = None 
    split_dict["pred_mask"] = None 
    split_dict["labels"] = None 

    # Split masks if present in data_dict
    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["obs_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["pred_mask"] = data_dict["mask"][:, n_observed_tp:].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()
            
    if ("pids" in data_dict) and (data_dict["pids"] is not None):
        split_dict["pids"] = data_dict["pids"]

    split_dict["mode"] = "extrap"
    return split_dict


def split_data_interp(data_dict):
    split_dict = {"obs_data": data_dict["data"].clone(),
                "obs_tps": data_dict["time_steps"].clone(),
                "data_to_pred": data_dict["data"].clone(),
                "tps_to_pred": data_dict["time_steps"].clone()}

    split_dict["obs_mask"] = None 
    split_dict["pred_mask"] = None 
    split_dict["labels"] = None 

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["obs_mask"] = data_dict["mask"].clone()
        split_dict["pred_mask"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()
            
    if ("pids" in data_dict) and (data_dict["pids"] is not None):
        split_dict["pids"] = data_dict["pids"]

    split_dict["mode"] = "interp"
    return split_dict


def add_mask(data_dict):

    data = data_dict["obs_data"]
    mask = data_dict["obs_mask"]

    # If no mask is provided, assume all data points are observed and create a mask of ones
    if mask is None:   # all data are observed
        mask = torch.ones_like(data).to(get_device(data))   # an array of ones with the same shape as 'data'

    data_dict["obs_mask"] = mask

    return data_dict


def subsample_obs_data(data_dict, n_tp_to_sample = None):
    """
    Subsamples the observed time points if requested.
    If n_tp_to_sample != None -> Randomly zeroes out (or removes) some time points so that the
    resulting timeline contains n_tp_to_sample points.

    Parameters:
        - data_dict
        - n_tp_to_sample: Number of time points to keep (if an integer > 1)
                        or a fraction (if <= 1) indicating the percentage to keep.

    Returns:
        - new_data_dict: Updated dictionary with subsampled "obs_data", "obs_tps", and "obs_mask".
    """

    if n_tp_to_sample is not None:
        # Randomly subsample time points
        data, time_steps, mask = subsample_tps(
            data_dict["obs_data"].clone(), 
            time_steps = data_dict["obs_tps"].clone(), 
            mask = (data_dict["obs_mask"].clone() if data_dict["obs_mask"] is not None else None),
            n_tp_to_sample = n_tp_to_sample)

    # Create a new data dict and update it with subsampled data
    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["obs_data"] = data.clone()
    new_data_dict["obs_tps"] = time_steps.clone()
    new_data_dict["obs_mask"] = mask.clone()

    return new_data_dict


def subsample_tps(data, time_steps, mask, n_tp_to_sample = None):
    """
    Subsamples time points from the data.

    Parameters:
        - data: A tensor of observed data with shape (batch_size, n_time_points, D).
        - time_steps: A tensor containing the corresponding time steps.
        - mask: A tensor containing the mask (or None if all points are observed).
        - n_tp_to_sample: The number or fraction of time points to sample.

    Returns:
        - data, time_steps, mask: The updated tensors with some time points zeroed out (i.e., removed).
    """
    # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
    if n_tp_to_sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)


    if n_tp_to_sample > 1:
        # Subsample exact number of points
        assert(n_tp_to_sample <= n_tp_in_batch)
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.size(0)):
            # Randomly choose indices that will be removed (set to zero) so that only n_tp_to_sample remain.
            # missing_idx: indices of the time points to be "removed" (i.e., not sampled).
            missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

            # Zero out the data at these indices
            data[i, missing_idx] = 0.
            # Also update the mask if it exists (set these positions to 0 indicating missing).
            if mask is not None:
                mask[i, missing_idx] = 0.
    
    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        # Subsample percentage of points from each time series
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.size(0)):
            # take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n_to_sample = int(n_tp_current * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
            tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

            # data = data.clone() # commented on 11/25/2025
            data[i, tp_to_set_to_zero] = 0. #commented on 05/31/2025 then uncommented
            if mask is not None:
                mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask


def subsample_tps2(data, tps, mask, n_tp_to_sample = None):
        # data shape: n_subj x n_tp x 1
        # mask shape: n_subj x n_tp x 1

        if n_tp_to_sample is None:
                return data, tps, mask   # no subsampling

        n_tp_in_batch = len(tps)

        if n_tp_to_sample > 1:   # subsample n_tp_to_sample time points

                n_tp_to_sample = int(n_tp_to_sample)
                assert(n_tp_to_sample <= n_tp_in_batch)

                for ii in range(data.size(0)):   # iterate over samples
                        missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))
                        # randomly sample n_tp_in_batch - n_tp_to_sample time points as missing

                        data[ii, missing_idx] = 0.   # set missing data points to zero

                        if mask is not None:
                                mask[ii, missing_idx] = 0.   # mask = 1: keep; mask = 0: mask

        elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):   # subsample percentage of points

                pct_tp_to_sample = n_tp_to_sample
                n_tp_to_sample = int(n_tp_in_batch * pct_tp_to_sample)

                for ii in range(data.size(0)):   # iterate through number of samples
                        missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

                        data[ii, missing_idx] = 0.   # set missing data points to zero

                        if mask is not None:
                                mask[ii, missing_idx] = 0.   # mask = 1: keep; mask = 0: mask

        return data, tps, mask


def inf_generator(iterable):
    # allow training with DataLoaders in an infinite loop

    iterator = iterable.__iter__()

    while True:
        try:
            yield iterator.__next__()

        except StopIteration:
            iterator = iterable.__iter__()


class ImbalancedBatchSampler(Sampler):
    """
    BatchSampler that ensures each batch has at least one positive sample
    and oversamples the minority class.
    """
    def __init__(self, dataset, batch_size, pos_label=1, oversample_ratio=1.0):
        """
        Args:
            dataset: PyTorch dataset with labels
            batch_size: Size of mini-batch
            pos_label: The positive class label (default: 1)
            oversample_ratio: Ratio of positive:negative samples (1.0 = balanced)
        """
        self.batch_size = batch_size
        
        # Extract labels from the dataset
        self.labels = []
        for i in range(len(dataset)):
            # Adjust based on your dataset structure to extract the label
            # For example: dataset[i][3] if label is the 4th element in each sample
            label = dataset[i][3]
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.labels.append(label)
        
        # Get indices of positive and negative samples
        self.pos_indices = [i for i, label in enumerate(self.labels) if label == pos_label]
        self.neg_indices = [i for i, label in enumerate(self.labels) if label != pos_label]
        
        print(f"Found {len(self.pos_indices)} positive samples and {len(self.neg_indices)} negative samples")
        
        # Calculate number of samples per batch after oversampling
        self.pos_samples_per_batch = max(1, int(batch_size * oversample_ratio / (1 + oversample_ratio)))
        self.neg_samples_per_batch = batch_size - self.pos_samples_per_batch
        
        print(f"Each batch will contain {self.pos_samples_per_batch} positive and {self.neg_samples_per_batch} negative samples")
        
        # Calculate number of batches
        self.num_batches = min(
            len(self.pos_indices) // self.pos_samples_per_batch,
            len(self.neg_indices) // self.neg_samples_per_batch
        )
        
        if self.num_batches == 0:
            self.num_batches = len(self.pos_indices)  # Fallback to at least having some batches
            print(f"Warning: Not enough samples to create balanced batches. Creating {self.num_batches} batches instead.")
    
    def __iter__(self):
        # Shuffle indices
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        random.shuffle(pos_indices)
        random.shuffle(neg_indices)
        
        # Create batches
        for i in range(self.num_batches):
            batch = []
            
            # Add positive samples (with oversampling)
            pos_to_add = min(self.pos_samples_per_batch, len(pos_indices))
            if pos_to_add < self.pos_samples_per_batch:
                # Need to oversample from existing positive samples
                batch.extend(pos_indices[:pos_to_add])  # Add all remaining positives
                # Oversample to reach the desired number
                oversampled = random.choices(
                    self.pos_indices, 
                    k=self.pos_samples_per_batch - pos_to_add
                )
                batch.extend(oversampled)
            else:
                # Have enough positive samples
                batch.extend(pos_indices[:self.pos_samples_per_batch])
                pos_indices = pos_indices[self.pos_samples_per_batch:]
            
            # Add negative samples
            neg_to_add = min(self.neg_samples_per_batch, len(neg_indices))
            batch.extend(neg_indices[:neg_to_add])
            neg_indices = neg_indices[neg_to_add:]
            
            # If we don't have enough negative samples, we can still use what we have
            if neg_to_add < self.neg_samples_per_batch:
                undersampled_batch_size = len(batch)
                print(f"Warning: Batch {i} has only {undersampled_batch_size} samples (missing negatives)")
            
            # Shuffle the batch to avoid having all positives first
            random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        return self.num_batches
    

